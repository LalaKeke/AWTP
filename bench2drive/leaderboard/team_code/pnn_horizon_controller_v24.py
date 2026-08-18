"""PNN-only low-level controller for the native 30-step rollout."""

import numpy as np


class PNNHorizonControllerV24:
    """Use short-horizon PNN feedforward with trajectory and TTC feedback."""

    def __init__(
        self,
        dt=0.1,
        max_accel=2.4,
        max_decel=4.05,
        max_steer_angle=1.066,
        max_throttle=0.75,
        wheelbase=2.85,
        min_lookahead=2.5,
        lookahead_time=0.45,
        steer_feedforward_weight=0.70,
        speed_feedback_gain=0.80,
        steer_rate_limit=0.12,
        throttle_rise_limit=0.08,
        throttle_fall_limit=0.20,
        brake_rise_limit=0.35,
        brake_release_limit=0.20,
        enable_ttc_guard=True,
    ):
        self.dt = float(dt)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_steer_angle = float(max_steer_angle)
        self.max_throttle = float(max_throttle)
        self.wheelbase = float(wheelbase)
        self.min_lookahead = float(min_lookahead)
        self.lookahead_time = float(lookahead_time)
        self.steer_feedforward_weight = float(steer_feedforward_weight)
        self.speed_feedback_gain = float(speed_feedback_gain)
        self.steer_rate_limit = float(steer_rate_limit)
        self.throttle_rise_limit = float(throttle_rise_limit)
        self.throttle_fall_limit = float(throttle_fall_limit)
        self.brake_rise_limit = float(brake_rise_limit)
        self.brake_release_limit = float(brake_release_limit)
        self.enable_ttc_guard = bool(enable_ttc_guard)
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0

    @staticmethod
    def _validate(dense_trajectory, control_horizon):
        dense = np.asarray(dense_trajectory, dtype=np.float32)
        controls = np.asarray(control_horizon, dtype=np.float32)
        if dense.shape != (30, 4):
            raise ValueError(f"dense_trajectory must be [30,4], got {dense.shape}")
        if controls.shape != (30, 2):
            raise ValueError(f"control_horizon must be [30,2], got {controls.shape}")
        if not np.isfinite(dense).all() or not np.isfinite(controls).all():
            raise ValueError("PNN trajectory and controls must be finite")
        return dense, controls

    def _path_feedback_steer(self, dense_xy, speed):
        lookahead = max(self.min_lookahead, float(speed) * self.lookahead_time)
        distance = np.linalg.norm(dense_xy, axis=-1)
        index = int(np.argmin(np.abs(distance - lookahead)))
        lateral = float(dense_xy[index, 0])
        forward = float(dense_xy[index, 1])
        length_sq = max(lateral * lateral + forward * forward, 0.25)
        curvature = 2.0 * lateral / length_sq
        wheel_angle = float(np.arctan(self.wheelbase * curvature))
        steer = float(np.clip(wheel_angle / self.max_steer_angle, -1.0, 1.0))
        return steer, index, lookahead

    @staticmethod
    def _future_points(agent):
        points = np.asarray(agent.get("future", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 2:
            return None
        return points

    def _continuous_safety_accel(self, dense_xy, speed, veh_agents, ped_agents):
        if not self.enable_ttc_guard:
            return self.max_accel, -1.0, 0.0, "none", float("inf")

        ego = dense_xy[[4, 9, 14, 19, 24, 29]]
        best_ttc = -1.0
        best_strength = 0.0
        best_kind = "none"
        best_distance = float("inf")

        for kind, agents, lat_limit, long_limit in (
            ("vehicle", veh_agents or [], 1.5, 2.8),
            ("pedestrian", ped_agents or [], 1.0, 1.8),
        ):
            for agent in agents:
                future = self._future_points(agent)
                if future is None:
                    continue
                count = min(len(ego), len(future))
                delta = future[:count] - ego[:count]
                for index in range(count):
                    lateral = abs(float(delta[index, 0]))
                    longitudinal = float(delta[index, 1])
                    distance = float(np.linalg.norm(delta[index]))
                    best_distance = min(best_distance, distance)
                    if longitudinal < -0.25:
                        continue
                    normalized = np.sqrt(
                        (lateral / lat_limit) ** 2
                        + (max(longitudinal, 0.0) / long_limit) ** 2
                    )
                    strength = float(np.clip(1.0 - normalized, 0.0, 1.0))
                    if strength <= 0.0:
                        continue
                    ttc = 0.5 * (index + 1)
                    if best_ttc < 0.0 or ttc < best_ttc or (
                        ttc == best_ttc and strength > best_strength
                    ):
                        best_ttc = ttc
                        best_strength = strength
                        best_kind = kind

        if best_ttc < 0.0:
            return self.max_accel, best_ttc, best_strength, best_kind, best_distance

        if best_ttc <= 1.5:
            stopping_time = max(best_ttc - 0.25, 0.25)
            required = -max(float(speed) - 0.2, 0.0) / stopping_time
            # Grazing predictions produce a proportional correction; deep
            # overlaps retain enough authority for emergency braking.
            gain = 0.35 + 0.65 * best_strength
            safe_accel = float(np.clip(required * gain, -self.max_decel, 0.0))
        else:
            # A distant overlap is replanned many times before it is reached.
            # Prevent acceleration but do not create a persistent early stop.
            safe_accel = 0.0
        return safe_accel, best_ttc, best_strength, best_kind, best_distance

    @staticmethod
    def _rate_limit(value, previous, rise, fall):
        lower = previous - float(fall)
        upper = previous + float(rise)
        return float(np.clip(value, lower, upper))

    def control(
        self,
        dense_trajectory,
        control_horizon,
        speed,
        target_point=None,
        veh_agents=None,
        ped_agents=None,
    ):
        dense, controls = self._validate(dense_trajectory, control_horizon)
        speed = float(np.asarray(speed).reshape(-1)[0])
        dense_xy = dense[:, :2]

        path_steer, lookahead_index, lookahead = self._path_feedback_steer(
            dense_xy, speed
        )
        # Only immediate PNN steering contributes to feedforward. Weighting the
        # full 3 s horizon turns future curvature into a premature command.
        steer_angle_short = 0.65 * float(controls[0, 1]) + 0.35 * float(
            controls[:3, 1].mean()
        )
        steer_feedforward = float(
            np.clip(-steer_angle_short / self.max_steer_angle, -1.0, 1.0)
        )
        ff_weight = self.steer_feedforward_weight
        steer_raw = ff_weight * steer_feedforward + (1.0 - ff_weight) * path_steer
        steer = self._rate_limit(
            float(np.clip(steer_raw, -1.0, 1.0)),
            self.last_steer,
            self.steer_rate_limit,
            self.steer_rate_limit,
        )

        accel_feedforward = 0.75 * float(controls[0, 0]) + 0.25 * float(
            controls[:3, 0].mean()
        )
        target_speed_03 = float(max(dense[2, 3], 0.0))
        target_speed_05 = float(max(dense[4, 3], 0.0))
        target_speed_10 = float(max(dense[9, 3], 0.0))
        speed_error = float(np.clip(target_speed_03 - speed, -1.5, 1.5))
        accel_feedback = self.speed_feedback_gain * speed_error
        accel_command = float(
            np.clip(
                accel_feedforward + accel_feedback,
                -self.max_decel,
                self.max_accel,
            )
        )

        safe_accel, risk_ttc, risk_strength, risk_kind, min_actor_distance = (
            self._continuous_safety_accel(
                dense_xy, speed, veh_agents, ped_agents
            )
        )
        accel_command = min(accel_command, safe_accel)

        if accel_command < -0.05:
            throttle_raw = 0.0
            brake_raw = float(np.clip(-accel_command / self.max_decel, 0.0, 1.0))
        else:
            throttle_raw = float(
                np.clip(
                    self.max_throttle * max(accel_command, 0.0) / self.max_accel,
                    0.0,
                    self.max_throttle,
                )
            )
            brake_raw = 0.0

        stop_intent = (
            target_speed_05 < 0.15
            and target_speed_10 < 0.30
            and accel_command <= 0.0
        )
        restart_intent = (
            speed < 0.25
            and target_speed_05 >= 0.45
            and accel_command > 0.05
            and risk_ttc < 0.0
        )
        if stop_intent and speed < 0.5:
            throttle_raw = 0.0
            brake_raw = max(brake_raw, 0.30)
        elif restart_intent:
            brake_raw = 0.0
            throttle_raw = max(throttle_raw, 0.18)

        if brake_raw > 0.05:
            throttle = 0.0
            brake = self._rate_limit(
                brake_raw,
                self.last_brake,
                self.brake_rise_limit,
                self.brake_release_limit,
            )
        else:
            brake = self._rate_limit(
                0.0,
                self.last_brake,
                self.brake_rise_limit,
                self.brake_release_limit,
            )
            if brake > 0.05:
                throttle = 0.0
            else:
                brake = 0.0
                throttle = self._rate_limit(
                    throttle_raw,
                    self.last_throttle,
                    self.throttle_rise_limit,
                    self.throttle_fall_limit,
                )

        self.last_steer = steer
        self.last_throttle = throttle
        self.last_brake = brake

        metadata = {
            "controller": "pnn_dense_receding_horizon_v2.4",
            "pnn_v24_steer_feedforward": steer_feedforward,
            "pnn_v24_path_steer": path_steer,
            "pnn_v24_steer_raw": steer_raw,
            "pnn_v24_lookahead_index": lookahead_index,
            "pnn_v24_lookahead_m": lookahead,
            "pnn_v24_accel_feedforward": accel_feedforward,
            "pnn_v24_accel_feedback": accel_feedback,
            "pnn_v24_accel_command": accel_command,
            "pnn_v24_target_speed_03": target_speed_03,
            "pnn_v24_target_speed_05": target_speed_05,
            "pnn_v24_target_speed_10": target_speed_10,
            "pnn_v24_risk_ttc": risk_ttc,
            "pnn_v24_risk_strength": risk_strength,
            "pnn_v24_risk_kind": risk_kind,
            "pnn_v24_min_actor_distance": min_actor_distance,
            "pnn_v24_stop_intent": stop_intent,
            "pnn_v24_restart_intent": restart_intent,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
        }
        return steer, throttle, brake, metadata
