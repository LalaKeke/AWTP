import numpy as np

from team_code.pid_controller import PID


class PNNHorizonController:
    """Track PNN's full control horizon instead of reducing it to six points."""

    def __init__(
        self,
        trajectory_pid=None,
        dt=0.1,
        accel_decay_steps=6.0,
        steer_feedforward_blend=0.35,
        max_accel=2.4,
        max_decel=4.05,
        max_steer_angle=1.066,
        max_throttle=0.75,
        enable_dynamic_guard=True,
        use_trajectory_pid=False,
        wheelbase=2.85,
        min_lookahead=2.5,
        lookahead_time=0.55,
        rolling_throttle=0.10,
        restart_throttle=0.22,
        restart_speed_threshold=0.35,
        restart_target_speed=0.45,
        restart_accel_threshold=0.10,
        stop_target_speed=0.45,
        stop_accel_ceiling=0.0,
        hard_guard_steps=3,
        long_risk_throttle_cap=0.25,
        guard_rear_tolerance=0.25,
        guard_release_decay=0.55,
        enable_ttc_guard=False,
        ttc_hard_seconds=1.25,
        ttc_soft_seconds=2.25,
        ttc_safety_buffer=2.5,
        ttc_low_speed_release=0.30,
        ttc_creep_clearance=5.0,
    ):
        self.trajectory_pid = trajectory_pid
        self.dt = float(dt)
        self.accel_decay_steps = float(accel_decay_steps)
        self.steer_feedforward_blend = float(steer_feedforward_blend)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.max_steer_angle = float(max_steer_angle)
        self.max_throttle = float(max_throttle)
        self.enable_dynamic_guard = bool(enable_dynamic_guard)
        self.use_trajectory_pid = bool(use_trajectory_pid)
        self.wheelbase = float(wheelbase)
        self.min_lookahead = float(min_lookahead)
        self.lookahead_time = float(lookahead_time)
        self.rolling_throttle = float(rolling_throttle)
        self.restart_throttle = float(restart_throttle)
        self.restart_speed_threshold = float(restart_speed_threshold)
        self.restart_target_speed = float(restart_target_speed)
        self.restart_accel_threshold = float(restart_accel_threshold)
        self.stop_target_speed = float(stop_target_speed)
        self.stop_accel_ceiling = float(stop_accel_ceiling)
        self.hard_guard_steps = int(hard_guard_steps)
        self.long_risk_throttle_cap = float(long_risk_throttle_cap)
        self.guard_rear_tolerance = float(guard_rear_tolerance)
        self.guard_release_decay = float(guard_release_decay)
        self.enable_ttc_guard = bool(enable_ttc_guard)
        self.ttc_hard_seconds = float(ttc_hard_seconds)
        self.ttc_soft_seconds = float(ttc_soft_seconds)
        self.ttc_safety_buffer = float(ttc_safety_buffer)
        self.ttc_low_speed_release = float(ttc_low_speed_release)
        self.ttc_creep_clearance = float(ttc_creep_clearance)
        self._brake_guard_state = 0.0
        self._guard_diagnostics = {}
        self.speed_controller = PID(K_P=0.35, K_I=0.05, K_D=0.05, n=10)

    def _dense_path_steer(self, dense_xy, speed):
        """Pure-pursuit steering from the native 30-point PNN rollout.

        HiP-AD trajectory coordinates are [lateral, forward]. CARLA steer is
        normalized, while the bicycle model uses a front-wheel angle in rad.
        """
        lookahead = max(self.min_lookahead, float(speed) * self.lookahead_time)
        distance = np.linalg.norm(dense_xy, axis=-1)
        index = int(np.argmin(np.abs(distance - lookahead)))
        lateral = float(dense_xy[index, 0])
        forward = float(dense_xy[index, 1])
        length_sq = max(lateral * lateral + forward * forward, 0.25)
        curvature = 2.0 * lateral / length_sq
        wheel_angle = float(np.arctan(self.wheelbase * curvature))
        steer = float(np.clip(wheel_angle / self.max_steer_angle, -1.0, 1.0))
        return steer, index, lookahead, wheel_angle

    @staticmethod
    def _future_points(agent):
        points = np.asarray(agent.get("future", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 2:
            return None
        return points

    def _dynamic_brake_floor(self, dense_xy, veh_agents, ped_agents, speed=None):
        """Return a conservative brake floor from six aligned future checks."""
        if not self.enable_dynamic_guard:
            return 0.0, False, float("inf"), -1.0, "none"

        ego = dense_xy[[4, 9, 14, 19, 24, 29]]
        best_floor = 0.0
        long_risk = False
        best_distance = float("inf")
        best_time = -1.0
        best_kind = "none"
        brake_by_step = (0.85, 0.65, 0.45, 0.30, 0.18, 0.10)
        best_overlap_time = float("inf")
        best_ttc = float("inf")
        best_required_decel = 0.0
        low_speed_release = False
        speed = 0.0 if speed is None else max(float(speed), 0.0)

        for kind, agents, lateral_limit, longitudinal_limit in (
            ("vehicle", veh_agents or [], 1.8, 3.5),
            ("pedestrian", ped_agents or [], 1.2, 2.2),
        ):
            for agent in agents:
                future = self._future_points(agent)
                if future is None:
                    continue
                count = min(len(ego), len(future))
                delta = future[:count] - ego[:count]
                distance = np.linalg.norm(delta, axis=-1)
                overlap_indices = []
                for index in range(count):
                    if distance[index] < best_distance:
                        best_distance = float(distance[index])
                        best_time = 0.5 * (index + 1)
                        best_kind = kind
                    # HiP-AD/PNN local coordinates are lateral x, forward y.
                    longitudinal = float(delta[index, 1])
                    if (
                        abs(float(delta[index, 0])) < lateral_limit
                        and -self.guard_rear_tolerance < longitudinal < longitudinal_limit
                    ):
                        overlap_indices.append(index)
                        if not self.enable_ttc_guard:
                            if index < self.hard_guard_steps:
                                best_floor = max(best_floor, brake_by_step[index])
                            else:
                                long_risk = True

                if self.enable_ttc_guard and overlap_indices:
                    first_index = overlap_indices[0]
                    overlap_time = 0.5 * (first_index + 1)
                    best_overlap_time = min(best_overlap_time, overlap_time)

                    current = np.asarray(
                        [agent.get("x", future[0, 0]), agent.get("y", future[0, 1])],
                        dtype=np.float32,
                    )
                    actor_forward_speed = float(future[0, 1] - current[1]) / 0.5
                    closing_speed = max(speed - actor_forward_speed, 0.0)
                    current_forward = float(current[1])
                    current_distance = float(np.linalg.norm(current))
                    available_distance = max(
                        current_forward - self.ttc_safety_buffer,
                        0.25,
                    )
                    ttc = (
                        available_distance / closing_speed
                        if closing_speed > 0.1 and current_forward > 0.0
                        else float("inf")
                    )
                    required_decel = (
                        closing_speed * closing_speed / (2.0 * available_distance)
                        if closing_speed > 0.1 and current_forward > 0.0
                        else 0.0
                    )
                    best_ttc = min(best_ttc, ttc)
                    best_required_decel = max(best_required_decel, required_decel)

                    immediate_current_risk = (
                        abs(float(current[0])) < lateral_limit
                        and -self.guard_rear_tolerance < current_forward
                        < self.ttc_safety_buffer + 1.0
                    )
                    can_creep = (
                        speed < self.ttc_low_speed_release
                        and current_distance > self.ttc_creep_clearance
                        and not immediate_current_risk
                        and (not np.isfinite(ttc) or ttc > self.ttc_soft_seconds)
                    )
                    if can_creep:
                        low_speed_release = True
                        long_risk = True
                        continue

                    effective_ttc = min(ttc, overlap_time)
                    if effective_ttc <= self.ttc_hard_seconds:
                        ttc_floor = max(0.45, min(0.90, required_decel / self.max_decel))
                        best_floor = max(best_floor, ttc_floor)
                    elif effective_ttc <= self.ttc_soft_seconds:
                        ttc_floor = max(0.12, min(0.45, required_decel / self.max_decel))
                        best_floor = max(best_floor, ttc_floor)
                    else:
                        long_risk = True

        # Apply a short release filter to avoid brake chatter, while allowing
        # the vehicle to restart quickly once the predicted overlap clears.
        if low_speed_release and best_floor <= 0.0:
            self._brake_guard_state = 0.0
        elif best_floor >= self._brake_guard_state:
            self._brake_guard_state = best_floor
        else:
            self._brake_guard_state *= self.guard_release_decay
            if self._brake_guard_state < 0.05:
                self._brake_guard_state = 0.0
        self._guard_diagnostics = {
            "pnn_dynamic_overlap_time": best_overlap_time,
            "pnn_dynamic_ttc": best_ttc,
            "pnn_dynamic_required_decel": best_required_decel,
            "pnn_dynamic_low_speed_release": low_speed_release,
            "pnn_dynamic_ttc_guard": self.enable_ttc_guard,
        }
        return self._brake_guard_state, long_risk, best_distance, best_time, best_kind

    def guard_pid_control(
        self,
        dense_trajectory,
        throttle,
        brake,
        veh_agents=None,
        ped_agents=None,
        speed=None,
    ):
        """Apply only the dynamic safety override to an external PID command."""
        dense = np.asarray(dense_trajectory, dtype=np.float32)
        if dense.shape != (30, 4):
            raise ValueError(f"dense_trajectory must be [30,4], got {dense.shape}")

        (
            brake_floor,
            long_risk,
            min_actor_distance,
            risk_time,
            risk_kind,
        ) = self._dynamic_brake_floor(dense[:, :2], veh_agents, ped_agents, speed=speed)

        throttle = float(throttle)
        brake = float(brake)
        if brake_floor > brake:
            brake = brake_floor
            throttle = 0.0
        elif long_risk and brake <= 0.0:
            throttle = min(throttle, self.long_risk_throttle_cap)

        metadata = {
            "controller": "pnn_hipad_pid_hybrid_v1",
            "pnn_dynamic_brake_floor": brake_floor,
            "pnn_dynamic_long_risk": long_risk,
            "pnn_dynamic_min_distance": min_actor_distance,
            "pnn_dynamic_risk_time": risk_time,
            "pnn_dynamic_risk_kind": risk_kind,
        }
        metadata.update(self._guard_diagnostics)
        return throttle, brake, metadata

    def control(
        self,
        dense_trajectory,
        control_horizon,
        speed,
        target_point,
        veh_agents=None,
        ped_agents=None,
    ):
        dense = np.asarray(dense_trajectory, dtype=np.float32)
        controls = np.asarray(control_horizon, dtype=np.float32)
        if dense.shape != (30, 4):
            raise ValueError(f"dense_trajectory must be [30,4], got {dense.shape}")
        if controls.shape != (30, 2):
            raise ValueError(f"control_horizon must be [30,2], got {controls.shape}")

        dense_xy = dense[:, :2]
        speed = float(np.asarray(speed).reshape(-1)[0])

        # Dense path feedback uses the native 30-point rollout. The default
        # path is independent of the legacy six-point PID controller.
        steer_dense, lookahead_index, lookahead, wheel_angle = self._dense_path_steer(
            dense_xy, speed
        )
        if self.use_trajectory_pid:
            if self.trajectory_pid is None:
                raise RuntimeError("trajectory_pid is required when use_trajectory_pid=True")
            steer_pid, _, _, pid_metadata = self.trajectory_pid.control_pid(
                dense_xy,
                dense_xy,
                np.asarray(speed, dtype=np.float32),
                np.asarray(target_point, dtype=np.float32),
            )
        else:
            steer_pid = steer_dense
            pid_metadata = {}

        indices = np.arange(30, dtype=np.float32)
        weights = np.exp(-indices / max(self.accel_decay_steps, 1e-3))
        weights /= weights.sum()
        near_weights = weights[:5] / weights[:5].sum()

        accel = controls[:, 0]
        steer_angle = controls[:, 1]
        accel_near = float(np.sum(accel[:5] * near_weights))
        accel_all = float(np.sum(accel * weights))
        # The first action is the receding-horizon command; later actions add
        # anticipation without overwhelming the immediate policy decision.
        accel_command = 0.55 * float(accel[0]) + 0.30 * accel_near + 0.15 * accel_all

        future_times = self.dt * (indices + 1.0)
        required_accel = (dense[:, 3] - speed) / future_times
        forecast_decel = float(np.min(required_accel))
        if forecast_decel < -0.1:
            accel_command = min(accel_command, 0.8 * forecast_decel)
        accel_command = float(np.clip(accel_command, -self.max_decel, self.max_accel))

        target_speed_05 = float(max(dense[4, 3], 0.0))
        target_speed_10 = float(max(dense[9, 3], 0.0))
        speed_error = float(np.clip(target_speed_05 - speed, -2.0, 2.0))
        speed_feedback = float(self.speed_controller.step(speed_error))

        (
            brake_floor,
            long_risk,
            min_actor_distance,
            risk_time,
            risk_kind,
        ) = self._dynamic_brake_floor(
            dense_xy,
            veh_agents,
            ped_agents,
            speed=speed,
        )
        if accel_command < -0.1:
            brake = float(np.clip(-accel_command / self.max_decel, 0.08, 1.0))
            throttle = 0.0
        else:
            accel_feedforward = max(accel_command, 0.0) / self.max_accel
            motion_feedforward = (
                self.rolling_throttle * np.clip(target_speed_05 / 5.0, 0.0, 1.5)
            )
            throttle = float(
                np.clip(
                    motion_feedforward
                    + 0.55 * accel_feedforward
                    + 0.45 * max(speed_feedback, 0.0),
                    0.0,
                    self.max_throttle,
                )
            )
            brake = 0.0

        if brake_floor > brake:
            brake = brake_floor
            throttle = 0.0

        # A short-horizon low speed alone is not a stop intent: from rest, a
        # physically bounded rollout naturally travels slowly during its first
        # 0.5 s. Require the longer horizon and acceleration command to agree.
        stop_intent = (
            target_speed_05 < 0.25
            and target_speed_10 < self.stop_target_speed
            and accel_command <= self.stop_accel_ceiling
            and speed < 0.4
            and brake_floor <= 0.05
        )
        restart_intent = (
            speed < self.restart_speed_threshold
            and brake_floor <= 0.05
            and (
                target_speed_10 >= self.restart_target_speed
                or accel_command >= self.restart_accel_threshold
            )
        )
        if stop_intent:
            brake = max(brake, 0.35)
            throttle = 0.0
        elif restart_intent:
            brake = 0.0
            throttle = max(throttle, self.restart_throttle)

        # At 20 Hz the policy will replan many times before a 2-3 s overlap.
        # Preserve progress now and cap throttle instead of turning a distant
        # possibility into a persistent brake deadlock.
        if long_risk and brake_floor <= 0.05 and brake <= 0.0:
            throttle = min(throttle, self.long_risk_throttle_cap)

        steer_weighted = float(np.sum(steer_angle * weights))
        # PNN steering is defined in pnn_xy; HiP-AD/CARLA steering has the
        # opposite sign after trajectory conversion.
        steer_feedforward = float(
            np.clip(-steer_weighted / self.max_steer_angle, -1.0, 1.0)
        )
        blend = self.steer_feedforward_blend
        steer = float(
            np.clip((1.0 - blend) * float(steer_pid) + blend * steer_feedforward, -1.0, 1.0)
        )

        metadata = dict(pid_metadata)
        metadata.update(
            {
                "controller": "pnn_dense_receding_horizon_v2.3",
                "pnn_uses_legacy_pid": self.use_trajectory_pid,
                "pnn_dense_steer": steer_dense,
                "pnn_dense_lookahead_index": lookahead_index,
                "pnn_dense_lookahead_m": lookahead,
                "pnn_dense_wheel_angle_rad": wheel_angle,
                "pnn_accel_first": float(accel[0]),
                "pnn_accel_near": accel_near,
                "pnn_accel_all": accel_all,
                "pnn_forecast_decel": forecast_decel,
                "pnn_accel_command": accel_command,
                "pnn_target_speed_05": target_speed_05,
                "pnn_target_speed_10": target_speed_10,
                "pnn_speed_feedback": speed_feedback,
                "pnn_steer_feedforward": steer_feedforward,
                "pnn_steer_pid": float(steer_pid),
                "pnn_dynamic_brake_floor": brake_floor,
                "pnn_dynamic_long_risk": long_risk,
                "pnn_dynamic_min_distance": min_actor_distance,
                "pnn_dynamic_risk_time": risk_time,
                "pnn_dynamic_risk_kind": risk_kind,
                "pnn_stop_intent": stop_intent,
                "pnn_restart_intent": restart_intent,
                "steer": steer,
                "throttle": throttle,
                "brake": brake,
            }
        )
        metadata.update(self._guard_diagnostics)
        return steer, throttle, brake, metadata
