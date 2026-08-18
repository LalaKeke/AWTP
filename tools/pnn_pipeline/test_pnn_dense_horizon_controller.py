import unittest

import numpy as np

from team_code.pnn_horizon_controller import PNNHorizonController


class DenseHorizonControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = PNNHorizonController(
            trajectory_pid=None,
            use_trajectory_pid=False,
            enable_dynamic_guard=True,
        )

    @staticmethod
    def trajectory(lateral=0.0, speed=5.0):
        t = np.arange(1, 31, dtype=np.float32) * 0.1
        out = np.zeros((30, 4), dtype=np.float32)
        out[:, 0] = lateral * t / 3.0
        out[:, 1] = speed * t
        out[:, 3] = speed
        return out

    def test_straight_path_has_small_steer(self):
        steer, throttle, brake, meta = self.controller.control(
            self.trajectory(), np.zeros((30, 2), np.float32), 5.0, [0.0, 10.0]
        )
        self.assertLess(abs(steer), 1e-5)
        self.assertGreaterEqual(throttle, 0.0)
        self.assertEqual(brake, 0.0)
        self.assertFalse(meta["pnn_uses_legacy_pid"])

    def test_commands_are_bounded(self):
        controls = np.tile(np.array([10.0, 1.066], np.float32), (30, 1))
        steer, throttle, brake, _ = self.controller.control(
            self.trajectory(lateral=2.0, speed=8.0), controls, 2.0, [0.0, 10.0]
        )
        self.assertLessEqual(abs(steer), 1.0)
        self.assertLessEqual(throttle, 0.75)
        self.assertGreaterEqual(brake, 0.0)

    def test_predicted_actor_overlap_forces_brake(self):
        trajectory = self.trajectory(speed=5.0)
        future = trajectory[[4, 9, 14, 19, 24, 29], :2]
        agent = {"future": future.copy()}
        _, throttle, brake, meta = self.controller.control(
            trajectory,
            np.tile(np.array([2.0, 0.0], np.float32), (30, 1)),
            5.0,
            [0.0, 10.0],
            veh_agents=[agent],
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreaterEqual(brake, 0.85)
        self.assertEqual(meta["pnn_dynamic_risk_kind"], "vehicle")

    def test_positive_acceleration_restarts_slow_rollout(self):
        # Regression for RouteScenario_1956: v2.1 applied brake=0.35 even
        # though PNN requested positive acceleration and no guard was active.
        trajectory = self.trajectory(speed=0.0)
        trajectory[:, 1] = np.linspace(0.01, 0.9, 30, dtype=np.float32)
        trajectory[:, 3] = np.linspace(0.03, 0.52, 30, dtype=np.float32)
        controls = np.tile(np.array([0.32, 0.0], np.float32), (30, 1))
        _, throttle, brake, meta = self.controller.control(
            trajectory, controls, 0.0, [0.0, 10.0]
        )
        self.assertEqual(brake, 0.0)
        self.assertGreaterEqual(throttle, self.controller.restart_throttle)
        self.assertTrue(meta["pnn_restart_intent"])
        self.assertFalse(meta["pnn_stop_intent"])

    def test_nonpositive_low_speed_rollout_keeps_stop(self):
        trajectory = self.trajectory(speed=0.0)
        trajectory[:, 1] = 0.0
        trajectory[:, 3] = 0.0
        controls = np.tile(np.array([-0.2, 0.0], np.float32), (30, 1))
        _, throttle, brake, meta = self.controller.control(
            trajectory, controls, 0.0, [0.0, 10.0]
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreaterEqual(brake, 0.35)
        self.assertTrue(meta["pnn_stop_intent"])
        self.assertFalse(meta["pnn_restart_intent"])

    def test_dynamic_guard_has_priority_over_restart(self):
        trajectory = self.trajectory(speed=0.5)
        controls = np.tile(np.array([0.5, 0.0], np.float32), (30, 1))
        future = trajectory[[4, 9, 14, 19, 24, 29], :2]
        _, throttle, brake, meta = self.controller.control(
            trajectory,
            controls,
            0.0,
            [0.0, 10.0],
            veh_agents=[{"future": future.copy()}],
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreaterEqual(brake, 0.85)
        self.assertFalse(meta["pnn_restart_intent"])

    def test_long_horizon_overlap_caps_throttle_without_braking(self):
        trajectory = self.trajectory(speed=3.0)
        controls = np.tile(np.array([1.0, 0.0], np.float32), (30, 1))
        future = trajectory[[4, 9, 14, 19, 24, 29], :2].copy()
        # No overlap through 1.5 s; overlap begins only at 2.0 s.
        future[:3, 0] += 5.0
        _, throttle, brake, meta = self.controller.control(
            trajectory,
            controls,
            1.0,
            [0.0, 10.0],
            veh_agents=[{"future": future}],
        )
        self.assertEqual(brake, 0.0)
        self.assertGreater(throttle, 0.0)
        self.assertLessEqual(throttle, self.controller.long_risk_throttle_cap)
        self.assertTrue(meta["pnn_dynamic_long_risk"])


if __name__ == "__main__":
    unittest.main()
