#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np


MAP_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAP_ROOT))

from pnn_temporal_alignment import ALIGNMENT_VERSION, align_hipad_motion_future


def test_constant_velocity_is_preserved():
    future = np.stack([np.array([0.0, 0.2 * i]) for i in range(1, 7)])
    aligned = align_hipad_motion_future(
        (0.0, 0.0), future, measured_velocity_xy=(0.0, 2.0)
    )
    np.testing.assert_allclose(aligned[:, 1], [1, 2, 3, 4, 5, 6], atol=1e-4)


def test_stationary_actor_jitter_does_not_become_motion():
    future = np.array(
        [[0.08, 0.12], [-0.06, 0.22], [0.05, 0.28], [-0.04, 0.35], [0.03, 0.42], [0.0, 0.48]],
        dtype=np.float32,
    )
    aligned = align_hipad_motion_future(
        (0.0, 0.0), future, measured_velocity_xy=(0.0, 0.05)
    )
    assert np.linalg.norm(aligned[-1] - aligned[0]) < 0.5


def test_hard_brake_low_speed_tail_is_bounded():
    # Representative failure: a 0.45 m/s vehicle was previously extrapolated
    # from y=5.15 m to about y=51 m at 3 s because of noisy 10 Hz points.
    current = np.array([-0.07, 5.145], dtype=np.float32)
    noisy_future = np.array(
        [[-0.4, 6.5], [-0.8, 8.0], [-1.0, 9.5], [-1.2, 11.0], [-1.3, 12.5], [-1.5, 14.0]],
        dtype=np.float32,
    )
    aligned = align_hipad_motion_future(
        current,
        noisy_future,
        measured_velocity_xy=(0.0, 0.45),
        max_speed=20.0,
    )
    displacement = float(np.linalg.norm(aligned[-1] - current))
    assert displacement < 3.0, (ALIGNMENT_VERSION, aligned[-1], displacement)

