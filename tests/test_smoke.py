import numpy as np
import torch

from hipad_pnn_adapter import hipad_points_to_pnn, pnn_points_to_hipad
from pnn_temporal_alignment import ALIGNMENT_VERSION, align_hipad_motion_future
from PCC_helpers_v8 import EluTimeControlEnhanced


def test_coordinate_round_trip():
    points = np.array([[1.0, 5.0], [-2.0, 8.0]], dtype=np.float32)
    restored = pnn_points_to_hipad(hipad_points_to_pnn(points))
    np.testing.assert_allclose(restored, points)


def test_temporal_alignment_contract():
    short_future = np.stack(
        [np.array([0.1 * i, 0.0], dtype=np.float32) for i in range(1, 7)]
    )
    aligned = align_hipad_motion_future(
        [0.0, 0.0], short_future, measured_velocity_xy=[1.0, 0.0]
    )
    assert ALIGNMENT_VERSION
    assert aligned.shape == (6, 2)
    assert np.isfinite(aligned).all()


def test_controlnet_forward_shapes():
    batch = 2
    model = EluTimeControlEnhanced(future_steps=30).eval()
    with torch.no_grad():
        ego, ped, veh = model(
            torch.zeros(batch, 10),
            torch.zeros(batch, 10, 6),
            torch.zeros(batch, 10, 6),
            torch.zeros(batch, 2, 20, 2),
            torch.zeros(batch, 10, dtype=torch.bool),
            torch.zeros(batch, 10, dtype=torch.bool),
        )
    assert ego.shape == (batch, 30, 2)
    assert ped.shape == (batch, 10, 30, 2)
    assert veh.shape == (batch, 10, 30, 2)
    assert torch.isfinite(ego).all()
