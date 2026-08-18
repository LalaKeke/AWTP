#!/usr/bin/env python3
"""
Standalone adapter between HiP-AD Bench2Drive planning outputs and the
PNN_EV_v12 differentiable trajectory optimizer.

This file intentionally does not modify either project. It can be imported from
a wrapper Bench2Drive agent, or used offline to build PNN-style .pt datasets.

Coordinate convention
---------------------
HiP-AD's closed-loop PID consumes planning points in an ego-local BEV frame.
The existing PNN bicycle model uses:

    dx = v * cos(theta), dy = v * sin(theta)

For compatibility with the current PNN training data and HiP-AD PID behavior,
the default ``hipad_xy`` bridge assumes the local frame has x as right/lateral
and y as forward, so the ego starts at [0, 0] with heading pi/2. The optional
``pnn_xy`` bridge converts to PNN's more conventional x-forward/y-left frame
with ego heading 0, then converts outputs back to HiP-AD coordinates.

PNN state layout, matching the thesis Section 2 and train_v10.py:

    ego_state:  [x, y, theta, v, xr1, yr1, xr2, yr2, xr3, yr3]
    agent_state:[x, y, theta, v, xr, yr]
    lane_points:[num_lanes, num_points, 2], first two lanes are boundaries
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HIPAD_PNN_ROOT", Path(__file__).resolve().parent)
).resolve()
DEFAULT_MAP_ROOT = os.environ.get("MAP_ROOT", str(DEFAULT_PROJECT_ROOT.parent))
DEFAULT_PNN_ROOT = os.environ.get(
    "PNN_ROOT",
    str(DEFAULT_PROJECT_ROOT / "pnn"),
)
DEFAULT_PNN_MAIN = os.path.join(DEFAULT_PNN_ROOT, "Main")
DEFAULT_STATS_PATH = str(DEFAULT_PROJECT_ROOT / "checkpoints" / "pnn_stats.pt")
DEFAULT_CONTROL_CKPT = str(DEFAULT_PROJECT_ROOT / "checkpoints" / "pnn_control.pth")
DEFAULT_WEIGHT_CKPT = os.path.join(
    DEFAULT_PROJECT_ROOT,
    "checkpoints",
    "pnn_weight.pth",
)

DT = 0.1
TRAJ_LEN = 30
NUM_PEDS = 10
NUM_VEHS = 10
NUM_LANES = 10
LANE_POINTS = 20
NUM_COSTS = 8
DEFAULT_LANE_WIDTH = 3.8
DEFAULT_ROUTE_TARGET_TIMES = (1.0, 2.0, 3.0)
DEFAULT_NAV_MIN_SPEED = 1.0
DEFAULT_NAV_MAX_SPEED = 15.0


def parse_float_sequence(value: Any, name: str = "value") -> Optional[List[float]]:
    """Parse a comma/space separated float sequence.

    Empty values return None so callers can distinguish "not configured" from
    an intentionally provided list.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        return [float(p) for p in parts]
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return [float(x) for x in arr.tolist()]
    return [float(value)]


def _per_target_scales(distance_scale: Any, target_count: int) -> np.ndarray:
    values = parse_float_sequence(distance_scale, name="distance_scale")
    if values is None:
        values = [1.0]
    if len(values) == 1:
        return np.full((target_count,), float(values[0]), dtype=np.float32)
    if len(values) != target_count:
        raise ValueError(
            f"distance_scale must be scalar or contain {target_count} values, got {values}"
        )
    return np.asarray(values, dtype=np.float32)


def _as_np(x: Any, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=dtype)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _ensure_2d_points(points: Any, name: str) -> np.ndarray:
    arr = _as_np(points)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [T,2], got {arr.shape}")
    return arr.astype(np.float32)


def _angle_wrap(theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta), np.cos(theta)).astype(np.float32)


def hipad_points_to_pnn(points: Any) -> np.ndarray:
    """Convert HiP-AD [right_x, forward_y] to PNN [forward_x, left_y]."""
    pts = np.asarray(points, dtype=np.float32)
    return np.stack([pts[..., 1], -pts[..., 0]], axis=-1).astype(np.float32)


def pnn_points_to_hipad(points: Any) -> np.ndarray:
    """Convert PNN [forward_x, left_y] back to HiP-AD [right_x, forward_y]."""
    pts = np.asarray(points, dtype=np.float32)
    return np.stack([-pts[..., 1], pts[..., 0]], axis=-1).astype(np.float32)


def hipad_yaw_to_pnn(yaw: Any) -> np.ndarray:
    """Convert heading under HiP-AD [right, fwd] axes to PNN [fwd, left] axes."""
    return _angle_wrap(np.asarray(yaw, dtype=np.float32) - np.pi / 2)


def pnn_yaw_to_hipad(yaw: Any) -> np.ndarray:
    """Inverse heading conversion for pnn_xy -> hipad_xy."""
    return _angle_wrap(np.asarray(yaw, dtype=np.float32) + np.pi / 2)


def agent_states_hipad_to_pnn(states: np.ndarray) -> np.ndarray:
    out = np.asarray(states, dtype=np.float32).copy()
    out[..., 0:2] = hipad_points_to_pnn(out[..., 0:2])
    out[..., 2] = hipad_yaw_to_pnn(out[..., 2])
    out[..., 4:6] = hipad_points_to_pnn(out[..., 4:6])
    return out


def traj_pnn_to_hipad(traj: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    x_pnn = out[..., 0].clone()
    y_pnn = out[..., 1].clone()
    out[..., 0] = -y_pnn
    out[..., 1] = x_pnn
    half_pi = out.new_tensor(np.pi / 2)
    out[..., 2] = torch.atan2(
        torch.sin(out[..., 2] + half_pi),
        torch.cos(out[..., 2] + half_pi),
    )
    return out


def masked_agent_minmax(
    states: torch.Tensor,
    mask: Optional[torch.Tensor],
    feature_dim: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute agent feature min/max without padded slots when masks exist."""
    states = states.reshape(-1, feature_dim)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return states.min(0).values, states.max(0).values


def tensor_feature_minmax(
    values: torch.Tensor,
    q_low: float = 0.0,
    q_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    if values.numel() == 0:
        raise ValueError("Cannot compute min/max on an empty tensor.")
    if q_low <= 0.0 and q_high >= 1.0:
        lo = values.min(0).values
        hi = values.max(0).values
    else:
        lo = torch.quantile(values, float(q_low), dim=0)
        hi = torch.quantile(values, float(q_high), dim=0)
        fallback_lo = values.min(0).values
        fallback_hi = values.max(0).values
        bad = (hi - lo).abs() < 1e-6
        lo = torch.where(bad, fallback_lo, lo)
        hi = torch.where(bad, fallback_hi, hi)
    return lo, hi


def masked_agent_stats_minmax(
    states: torch.Tensor,
    mask: Optional[torch.Tensor],
    feature_dim: int = 6,
    q_low: float = 0.0,
    q_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = states.reshape(-1, feature_dim)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return tensor_feature_minmax(states, q_low=q_low, q_high=q_high)


def resample_polyline(points: Any, num: int = LANE_POINTS) -> np.ndarray:
    """Arc-length resample a polyline to a fixed number of points."""
    pts = _ensure_2d_points(points, "points")
    if len(pts) == 1:
        return np.repeat(pts, num, axis=0)

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if float(dist[-1]) < 1e-6:
        return np.repeat(pts[:1], num, axis=0)

    target = np.linspace(0.0, float(dist[-1]), num, dtype=np.float32)
    out = np.stack(
        [
            np.interp(target, dist, pts[:, 0]),
            np.interp(target, dist, pts[:, 1]),
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def navigation_curve_points(
    points: Any,
    mode: str = "spline",
    samples_per_segment: int = 32,
) -> np.ndarray:
    """Return a dense navigation curve through sparse ego-local route points.

    ``polyline`` keeps the original piecewise-linear geometry. ``spline`` uses
    a chord-length cubic Hermite curve that passes through every input point
    and is C1-continuous at interior points. It deliberately has no dependency
    on SciPy so the same path is available in training, open-loop evaluation
    and closed-loop CARLA runs.
    """
    pts = _ensure_2d_points(points, "navigation_curve_points")
    mode = (mode or "spline").strip().lower()
    if mode in {"polyline", "linear", "line"} or len(pts) < 3:
        return pts.astype(np.float32)
    if mode not in {"spline", "catmull", "catmull_rom", "hermite"}:
        raise ValueError(f"Unsupported navigation interpolation mode: {mode!r}")

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    keep = np.concatenate([[True], seg > 1e-3])
    pts = pts[keep]
    if len(pts) < 3:
        return pts.astype(np.float32)

    # Centripetal/chord-length parameterization reduces loops/overshoot compared
    # with uniform Catmull-Rom when near/far spacing is uneven.
    seg = np.maximum(np.linalg.norm(pts[1:] - pts[:-1], axis=-1), 1e-4)
    t = np.concatenate([[0.0], np.cumsum(np.sqrt(seg))]).astype(np.float32)

    dense: List[np.ndarray] = []
    samples_per_segment = max(int(samples_per_segment), 4)
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        dt = float(t[i + 1] - t[i])
        if dt < 1e-6:
            continue

        if i == 0:
            m0 = (pts[i + 1] - pts[i]) / max(float(t[i + 1] - t[i]), 1e-6)
        else:
            m0 = (pts[i + 1] - pts[i - 1]) / max(float(t[i + 1] - t[i - 1]), 1e-6)

        if i + 1 == n - 1:
            m1 = (pts[i + 1] - pts[i]) / max(float(t[i + 1] - t[i]), 1e-6)
        else:
            m1 = (pts[i + 2] - pts[i]) / max(float(t[i + 2] - t[i]), 1e-6)

        us = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32)
        u2 = us * us
        u3 = u2 * us
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + us
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        curve = (
            h00[:, None] * p0
            + h10[:, None] * dt * m0
            + h01[:, None] * p1
            + h11[:, None] * dt * m1
        )
        dense.append(curve.astype(np.float32))

    dense.append(pts[-1:])  # include final waypoint exactly
    return np.concatenate(dense, axis=0).astype(np.float32)


def select_left_right_lane_boundaries(
    vectors: Sequence[Any],
    scores: Optional[Any] = None,
    labels: Optional[Any] = None,
    reference_plan: Optional[Any] = None,
    num_lanes: int = NUM_LANES,
    num_points: int = LANE_POINTS,
) -> np.ndarray:
    """Select lane polylines with the first two ordered as left/right boundaries.

    HiP-AD map labels are expected to be:
        0 Broken, 1 Solid, 2 SolidSolid, 3 Center.

    PNN's lane objective assumes lane_points[0] and lane_points[1] are the two
    boundaries around ego, ordered as left/right. In HiP-AD's local BEV frame,
    y is forward and x is right/lateral, so the left boundary has smaller
    x (usually negative) and the right boundary has larger x (usually
    positive). If one side is missing, this falls back to the highest-scoring
    boundary candidates instead of inventing geometry.
    """
    vector_list = [np.asarray(vec, dtype=np.float32) for vec in vectors]
    if scores is None:
        score_arr = np.ones((len(vector_list),), dtype=np.float32)
    else:
        score_arr = np.asarray(scores, dtype=np.float32)
    if labels is None:
        label_arr = np.zeros((len(vector_list),), dtype=np.int64)
    else:
        label_arr = np.asarray(labels, dtype=np.int64)

    candidates = []
    for idx, (vec, score, label) in enumerate(zip(vector_list, score_arr, label_arr)):
        if vec.ndim != 2 or vec.shape[-1] != 2:
            continue
        line = resample_polyline(vec, num_points)
        # HiP-AD occasionally emits the same geometry in reverse point order.
        # Canonicalize every candidate from near to far before pairing or
        # point-wise interpolation.
        end_window = max(1, num_points // 4)
        if np.median(line[-end_window:, 1]) < np.median(line[:end_window, 1]):
            line = line[::-1].copy()
        forward_mask = line[:, 1] > -5.0
        side_points = line[forward_mask] if forward_mask.any() else line
        lateral = float(np.median(side_points[:, 0]))
        priority = 0 if int(label) in (0, 1, 2) else 1
        candidates.append(
            {
                "priority": priority,
                "score": float(score),
                "idx": idx,
                "line": line,
                "lateral": lateral,
                "abs_lateral": abs(lateral),
            }
        )

    if len(candidates) < 2:
        return build_lane_corridor_from_plan(
            np.array([[0.0, 5.0], [0.0, 10.0], [0.0, 15.0]], dtype=np.float32),
            num_lanes=num_lanes,
            num_points=num_points,
        )

    boundary = [c for c in candidates if c["priority"] == 0]
    pool = boundary if len(boundary) >= 2 else candidates

    ref_line = None
    if reference_plan is not None:
        try:
            ref_line = resample_polyline(_ensure_2d_points(reference_plan, "reference_plan"), num_points)
            if np.median(ref_line[-max(1, num_points // 4):, 1]) < np.median(ref_line[:max(1, num_points // 4), 1]):
                ref_line = ref_line[::-1].copy()
        except (TypeError, ValueError):
            ref_line = None

    def side_rank(c):
        return (c["abs_lateral"], -c["score"], c["idx"])

    # HiP-AD coordinates are [right_x, forward_y]. Left is therefore the
    # smaller lateral-x side, right is the larger lateral-x side. Keeping this
    # order is important: lane interpolation, lane prior, lane-boundary loss
    # and Theseus' lane objective all consume lane_points[0]=left,
    # lane_points[1]=right.
    left_pool = [c for c in pool if c["lateral"] < 0.0]
    right_pool = [c for c in pool if c["lateral"] >= 0.0]
    if left_pool and right_pool:
        def pair_rank(pair):
            left_c, right_c = pair
            width = np.linalg.norm(right_c["line"] - left_c["line"], axis=-1)
            width_penalty = float(np.mean(np.maximum(2.2 - width, 0.0) + np.maximum(width - 5.5, 0.0)))
            center = 0.5 * (left_c["line"] + right_c["line"])
            if ref_line is None:
                center_error = abs(float(np.median(center[:, 0])))
            else:
                pair_dist = np.linalg.norm(center[:, None, :] - ref_line[None, :, :], axis=-1)
                center_error = 0.5 * float(
                    pair_dist.min(axis=1).mean() + pair_dist.min(axis=0).mean()
                )
            return (width_penalty * 10.0 + center_error, -left_c["score"] - right_c["score"], left_c["idx"], right_c["idx"])

        left, right = min(
            ((left_c, right_c) for left_c in left_pool for right_c in right_pool),
            key=pair_rank,
        )
        selected = [left, right]
    else:
        selected = sorted(pool, key=lambda c: (c["priority"], -c["score"], c["idx"]))[:2]
        selected = sorted(selected, key=lambda c: c["lateral"])

    # Final order must match PNN's convention: lane_points[0] left,
    # lane_points[1] right. Use the full resampled line as the final arbiter.
    if np.median(selected[0]["line"][:, 0]) > np.median(selected[1]["line"][:, 0]):
        selected = [selected[1], selected[0]]

    used = {c["idx"] for c in selected}
    rest = [
        c for c in sorted(candidates, key=lambda c: (c["priority"], -c["score"], c["idx"]))
        if c["idx"] not in used
    ]
    selected.extend(rest[: max(0, num_lanes - len(selected))])
    while len(selected) < num_lanes:
        selected.append(selected[-1])
    return np.stack([c["line"] for c in selected[:num_lanes]], axis=0).astype(np.float32)


def _canonicalize_lane_line(line: np.ndarray, num_points: int) -> np.ndarray:
    line = resample_polyline(line, num_points)
    end_window = max(1, num_points // 4)
    if np.median(line[-end_window:, 1]) < np.median(line[:end_window, 1]):
        line = line[::-1].copy()
    return line.astype(np.float32)


def _reference_frame_for_points(points: np.ndarray, reference_line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = np.zeros_like(reference_line)
    diff[:-1] = reference_line[1:] - reference_line[:-1]
    diff[-1] = diff[-2] if len(reference_line) > 1 else np.array([0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(min=1e-6)
    tangent = diff / norm
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    dist = np.linalg.norm(points[:, None, :] - reference_line[None, :, :], axis=-1)
    nearest = dist.argmin(axis=1)
    rel = points - reference_line[nearest]
    signed_left = np.sum(rel * left_normal[nearest], axis=-1)
    progress = nearest.astype(np.float32) / max(1, len(reference_line) - 1)
    return signed_left.astype(np.float32), progress.astype(np.float32)


def _lane_pair_quality(left: np.ndarray, right: np.ndarray, reference_line: np.ndarray) -> Dict[str, float]:
    left_signed, _ = _reference_frame_for_points(left, reference_line)
    right_signed, _ = _reference_frame_for_points(right, reference_line)
    width = left_signed - right_signed
    center = 0.5 * (left_signed + right_signed)
    crossing_ratio = float(np.mean(width <= 0.25))
    width_bad_ratio = float(np.mean((width < 2.2) | (width > 5.8)))
    return {
        "median_width": float(np.median(width)),
        "width_bad_ratio": width_bad_ratio,
        "crossing_ratio": crossing_ratio,
        "center_offset_median": float(np.median(np.abs(center))),
        "center_offset_mean": float(np.mean(np.abs(center))),
    }


def select_left_right_lane_boundaries_v2(
    vectors: Sequence[Any],
    scores: Optional[Any] = None,
    labels: Optional[Any] = None,
    reference_plan: Optional[Any] = None,
    num_lanes: int = NUM_LANES,
    num_points: int = LANE_POINTS,
    return_info: bool = False,
) -> Any:
    """Reference-plan guided lane boundary selector for closed-loop inference.

    The original selector uses global lateral medians, which can pair lanes from
    different branches at turns and junctions. This version projects candidates
    into a local frame along HiP-AD's own temporal plan, then chooses one left
    and one right boundary that enclose that reference corridor.
    """
    info: Dict[str, Any] = {
        "selector": "reference_plan_v2",
        "fallback": False,
        "reason": "",
    }

    if reference_plan is None:
        lanes = select_left_right_lane_boundaries(
            vectors=vectors,
            scores=scores,
            labels=labels,
            reference_plan=reference_plan,
            num_lanes=num_lanes,
            num_points=num_points,
        )
        info["fallback"] = True
        info["reason"] = "missing_reference_plan"
        return (lanes, info) if return_info else lanes

    try:
        ref_raw = _ensure_2d_points(reference_plan, "reference_plan")
        ref_line = resample_polyline(
            np.concatenate([np.zeros((1, 2), dtype=np.float32), ref_raw], axis=0),
            num_points,
        )
        if np.median(ref_line[-max(1, num_points // 4):, 1]) < np.median(ref_line[:max(1, num_points // 4), 1]):
            ref_line = ref_line[::-1].copy()
    except (TypeError, ValueError):
        lanes = build_lane_corridor_from_plan(
            np.array([[0.0, 5.0], [0.0, 10.0], [0.0, 15.0]], dtype=np.float32),
            num_lanes=num_lanes,
            num_points=num_points,
        )
        info["fallback"] = True
        info["reason"] = "invalid_reference_plan"
        return (lanes, info) if return_info else lanes

    vector_list = [np.asarray(vec, dtype=np.float32) for vec in vectors]
    score_arr = (
        np.ones((len(vector_list),), dtype=np.float32)
        if scores is None
        else np.asarray(scores, dtype=np.float32)
    )
    label_arr = (
        np.zeros((len(vector_list),), dtype=np.int64)
        if labels is None
        else np.asarray(labels, dtype=np.int64)
    )

    candidates: List[Dict[str, Any]] = []
    for idx, (vec, score, label) in enumerate(zip(vector_list, score_arr, label_arr)):
        if vec.ndim != 2 or vec.shape[-1] != 2:
            continue
        line = _canonicalize_lane_line(vec, num_points)
        signed_left, progress = _reference_frame_for_points(line, ref_line)
        forward_mask = line[:, 1] > -4.0
        if not forward_mask.any():
            forward_mask = np.ones((len(line),), dtype=bool)
        side = float(np.median(signed_left[forward_mask]))
        abs_side = abs(side)
        coverage = float(np.ptp(progress[forward_mask])) if forward_mask.any() else 0.0
        center_touch = float(np.mean(np.abs(signed_left[forward_mask]) < 0.6))
        label_int = int(label)
        if label_int in (1, 2):
            priority = 0
        elif label_int == 0:
            priority = 1
        else:
            priority = 2
        candidates.append(
            {
                "idx": idx,
                "label": label_int,
                "score": float(score),
                "priority": priority,
                "line": line,
                "side": side,
                "abs_side": abs_side,
                "coverage": coverage,
                "center_touch": center_touch,
            }
        )

    usable = [
        c for c in candidates
        if 0.75 <= c["abs_side"] <= 5.0 and c["coverage"] >= 0.25 and c["center_touch"] < 0.6
    ]
    pool = usable if len(usable) >= 2 else candidates
    left_pool = [c for c in pool if c["side"] > 0.0]
    right_pool = [c for c in pool if c["side"] < 0.0]

    if not left_pool or not right_pool:
        lanes = build_lane_corridor_from_plan(ref_raw, num_lanes=num_lanes, num_points=num_points)
        info["fallback"] = True
        info["reason"] = "missing_left_or_right"
        info["num_candidates"] = len(candidates)
        return (lanes, info) if return_info else lanes

    def pair_rank(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[float, float, int, int]:
        left_c, right_c = pair
        quality = _lane_pair_quality(left_c["line"], right_c["line"], ref_line)
        width_penalty = quality["width_bad_ratio"] * 8.0 + quality["crossing_ratio"] * 20.0
        center_penalty = quality["center_offset_mean"] * 1.2
        side_penalty = abs((left_c["side"] - right_c["side"]) - DEFAULT_LANE_WIDTH) * 0.3
        prior_penalty = float(left_c["priority"] + right_c["priority"]) * 0.4
        score_bonus = left_c["score"] + right_c["score"]
        return (
            width_penalty + center_penalty + side_penalty + prior_penalty - 0.05 * score_bonus,
            -score_bonus,
            left_c["idx"],
            right_c["idx"],
        )

    left, right = min(((l, r) for l in left_pool for r in right_pool), key=pair_rank)
    lanes0 = np.stack([left["line"], right["line"]], axis=0)
    quality = _lane_pair_quality(lanes0[0], lanes0[1], ref_line)
    if (
        quality["crossing_ratio"] > 0.05
        or quality["width_bad_ratio"] > 0.35
        or quality["center_offset_median"] > 2.2
    ):
        lanes = build_lane_corridor_from_plan(ref_raw, num_lanes=num_lanes, num_points=num_points)
        info["fallback"] = True
        info["reason"] = "bad_pair_quality"
        info.update(quality)
        info["selected_indices"] = [int(left["idx"]), int(right["idx"])]
        info["selected_labels"] = [int(left["label"]), int(right["label"])]
        return (lanes, info) if return_info else lanes

    used = {left["idx"], right["idx"]}
    rest = [
        c for c in sorted(candidates, key=lambda c: (c["priority"], -c["score"], c["idx"]))
        if c["idx"] not in used
    ]
    selected = [left, right] + rest[: max(0, num_lanes - 2)]
    while len(selected) < num_lanes:
        selected.append(selected[-1])
    lanes = np.stack([c["line"] for c in selected[:num_lanes]], axis=0).astype(np.float32)
    info.update(quality)
    info["num_candidates"] = len(candidates)
    info["num_usable"] = len(usable)
    info["selected_indices"] = [int(left["idx"]), int(right["idx"])]
    info["selected_labels"] = [int(left["label"]), int(right["label"])]
    info["selected_sides"] = [float(left["side"]), float(right["side"])]
    return (lanes, info) if return_info else lanes


def interpolate_plan(points: Any, num: int = TRAJ_LEN) -> np.ndarray:
    """Convert HiP-AD [6,2] planning points to 30 PNN steps at 0.1 s."""
    plan = _ensure_2d_points(points, "plan")
    src_t = np.linspace(0.5, 3.0, len(plan), dtype=np.float32)
    dst_t = np.arange(1, num + 1, dtype=np.float32) * DT
    out = np.stack(
        [
            np.interp(dst_t, src_t, plan[:, 0], left=0.0, right=plan[-1, 0]),
            np.interp(dst_t, src_t, plan[:, 1], left=0.0, right=plan[-1, 1]),
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def plan_to_route_targets(plan: Any) -> np.ndarray:
    """Pick 1s, 2s, 3s route targets from a HiP-AD planning trajectory.

    HiP-AD's closed-loop plan has six points at 0.5 s intervals, so the
    1/2/3 s targets are directly the 2nd, 4th, and 6th points. For nonstandard
    plan lengths, fall back to time interpolation.
    """
    plan_arr = _ensure_2d_points(plan, "plan")
    if plan_arr.shape[0] >= 6:
        targets = plan_arr[[1, 3, 5]].reshape(-1)
    else:
        dense = interpolate_plan(plan_arr, TRAJ_LEN)
        targets = dense[[9, 19, 29]].reshape(-1)
    return targets.astype(np.float32)


def navigation_points_to_route_targets(
    navigation_points: Any,
    ego_speed: float,
    target_times: Sequence[float] = DEFAULT_ROUTE_TARGET_TIMES,
    min_speed: float = DEFAULT_NAV_MIN_SPEED,
    max_speed: float = DEFAULT_NAV_MAX_SPEED,
    distance_scale: Any = 1.0,
    interpolation: str = "spline",
) -> np.ndarray:
    """Infer fixed-time route targets by sampling an ego-local navigation curve.

    Navigation points carry route geometry but no timestamps. The route is
    first densified with either a piecewise-linear path (``polyline``) or a
    chord-length cubic Hermite spline (``spline``). We then use a constant-speed
    prior and sample the curve at

        distance(t) = clip(ego_speed, min_speed, max_speed) * t * distance_scale.

    ``distance_scale`` may be either a scalar or one value per target time. For
    example, "0.96,1.03,1.12" calibrates the 1s/2s/3s navigation targets
    independently against the empirical ground-truth trajectory distribution.

    The ego origin is prepended automatically. Duplicate navigation points are
    removed, and targets beyond the available route are clamped to its endpoint.
    The returned layout matches PNN's ego state: [x1,y1,x2,y2,x3,y3].
    """
    points = _ensure_2d_points(navigation_points, "navigation_points")
    origin = np.zeros((1, 2), dtype=np.float32)
    points = np.concatenate([origin, points.astype(np.float32)], axis=0)

    compact = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - compact[-1])) > 1e-3:
            compact.append(point)
    curve = navigation_curve_points(
        np.asarray(compact, dtype=np.float32),
        mode=interpolation,
    )

    times = np.asarray(target_times, dtype=np.float32)
    if times.ndim != 1 or len(times) == 0 or np.any(times <= 0):
        raise ValueError(f"target_times must contain positive values, got {target_times}")

    speed = float(np.clip(float(ego_speed), float(min_speed), float(max_speed)))
    scales = _per_target_scales(distance_scale, len(times))
    query_distances = times * speed * scales

    if len(curve) == 1:
        # No usable navigation geometry. Keep the conventional forward heading.
        targets = np.stack(
            [np.zeros_like(query_distances), query_distances],
            axis=-1,
        )
        return targets.reshape(-1).astype(np.float32)

    segment_lengths = np.linalg.norm(curve[1:] - curve[:-1], axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float32)
    total_length = float(cumulative[-1])
    if total_length < 1e-4:
        targets = np.stack(
            [np.zeros_like(query_distances), query_distances],
            axis=-1,
        )
        return targets.reshape(-1).astype(np.float32)

    query_distances = np.clip(query_distances, 0.0, total_length)
    targets = np.stack(
        [
            np.interp(query_distances, cumulative, curve[:, 0]),
            np.interp(query_distances, cumulative, curve[:, 1]),
        ],
        axis=-1,
    )
    return targets.reshape(-1).astype(np.float32)


def headings_from_points(points: np.ndarray, default: float = np.pi / 2) -> np.ndarray:
    pts = _ensure_2d_points(points, "points")
    if len(pts) < 2:
        return np.full((len(pts),), default, dtype=np.float32)
    diff = np.zeros_like(pts)
    diff[:-1] = pts[1:] - pts[:-1]
    diff[-1] = diff[-2]
    valid = np.linalg.norm(diff, axis=-1) > 1e-4
    theta = np.full((len(pts),), default, dtype=np.float32)
    theta[valid] = np.arctan2(diff[valid, 1], diff[valid, 0])
    return _angle_wrap(theta)


def build_lane_corridor_from_plan(
    plan: Any,
    lane_width: float = DEFAULT_LANE_WIDTH,
    num_lanes: int = NUM_LANES,
    num_points: int = LANE_POINTS,
) -> np.ndarray:
    """
    Build a synthetic lane bundle around the predicted route.

    This is a fallback for online closed-loop use when map/lane outputs are not
    yet extracted. The first two lines are left/right boundaries; additional
    lines interpolate between and slightly outside them for WeightNet tokens.
    """
    center = resample_polyline(np.concatenate([np.zeros((1, 2), np.float32), _ensure_2d_points(plan, "plan")]), num_points)
    if len(center) >= 2:
        diff = np.zeros_like(center)
        diff[:-1] = center[1:] - center[:-1]
        diff[-1] = diff[-2]
    else:
        diff = np.array([[0.0, 1.0]], dtype=np.float32)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(min=1e-6)
    tangent = diff / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    half = float(lane_width) * 0.5
    left = center + normal * half
    right = center - normal * half
    lanes = [left, right]
    if num_lanes > 2:
        offsets = np.linspace(-half * 1.25, half * 1.25, num_lanes - 2, dtype=np.float32)
        lanes.extend([center + normal * off for off in offsets])
    return np.stack(lanes[:num_lanes], axis=0).astype(np.float32)


def normalize_lane_points(lane_points: Any, fallback_plan: Any = None) -> np.ndarray:
    """Return [10,20,2] lane tensor. First two entries must be lane boundaries."""
    if lane_points is None:
        if fallback_plan is None:
            fallback_plan = np.array([[0.0, 5.0], [0.0, 10.0], [0.0, 15.0]], dtype=np.float32)
        return build_lane_corridor_from_plan(fallback_plan)

    lanes = _as_np(lane_points)
    if lanes.ndim == 4 and lanes.shape[0] == 1:
        lanes = lanes[0]
    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        raise ValueError(f"lane_points must have shape [L,P,2], got {lanes.shape}")

    fixed: List[np.ndarray] = [resample_polyline(line, LANE_POINTS) for line in lanes[:NUM_LANES]]
    if not fixed:
        return build_lane_corridor_from_plan(fallback_plan)
    while len(fixed) < NUM_LANES:
        fixed.append(fixed[-1].copy())
    return np.stack(fixed[:NUM_LANES], axis=0).astype(np.float32)


def _agent_state_from_mapping(agent: Mapping[str, Any], horizon_s: float = 3.0) -> np.ndarray:
    x = float(agent.get("x", agent.get("pos", [0.0, 0.0])[0]))
    y = float(agent.get("y", agent.get("pos", [0.0, 0.0])[1]))
    yaw = float(agent.get("yaw", agent.get("theta", np.pi / 2)))
    speed = float(agent.get("speed", agent.get("v", 0.0)))
    if "goal" in agent:
        gx, gy = agent["goal"]
    elif "future" in agent:
        fut = _ensure_2d_points(agent["future"], "agent.future")
        gx, gy = fut[-1]
    else:
        gx = x + speed * np.cos(yaw) * horizon_s
        gy = y + speed * np.sin(yaw) * horizon_s
    return np.array([x, y, yaw, speed, gx, gy], dtype=np.float32)


def pack_agents(
    agents: Optional[Iterable[Mapping[str, Any]]],
    max_agents: int,
) -> Tuple[np.ndarray, np.ndarray]:
    states = np.zeros((max_agents, 6), dtype=np.float32)
    mask = np.zeros((max_agents,), dtype=bool)
    if agents is None:
        return states, mask
    packed = [_agent_state_from_mapping(a) for a in agents]
    packed.sort(key=lambda s: float(np.linalg.norm(s[:2])))
    for i, state in enumerate(packed[:max_agents]):
        states[i] = state
        mask[i] = True
    return states, mask


@dataclass
class PNNAdapterConfig:
    pnn_root: str = DEFAULT_PNN_ROOT
    pnn_main: str = DEFAULT_PNN_MAIN
    stats_path: str = DEFAULT_STATS_PATH
    control_ckpt_path: str = DEFAULT_CONTROL_CKPT
    weight_ckpt_path: Optional[str] = DEFAULT_WEIGHT_CKPT
    device: str = "cuda:0"
    lane_width: float = DEFAULT_LANE_WIDTH
    use_weight_net: bool = True
    # When enabled, WeightNet weights are applied by the same differentiable
    # DIPP optimizer used during training. If optimization fails for one
    # sample, inference falls back to the frozen ControlNet rollout.
    use_theseus_refine: bool = False
    planner_optimizer: str = "levenberg_marquardt"
    planner_max_iterations: int = 10
    planner_step_size: float = 0.10
    planner_ped_safety_distance: float = 2.5
    planner_veh_safety_distance: float = 4.0
    planner_ped_lateral_safety_distance: float = 1.2
    planner_veh_lateral_safety_distance: float = 1.8
    planner_control_anchor_weight: float = 500.0
    planner_control_anchor_risk_floor: float = 0.05
    weight_temperature: float = 1.5
    weight_initial_refine_gate: float = 0.01
    default_cost_weights: Tuple[float, ...] = (1.0, 2.0, 0.6, 2.0, 3.0, 2.0, 1.2, 10.0)
    weight_delta_max: Any = 0.7
    safe_dist: float = 10.0
    collision_dist: float = 3.0
    collision_risk_sharpness: float = 1.5
    prior_dense_gain: float = 1.4
    prior_turn_gain: float = 1.2
    prior_high_speed_gain: float = 0.9
    # Coordinate convention at the PNN network/dynamics boundary:
    #   hipad_xy: current bridge behavior, x=right/lateral and y=forward, ego yaw=pi/2.
    #   pnn_xy:   original PNN-style local frame, x=forward and y=left, ego yaw=0.
    coord_convention: str = "hipad_xy"
    # Optional post-rollout reference-point correction in meters. Positive
    # values shift each predicted ego state along its own heading direction
    # before exposing final_planning/dense_trajectory to HiP-AD metrics or PID.
    # Keep 0.0 by default to preserve the original behavior.
    output_forward_offset: float = 0.0
    # PNN internal ego state reference point offset relative to the HiP-AD/GT
    # ego reference point. Positive means the PNN state point is physically
    # ahead of the HiP-AD/GT reference point. Inputs to PNN are shifted forward
    # by this value; PNN outputs are shifted backward by the same value before
    # being exposed to HiP-AD metrics/PID.
    reference_forward_offset: float = 0.0
    stats_quantile_low: float = 0.0
    stats_quantile_high: float = 1.0
    clamp_normalized_inputs: bool = False
    control_min_accel: float = -10.0
    control_max_accel: float = 10.0
    control_max_steer: float = 1.066
    min_weight: Tuple[float, ...] = (0.05, 0.05, 0.02, 0.02, 0.20, 0.20, 0.20, 0.50)
    max_weight: Tuple[float, ...] = (8.0, 8.0, 5.0, 6.0, 14.0, 14.0, 16.0, 16.0)


class PNNOptimizerAdapter:
    """Load PNN modules and refine HiP-AD planning trajectories."""

    def __init__(self, config: PNNAdapterConfig = PNNAdapterConfig()):
        self.cfg = config
        if self.cfg.coord_convention not in ("hipad_xy", "pnn_xy"):
            raise ValueError(
                "coord_convention must be 'hipad_xy' or 'pnn_xy', "
                f"got {self.cfg.coord_convention!r}"
            )
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._install_paths()
        self._load_pnn_symbols()
        self.stats = self._load_stats(config.stats_path)
        self.nnc_dyn = self._load_control_network(config.control_ckpt_path)
        self.weight_model = self._load_weight_network(config.weight_ckpt_path) if config.use_weight_net else None
        self.upper_dipp = None
        if config.use_theseus_refine:
            self._load_dipp_symbols()
            self.upper_dipp = self.MotionPlannerCompatible(
                trajectory_len=TRAJ_LEN,
                feature_len=NUM_COSTS,
                num_objects=NUM_PEDS + NUM_VEHS,
                device=self.device,
                optimizer_type=config.planner_optimizer,
                max_iterations=config.planner_max_iterations,
                step_size=config.planner_step_size,
            )

    def _install_paths(self) -> None:
        for path in [self.cfg.pnn_root, self.cfg.pnn_main]:
            if path and path not in sys.path:
                sys.path.insert(0, path)

    def _load_pnn_symbols(self) -> None:
        from PCC_helpers_v8 import EluTimeControlEnhanced, normalize, inverse_normalize  # type: ignore
        from weight_model_v10 import WeightNet  # type: ignore

        self.EluTimeControlEnhanced = EluTimeControlEnhanced
        self.normalize = normalize
        self.inverse_normalize = inverse_normalize
        self.WeightNet = WeightNet

    def _load_dipp_symbols(self) -> None:
        from train_v10 import (  # type: ignore
            MotionPlannerCompatible,
            build_scene_adaptive_cost_prior,
            build_scene_feature_vector_from_batch,
            compute_rollout_collision_risk,
            run_upper_dipp,
        )

        self.MotionPlannerCompatible = MotionPlannerCompatible
        self.build_scene_adaptive_cost_prior = build_scene_adaptive_cost_prior
        self.build_scene_feature_vector_from_batch = build_scene_feature_vector_from_batch
        self.compute_rollout_collision_risk = compute_rollout_collision_risk
        self.run_upper_dipp = run_upper_dipp

    def _load_stats(self, path: str) -> Dict[str, torch.Tensor]:
        data = torch.load(path, map_location="cpu")
        q_low = float(self.cfg.stats_quantile_low)
        q_high = float(self.cfg.stats_quantile_high)
        precomputed_formats = {
            "pnn_precomputed_normalization_v1",
            # The time-aligned data pipeline reuses the static-data stats
            # writer.  Its common ControlNet fields have the same meaning;
            # the additional static ranges are intentionally ignored here.
            "pnn_static_precomputed_normalization_v1",
        }
        if data.get("format") in precomputed_formats:
            saved_q_low = float(data["q_low"])
            saved_q_high = float(data["q_high"])
            if abs(saved_q_low - q_low) > 1e-9 or abs(saved_q_high - q_high) > 1e-9:
                raise ValueError(
                    "Precomputed PNN normalization quantiles do not match: "
                    f"file=({saved_q_low},{saved_q_high}) "
                    f"runtime=({q_low},{q_high})"
                )
            keys = (
                "min_ego",
                "max_ego",
                "min_ped",
                "max_ped",
                "min_veh",
                "max_veh",
                "min_lane",
                "max_lane",
            )
            missing = [key for key in keys if key not in data]
            if missing:
                raise KeyError(
                    f"Precomputed PNN normalization stats missing keys: {missing}"
                )
            print(
                f"[PNNAdapter] precomputed normalization stats: q_low={q_low} "
                f"q_high={q_high} clamp={self.cfg.clamp_normalized_inputs}"
            )
            return {key: data[key].to(self.device) for key in keys}

        min_ego, max_ego = tensor_feature_minmax(
            data["ego_state"],
            q_low=q_low,
            q_high=q_high,
        )
        min_ped, max_ped = masked_agent_stats_minmax(
            data["ped_states"],
            data.get("ped_mask"),
            q_low=q_low,
            q_high=q_high,
        )
        min_veh, max_veh = masked_agent_stats_minmax(
            data["veh_states"],
            data.get("veh_mask"),
            q_low=q_low,
            q_high=q_high,
        )
        min_lane, max_lane = tensor_feature_minmax(
            data["lane_points"][:, 0:2].reshape(-1, 2),
            q_low=q_low,
            q_high=q_high,
        )
        print(
            f"[PNNAdapter] normalization stats: q_low={q_low} "
            f"q_high={q_high} clamp={self.cfg.clamp_normalized_inputs}"
        )
        return {
            "min_ego": min_ego.to(self.device),
            "max_ego": max_ego.to(self.device),
            "min_ped": min_ped.to(self.device),
            "max_ped": max_ped.to(self.device),
            "min_veh": min_veh.to(self.device),
            "max_veh": max_veh.to(self.device),
            "min_lane": min_lane.to(self.device),
            "max_lane": max_lane.to(self.device),
        }

    def _load_control_network(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "neural_net" not in ckpt:
            raise KeyError(f"{ckpt_path} does not contain 'neural_net'")
        core = self.EluTimeControlEnhanced(embed_dim=128, num_heads=4, future_steps=TRAJ_LEN).to(self.device)
        core.load_state_dict(ckpt["neural_net"], strict=True)
        core.eval()
        return core

    def _load_weight_network(self, ckpt_path: Optional[str]):
        model = self.WeightNet(
            embed_dim=128,
            num_heads=4,
            num_tasks=NUM_COSTS,
            temperature=self.cfg.weight_temperature,
            use_prior_context=True,
            prior_context_mode="log",
            initial_refine_gate=self.cfg.weight_initial_refine_gate,
        ).to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt.get("weight_model", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[PNNAdapter] weight_model missing keys: {len(missing)}")
            if unexpected:
                print(f"[PNNAdapter] weight_model unexpected keys: {len(unexpected)}")
        else:
            print("[PNNAdapter] no weight checkpoint found; using randomly initialized WeightNet")
        model.eval()
        return model

    @staticmethod
    def _split_lane_for_control_and_weight(lane_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lane_points.shape[1] == 2:
            left_line = lane_points[:, 0:1]
            right_line = lane_points[:, 1:2]
            alphas = torch.linspace(0.0, 1.0, NUM_LANES, device=lane_points.device, dtype=lane_points.dtype)
            lane_weight = left_line * (1.0 - alphas.view(1, NUM_LANES, 1, 1)) + right_line * alphas.view(1, NUM_LANES, 1, 1)
            return lane_points, lane_weight
        if lane_points.shape[1] >= NUM_LANES:
            return lane_points[:, :2], lane_points[:, :NUM_LANES]
        raise ValueError(f"lane_points second dim should be 2 or >=10, got {lane_points.shape[1]}")

    @staticmethod
    def _rollout_bicycle(control: torch.Tensor, ego_state: torch.Tensor, wheelbase: float = 2.85) -> torch.Tensor:
        state = ego_state[:, :4]
        states = []
        wb = torch.as_tensor(wheelbase, device=control.device, dtype=control.dtype)
        for t in range(control.shape[1]):
            theta = torch.atan2(torch.sin(state[:, 2]), torch.cos(state[:, 2]))
            v = state[:, 3].clamp_min(1e-8)
            acc = control[:, t, 0]
            steer = control[:, t, 1]
            dx = torch.stack(
                [
                    v * torch.cos(theta),
                    v * torch.sin(theta),
                    (v / wb) * torch.tan(steer),
                    acc,
                ],
                dim=-1,
            )
            state = state + DT * dx
            state = torch.stack(
                [
                    state[:, 0],
                    state[:, 1],
                    torch.atan2(torch.sin(state[:, 2]), torch.cos(state[:, 2])),
                    state[:, 3].clamp_min(0.0),
                ],
                dim=-1,
            )
            states.append(state)
        return torch.stack(states, dim=1)

    @staticmethod
    def _apply_output_forward_offset(traj: torch.Tensor, offset: float) -> torch.Tensor:
        """Shift rollout xy by ``offset`` along each state's heading.

        PNN's bicycle rollout state may represent a different vehicle reference
        point than HiP-AD/Bench2Drive planning metrics. This post-processing
        lets us test and deploy a small reference-point correction without
        changing ControlNet outputs or the learned dynamics.
        """
        offset = float(offset)
        if abs(offset) < 1e-8:
            return traj
        shifted = traj.clone()
        theta = shifted[..., 2]
        shifted[..., 0] = shifted[..., 0] + offset * torch.cos(theta)
        shifted[..., 1] = shifted[..., 1] + offset * torch.sin(theta)
        return shifted

    @staticmethod
    def _apply_reference_forward_offset_np(
        points: Any,
        offset: float,
        coord_convention: str,
    ) -> np.ndarray:
        offset = float(offset)
        pts = np.asarray(points, dtype=np.float32).copy()
        if abs(offset) < 1e-8:
            return pts
        if coord_convention == "pnn_xy":
            pts[..., 0] += offset
        elif coord_convention == "hipad_xy":
            pts[..., 1] += offset
        else:
            raise ValueError(f"Unsupported coord_convention={coord_convention!r}")
        return pts.astype(np.float32)

    def _build_default_cost_weights(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weights = torch.as_tensor(self.cfg.default_cost_weights, device=device, dtype=dtype)
        if weights.numel() != NUM_COSTS:
            raise ValueError(
                f"default_cost_weights must have {NUM_COSTS} values, got {weights.numel()}"
            )
        return weights.view(1, -1).repeat(batch_size, 1)

    @staticmethod
    def _normalize_prob(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _sanitize_weights(
        weights: torch.Tensor,
        min_weight: Sequence[float],
        max_weight: Sequence[float],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        min_w = torch.as_tensor(min_weight, device=weights.device, dtype=weights.dtype).view(1, -1)
        max_w = torch.as_tensor(max_weight, device=weights.device, dtype=weights.dtype).view(1, -1)
        out = torch.nan_to_num(weights, nan=1.0, posinf=20.0, neginf=1e-3)
        out = torch.where(torch.isfinite(out), out, fallback)
        return out.clamp(min=min_w, max=max_w)

    @staticmethod
    def _expand_cost_vector(value: Any, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, (int, float)):
            return torch.full((batch_size, NUM_COSTS), float(value), device=device, dtype=dtype)
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(batch_size, NUM_COSTS)
        if tensor.ndim == 1:
            if tensor.numel() != NUM_COSTS:
                raise ValueError(f"cost vector must have {NUM_COSTS} values, got {tensor.numel()}")
            return tensor.view(1, NUM_COSTS).expand(batch_size, NUM_COSTS)
        if tensor.ndim == 2:
            if tensor.shape == (1, NUM_COSTS):
                return tensor.expand(batch_size, NUM_COSTS)
            if tensor.shape == (batch_size, NUM_COSTS):
                return tensor
        raise ValueError(f"unsupported cost vector shape: {tuple(tensor.shape)}")

    def _scene_prior(
        self,
        ego_state,
        ped_states,
        veh_states,
        lane_points,
        ped_mask,
        veh_mask,
        ego_rollout=None,
    ) -> torch.Tensor:
        base = self._build_default_cost_weights(ego_state.shape[0], ego_state.device, ego_state.dtype)
        if ego_rollout is not None and hasattr(self, "build_scene_adaptive_cost_prior"):
            collision_risk = self.compute_rollout_collision_risk(
                ego_traj=ego_rollout,
                ped_states=ped_states,
                veh_states=veh_states,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                safe_dist=self.cfg.safe_dist,
                collision_dist=self.cfg.collision_dist,
                sharpness=self.cfg.collision_risk_sharpness,
            )
            scene_vec = self.build_scene_feature_vector_from_batch(
                ego_state=ego_state,
                ped_states=ped_states,
                veh_states=veh_states,
                lane_points=lane_points,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                safe_dist=self.cfg.safe_dist,
            )
            prior_out = self.build_scene_adaptive_cost_prior(
                scene_vec=scene_vec,
                base_weights=base,
                collision_risk=collision_risk,
                dense_gain=self.cfg.prior_dense_gain,
                turn_gain=self.cfg.prior_turn_gain,
                high_speed_gain=self.cfg.prior_high_speed_gain,
            )
            return self._sanitize_weights(
                prior_out["scene_prior_weights"],
                self.cfg.min_weight,
                self.cfg.max_weight,
                base,
            )

        ego_xy = ego_state[:, :2]
        far = self.cfg.safe_dist * 5.0

        def min_dist(states, mask):
            d = torch.norm(states[:, :, :2] - ego_xy.unsqueeze(1), dim=-1)
            d = torch.where(mask.bool(), d, torch.full_like(d, far))
            return d.min(dim=1, keepdim=True).values.clamp(0.0, far)

        min_agent = torch.minimum(min_dist(ped_states, ped_mask), min_dist(veh_states, veh_mask))
        risk = torch.sigmoid((self.cfg.safe_dist - min_agent) / 1.5)
        speed = ego_state[:, 3:4].clamp(0.0, 30.0)
        left, right = lane_points[:, 0], lane_points[:, 1]
        lane_width = torch.norm(left - right, dim=-1).mean(dim=1, keepdim=True)
        narrow = ((3.8 - lane_width) / 1.5).clamp(0.0, 1.0)

        log_scale = torch.zeros_like(base)
        log_scale[:, 1:2] += 0.35 * risk
        log_scale[:, 3:4] += 0.35 * risk
        log_scale[:, 4:5] += 0.45 * risk + 0.35 * narrow
        log_scale[:, 5:6] += 0.25 * narrow
        log_scale[:, 6:7] += -0.30 * risk
        log_scale[:, 7:8] += 0.95 * risk
        log_scale[:, 2:3] += 0.20 * torch.sigmoid((speed - 12.0) / 3.0)
        prior = base * torch.exp(log_scale)
        return self._sanitize_weights(prior, self.cfg.min_weight, self.cfg.max_weight, base)

    def build_sample(
        self,
        hipad_plan: Any,
        ego_speed: float,
        spatial_plan: Any = None,
        ped_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        veh_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        lane_points: Any = None,
        ego_xy: Sequence[float] = (0.0, 0.0),
        ego_yaw: float = np.pi / 2,
        navigation_points: Any = None,
        route_source: str = "hipad_plan",
        navigation_min_speed: float = DEFAULT_NAV_MIN_SPEED,
        navigation_max_speed: float = DEFAULT_NAV_MAX_SPEED,
        navigation_distance_scale: Any = 1.0,
        navigation_interpolation: str = "spline",
    ) -> Dict[str, np.ndarray]:
        if route_source == "navigation":
            if navigation_points is None:
                raise ValueError("route_source=navigation requires navigation_points")
            nav_points = _ensure_2d_points(navigation_points, "navigation_points")
            route = navigation_points_to_route_targets(
                nav_points,
                ego_speed=ego_speed,
                min_speed=navigation_min_speed,
                max_speed=navigation_max_speed,
                distance_scale=navigation_distance_scale,
                interpolation=navigation_interpolation,
            )
            fallback_route = nav_points
            plan = (
                _ensure_2d_points(hipad_plan, "hipad_plan")
                if hipad_plan is not None
                else nav_points
            )
        elif route_source == "hipad_plan":
            if hipad_plan is None:
                raise ValueError("route_source=hipad_plan requires hipad_plan")
            plan = _ensure_2d_points(hipad_plan, "hipad_plan")
            route = plan_to_route_targets(plan)
            fallback_route = plan
        else:
            raise ValueError(f"Unsupported route_source={route_source!r}")

        lane_fallback = spatial_plan if spatial_plan is not None else fallback_route
        if self.cfg.coord_convention == "pnn_xy":
            route = hipad_points_to_pnn(route.reshape(3, 2)).reshape(-1)
            plan = hipad_points_to_pnn(plan)
            ego_xy = hipad_points_to_pnn(np.asarray(ego_xy, dtype=np.float32))
            ego_yaw = float(hipad_yaw_to_pnn(np.asarray(ego_yaw, dtype=np.float32)))

        ego_state = np.array(
            [float(ego_xy[0]), float(ego_xy[1]), float(ego_yaw), float(ego_speed), *route],
            dtype=np.float32,
        )
        reference_forward_offset = float(self.cfg.reference_forward_offset)
        if abs(reference_forward_offset) > 1e-8:
            ego_xy = self._apply_reference_forward_offset_np(
                ego_state[0:2],
                reference_forward_offset,
                self.cfg.coord_convention,
            )
            route = self._apply_reference_forward_offset_np(
                route.reshape(3, 2),
                reference_forward_offset,
                self.cfg.coord_convention,
            ).reshape(-1)
            ego_state = np.array(
                [float(ego_xy[0]), float(ego_xy[1]), float(ego_yaw), float(ego_speed), *route],
                dtype=np.float32,
            )
        ped_states, ped_mask = pack_agents(ped_agents, NUM_PEDS)
        veh_states, veh_mask = pack_agents(veh_agents, NUM_VEHS)
        if self.cfg.coord_convention == "pnn_xy":
            ped_states = agent_states_hipad_to_pnn(ped_states)
            veh_states = agent_states_hipad_to_pnn(veh_states)
        lane = normalize_lane_points(
            lane_points,
            lane_fallback,
        )
        if self.cfg.coord_convention == "pnn_xy":
            lane = hipad_points_to_pnn(lane)
        return {
            "ego_state": ego_state,
            "ped_states": ped_states,
            "veh_states": veh_states,
            "lane_points": lane,
            "ped_mask": ped_mask,
            "veh_mask": veh_mask,
            "hipad_plan": plan.astype(np.float32),
            "route_source": route_source,
            "route_targets": route.reshape(3, 2).astype(np.float32),
        }

    def _to_batch(self, sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in ["ego_state", "ped_states", "veh_states", "lane_points"]:
            arr = _as_np(sample[key])
            if arr.ndim in (1, 2, 3):
                arr = arr[None]
            batch[key] = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        for key in ["ped_mask", "veh_mask"]:
            arr = np.asarray(sample[key], dtype=bool)
            if arr.ndim == 1:
                arr = arr[None]
            batch[key] = torch.as_tensor(arr, dtype=torch.bool, device=self.device)
        return batch

    @torch.no_grad()
    def refine_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        b = self._to_batch(sample)
        ego_state = b["ego_state"]
        ped_states = b["ped_states"]
        veh_states = b["veh_states"]
        lane_points = b["lane_points"]
        ped_mask = b["ped_mask"]
        veh_mask = b["veh_mask"]
        B = ego_state.shape[0]

        lane_control, lane_weight = self._split_lane_for_control_and_weight(lane_points)
        ego_n = self.normalize(ego_state, self.stats["min_ego"], self.stats["max_ego"])
        ped_n = self.normalize(ped_states, self.stats["min_ped"], self.stats["max_ped"])
        veh_n = self.normalize(veh_states, self.stats["min_veh"], self.stats["max_veh"])
        lane_n = self.normalize(
            lane_control.reshape(B, -1, 2),
            self.stats["min_lane"],
            self.stats["max_lane"],
        ).reshape(B, lane_control.shape[1], lane_control.shape[2], 2)
        lane_weight_n = self.normalize(
            lane_weight.reshape(B, -1, 2),
            self.stats["min_lane"],
            self.stats["max_lane"],
        ).reshape(B, lane_weight.shape[1], lane_weight.shape[2], 2)
        if self.cfg.clamp_normalized_inputs:
            ego_n = ego_n.clamp(-1.0, 1.0)
            ped_n = ped_n.clamp(-1.0, 1.0)
            veh_n = veh_n.clamp(-1.0, 1.0)
            lane_n = lane_n.clamp(-1.0, 1.0)
            lane_weight_n = lane_weight_n.clamp(-1.0, 1.0)

        u_ego_n, _u_peds_n, _u_vehs_n = self.nnc_dyn(
            ego_n,
            ped_n,
            veh_n,
            lane_n,
            ~ped_mask,
            ~veh_mask,
        )
        a_ego = torch.tensor(
            [self.cfg.control_min_accel, -self.cfg.control_max_steer],
            device=self.device,
        ).view(1, 1, 2)
        b_ego = torch.tensor(
            [self.cfg.control_max_accel, self.cfg.control_max_steer],
            device=self.device,
        ).view(1, 1, 2)
        init_control = self.inverse_normalize(u_ego_n, a_ego, b_ego)
        init_control = torch.stack(
            [
                init_control[..., 0].clamp(
                    self.cfg.control_min_accel, self.cfg.control_max_accel
                ),
                init_control[..., 1].clamp(
                    -self.cfg.control_max_steer, self.cfg.control_max_steer
                ),
            ],
            dim=-1,
        )

        ego_rollout = self._rollout_bicycle(init_control, ego_state)
        scene_prior_weights = self._scene_prior(
            ego_state,
            ped_states,
            veh_states,
            lane_weight,
            ped_mask,
            veh_mask,
            ego_rollout=ego_rollout,
        )

        if self.weight_model is not None:
            prior_log_weights = scene_prior_weights.clamp_min(1e-8).log()
            weight_out = self.weight_model(
                ego_state=ego_n,
                ped_states=ped_n,
                veh_states=veh_n,
                lane_points=lane_weight_n,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                prior_log_weights=prior_log_weights,
                return_logits=True,
            )
            logits = weight_out[1] if isinstance(weight_out, (tuple, list)) else weight_out
            refine_gate = (
                weight_out[2]
                if isinstance(weight_out, (tuple, list)) and len(weight_out) >= 3
                else torch.ones((B, 1), device=ego_state.device, dtype=ego_state.dtype)
            )
            refine_gate = torch.nan_to_num(refine_gate, nan=0.0).clamp(0.0, 1.0)
            delta_max = self._expand_cost_vector(
                self.cfg.weight_delta_max,
                ego_state.shape[0],
                logits.device,
                logits.dtype,
            )
            delta = delta_max * torch.tanh(logits)
            cost_weights = self._sanitize_weights(
                scene_prior_weights * torch.exp(delta),
                self.cfg.min_weight,
                self.cfg.max_weight,
                scene_prior_weights,
            )
        else:
            cost_weights = scene_prior_weights
            refine_gate = torch.ones((B, 1), device=ego_state.device, dtype=ego_state.dtype)

        refined_control = init_control
        refined_traj = ego_rollout
        dipp_failed = False
        if self.upper_dipp is not None:
            try:
                # Theseus builds Jacobians internally, so it needs autograd even
                # though this adapter never backpropagates into either network.
                with torch.enable_grad():
                    dipp_control, _dipp_traj, _ = self.run_upper_dipp(
                        self.upper_dipp,
                        ego_state.detach(),
                        lane_control.detach(),
                        init_control.detach(),
                        cost_weights.detach(),
                        ped_states.detach(),
                        veh_states.detach(),
                        ped_mask.detach(),
                        veh_mask.detach(),
                        planner_weight_min=self.cfg.min_weight,
                        planner_weight_max=self.cfg.max_weight,
                        planner_weight_renormalize_to_default_sum=False,
                        ped_safety_distance=self.cfg.planner_ped_safety_distance,
                        veh_safety_distance=self.cfg.planner_veh_safety_distance,
                        ped_lateral_safety_distance=self.cfg.planner_ped_lateral_safety_distance,
                        veh_lateral_safety_distance=self.cfg.planner_veh_lateral_safety_distance,
                        control_anchor_weight=self.cfg.planner_control_anchor_weight,
                        control_anchor_risk_floor=self.cfg.planner_control_anchor_risk_floor,
                    )
                dipp_control = dipp_control.detach()
                refined_control = init_control + refine_gate.view(B, 1, 1) * (
                    dipp_control - init_control
                )
                refined_control = torch.stack(
                    [
                        refined_control[..., 0].clamp(-10.0, 10.0),
                        refined_control[..., 1].clamp(-1.066, 1.066),
                    ],
                    dim=-1,
                )
                refined_traj = self._rollout_bicycle(refined_control, ego_state)
            except Exception as exc:
                dipp_failed = True
                print(f"[PNNAdapter] DIPP failed; using ControlNet rollout: {exc}")

        effective_output_forward_offset = (
            float(self.cfg.output_forward_offset)
            - float(self.cfg.reference_forward_offset)
        )
        output_traj = self._apply_output_forward_offset(
            refined_traj,
            effective_output_forward_offset,
        )
        output_initial_traj = self._apply_output_forward_offset(
            ego_rollout,
            effective_output_forward_offset,
        )

        raw_output_traj = output_traj
        raw_output_initial_traj = output_initial_traj
        if self.cfg.coord_convention == "pnn_xy":
            output_traj = traj_pnn_to_hipad(output_traj)
            output_initial_traj = traj_pnn_to_hipad(output_initial_traj)

        dense_xy = output_traj[0, :, :2].detach().cpu().numpy().astype(np.float32)
        final_planning = dense_xy[[4, 9, 14, 19, 24, 29]]
        return {
            "final_planning": final_planning,
            "dense_trajectory": output_traj[0].detach().cpu().numpy().astype(np.float32),
            "raw_dense_trajectory": raw_output_traj[0].detach().cpu().numpy().astype(np.float32),
            "control": refined_control[0].detach().cpu().numpy().astype(np.float32),
            "cost_weights": cost_weights[0].detach().cpu().numpy().astype(np.float32),
            "refine_gate": float(refine_gate[0, 0].detach().cpu().item()),
            "initial_final_planning": output_initial_traj[0, [4, 9, 14, 19, 24, 29], :2].detach().cpu().numpy().astype(np.float32),
            "raw_initial_final_planning": raw_output_initial_traj[0, [4, 9, 14, 19, 24, 29], :2].detach().cpu().numpy().astype(np.float32),
            "output_forward_offset": float(self.cfg.output_forward_offset),
            "reference_forward_offset": float(self.cfg.reference_forward_offset),
            "effective_output_forward_offset": float(effective_output_forward_offset),
            "coord_convention": self.cfg.coord_convention,
            "dipp_failed": dipp_failed,
        }

    def refine_hipad_plan(
        self,
        hipad_plan: Any,
        ego_speed: float,
        spatial_plan: Any = None,
        ped_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        veh_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        lane_points: Any = None,
    ) -> Dict[str, Any]:
        sample = self.build_sample(
            hipad_plan=hipad_plan,
            ego_speed=ego_speed,
            spatial_plan=spatial_plan,
            ped_agents=ped_agents,
            veh_agents=veh_agents,
            lane_points=lane_points,
        )
        result = self.refine_sample(sample)
        route_targets = sample["route_targets"]
        if self.cfg.coord_convention == "pnn_xy":
            route_targets = pnn_points_to_hipad(route_targets)
        result["route_targets"] = route_targets
        result["route_source"] = "hipad_plan"
        return result

    def refine_navigation_route(
        self,
        navigation_points: Any,
        ego_speed: float,
        hipad_plan: Any = None,
        spatial_plan: Any = None,
        ped_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        veh_agents: Optional[Iterable[Mapping[str, Any]]] = None,
        lane_points: Any = None,
        navigation_min_speed: float = DEFAULT_NAV_MIN_SPEED,
        navigation_max_speed: float = DEFAULT_NAV_MAX_SPEED,
        navigation_distance_scale: Any = 1.0,
        navigation_interpolation: str = "spline",
    ) -> Dict[str, Any]:
        sample = self.build_sample(
            hipad_plan=hipad_plan,
            ego_speed=ego_speed,
            spatial_plan=spatial_plan,
            ped_agents=ped_agents,
            veh_agents=veh_agents,
            lane_points=lane_points,
            navigation_points=navigation_points,
            route_source="navigation",
            navigation_min_speed=navigation_min_speed,
            navigation_max_speed=navigation_max_speed,
            navigation_distance_scale=navigation_distance_scale,
            navigation_interpolation=navigation_interpolation,
        )
        result = self.refine_sample(sample)
        route_targets = sample["route_targets"]
        if self.cfg.coord_convention == "pnn_xy":
            route_targets = pnn_points_to_hipad(route_targets)
        result["route_targets"] = route_targets
        result["route_source"] = "navigation"
        return result


def build_pt_from_records(records: Sequence[Mapping[str, Any]], output_path: str) -> Dict[str, torch.Tensor]:
    """
    Convert saved HiP-AD/Bench2Drive records into a PNN training .pt file.

    Each record should contain at least:
        hipad_plan, ego_speed
    Optional:
        spatial_plan, ped_agents, veh_agents, lane_points, ego_future_gt
    """
    adapter = PNNOptimizerAdapter(PNNAdapterConfig(use_weight_net=False, use_theseus_refine=False, device="cpu"))
    samples = [
        adapter.build_sample(
            hipad_plan=r["hipad_plan"],
            ego_speed=float(r.get("ego_speed", 0.0)),
            spatial_plan=r.get("spatial_plan"),
            ped_agents=r.get("ped_agents"),
            veh_agents=r.get("veh_agents"),
            lane_points=r.get("lane_points"),
        )
        for r in records
    ]
    data = {
        "ego_state": torch.as_tensor(np.stack([s["ego_state"] for s in samples]), dtype=torch.float32),
        "ped_states": torch.as_tensor(np.stack([s["ped_states"] for s in samples]), dtype=torch.float32),
        "veh_states": torch.as_tensor(np.stack([s["veh_states"] for s in samples]), dtype=torch.float32),
        "lane_points": torch.as_tensor(np.stack([s["lane_points"] for s in samples]), dtype=torch.float32),
        "ped_mask": torch.as_tensor(np.stack([s["ped_mask"] for s in samples]), dtype=torch.bool),
        "veh_mask": torch.as_tensor(np.stack([s["veh_mask"] for s in samples]), dtype=torch.bool),
    }
    gt = [r.get("ego_future_gt") for r in records]
    if any(x is not None for x in gt):
        data["ego_future_gt"] = torch.as_tensor(
            np.stack([plan_to_route_targets(x).reshape(3, 2) if x is not None else np.zeros((3, 2), np.float32) for x in gt]),
            dtype=torch.float32,
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data, output_path)
    return data


def _demo(args: argparse.Namespace) -> None:
    plan = np.asarray(args.plan, dtype=np.float32).reshape(-1, 2)
    cfg = PNNAdapterConfig(
        device=args.device,
        control_ckpt_path=args.control_ckpt,
        weight_ckpt_path=args.weight_ckpt,
        stats_path=args.stats,
        use_weight_net=not args.no_weight,
        use_theseus_refine=args.use_theseus,
    )
    adapter = PNNOptimizerAdapter(cfg)
    result = adapter.refine_hipad_plan(plan, ego_speed=args.ego_speed)
    print("input_plan:")
    print(plan)
    print("optimized_final_planning:")
    print(result["final_planning"])
    print("cost_weights:")
    print(result["cost_weights"])
    print("dipp_failed:", result["dipp_failed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone HiP-AD -> PNN optimizer adapter demo.")
    parser.add_argument("--plan", nargs="+", type=float, default=[0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12])
    parser.add_argument("--ego-speed", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH)
    parser.add_argument("--control-ckpt", default=DEFAULT_CONTROL_CKPT)
    parser.add_argument("--weight-ckpt", default=DEFAULT_WEIGHT_CKPT)
    parser.add_argument("--no-weight", action="store_true")
    parser.add_argument("--use-theseus", action="store_true")
    args = parser.parse_args()
    _demo(args)


if __name__ == "__main__":
    main()
