"""Shared temporal alignment for HiP-AD motion predictions consumed by PNN."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


HIPAD_MOTION_DT = 0.1
PNN_ACTOR_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ALIGNMENT_VERSION = "hipad_motion_10hz_to_pnn_2hz_measured_velocity_tail_v3"


def _finite_velocity(value: Optional[Any]) -> Optional[np.ndarray]:
    if value is None:
        return None
    velocity = np.asarray(value, dtype=np.float32).reshape(-1)
    if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
        return None
    return velocity[:2].astype(np.float32)


def align_hipad_motion_future(
    current_xy: Any,
    future: Any,
    *,
    source_dt: float = HIPAD_MOTION_DT,
    target_times: Sequence[float] = PNN_ACTOR_TIMES,
    measured_velocity_xy: Optional[Any] = None,
    max_speed: float = 20.0,
) -> np.ndarray:
    """Convert HiP-AD's 10 Hz short motion future to PNN's 2 Hz horizon.

    HiP-AD predicts six absolute positions at 0.1--0.6 s. Values inside that
    interval are interpolated directly. Later positions primarily use the
    decoded box velocity. The short predicted trajectory only provides a
    bounded correction because small 10 Hz position jitter otherwise becomes
    a very large 3 s displacement.
    """
    current = np.asarray(current_xy, dtype=np.float32).reshape(-1)
    points = np.asarray(future, dtype=np.float32)
    times = np.asarray(target_times, dtype=np.float32)
    source_dt = float(source_dt)
    if current.size < 2 or not np.isfinite(current[:2]).all():
        raise ValueError("current_xy must contain two finite coordinates")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"future must have shape [T,2] with T > 0, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("future contains non-finite coordinates")
    if source_dt <= 0.0:
        raise ValueError(f"source_dt must be positive, got {source_dt}")
    if times.ndim != 1 or len(times) == 0 or np.any(times <= 0.0):
        raise ValueError(f"target_times must be positive, got {target_times}")

    current = current[:2].astype(np.float32)
    measured_velocity = _finite_velocity(measured_velocity_xy)
    measured_speed = (
        float(np.linalg.norm(measured_velocity))
        if measured_velocity is not None
        else None
    )

    # Bound the observed 10 Hz motion by the decoded box speed when available.
    # A small additive allowance preserves real acceleration and prediction
    # noise without allowing a nearly stationary actor to jump several metres.
    observed_speed_cap = float(max_speed)
    if measured_speed is not None:
        observed_speed_cap = min(
            observed_speed_cap,
            max(0.5, 1.5 * measured_speed + 0.5),
        )
    sanitized = np.empty_like(points)
    previous = current
    max_segment = observed_speed_cap * source_dt
    for index, point in enumerate(points):
        delta = point - previous
        distance = float(np.linalg.norm(delta))
        if distance > max_segment:
            delta = delta * (max_segment / max(distance, 1e-6))
        sanitized[index] = previous + delta
        previous = sanitized[index]

    observed = np.concatenate([current[None], sanitized], axis=0)
    observed_times = np.arange(len(observed), dtype=np.float32) * source_dt

    segment_velocity = np.diff(observed, axis=0) / source_dt
    tail_count = min(3, len(segment_velocity))
    tail_velocity = np.median(segment_velocity[-tail_count:], axis=0).astype(np.float32)
    tail_speed = float(np.linalg.norm(tail_velocity))
    if measured_velocity is not None:
        if measured_speed < 0.2:
            tail_velocity = np.zeros(2, dtype=np.float32)
        elif measured_speed < 0.75:
            tail_velocity = measured_velocity
        elif tail_speed >= 0.05:
            cosine = float(np.dot(tail_velocity, measured_velocity)) / max(
                tail_speed * measured_speed, 1e-6
            )
            if cosine > 0.5:
                tail_velocity = 0.8 * measured_velocity + 0.2 * tail_velocity
            else:
                tail_velocity = measured_velocity
        else:
            tail_velocity = measured_velocity

        # Never extrapolate much faster than the measured actor speed. This is
        # the critical guard for stopped vehicles and hard-braking scenarios.
        tail_cap = min(float(max_speed), max(0.5, 1.5 * measured_speed + 0.5))
        tail_speed = float(np.linalg.norm(tail_velocity))
        if tail_speed > tail_cap:
            tail_velocity = tail_velocity * (tail_cap / max(tail_speed, 1e-6))

    speed = float(np.linalg.norm(tail_velocity))
    if speed > float(max_speed):
        tail_velocity = tail_velocity * (float(max_speed) / max(speed, 1e-6))

    last_time = float(observed_times[-1])
    last_point = observed[-1]
    aligned = np.empty((len(times), 2), dtype=np.float32)
    for axis in range(2):
        aligned[:, axis] = np.interp(
            np.minimum(times, last_time), observed_times, observed[:, axis]
        )
    after = times > last_time
    if np.any(after):
        aligned[after] = last_point + (times[after] - last_time)[:, None] * tail_velocity[None]
    return aligned.astype(np.float32)
