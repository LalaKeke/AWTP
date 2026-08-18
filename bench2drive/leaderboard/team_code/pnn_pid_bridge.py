"""Convert the native PNN rollout into the two plans expected by HiP-AD PID."""

import numpy as np


class PNNHiPADPIDBridge:
    """Keep PNN geometry, but preserve HiP-AD's temporal/spatial PID interface."""

    def __init__(self, dt=0.1, temporal_hz=2.0, spatial_step_m=2.0):
        self.dt = float(dt)
        self.temporal_hz = float(temporal_hz)
        self.spatial_step_m = float(spatial_step_m)

    @staticmethod
    def _interp_axis(axis, values, query):
        axis = np.asarray(axis, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        query = np.asarray(query, dtype=np.float32)
        keep = np.concatenate(([True], np.diff(axis) > 1e-5))
        axis = axis[keep]
        values = values[keep]
        if len(axis) == 1:
            return np.repeat(values[:1], len(query), axis=0)
        return np.stack(
            [np.interp(query, axis, values[:, col]) for col in range(values.shape[1])],
            axis=1,
        ).astype(np.float32)

    def build(self, dense_trajectory):
        dense = np.asarray(dense_trajectory, dtype=np.float32)
        if dense.shape != (30, 4):
            raise ValueError("PNN dense trajectory must have shape [30, 4]")

        xy = dense[:, :2]
        # PNN points are at 0.1 s. The official PID expects six 0.5 s points.
        time_axis = self.dt * np.arange(1, len(xy) + 1, dtype=np.float32)
        query_time = np.arange(1, 7, dtype=np.float32) / self.temporal_hz
        temporal = self._interp_axis(time_axis, xy, query_time)

        # Build the spatial plan by arc length, rather than selecting temporal
        # indices. This keeps the official PID's 2 m steering semantics intact.
        points = np.vstack((np.zeros((1, 2), dtype=np.float32), xy))
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(segment))).astype(np.float32)
        query_arc = self.spatial_step_m * np.arange(1, 7, dtype=np.float32)
        spatial = self._interp_axis(arc, points, query_arc)

        metadata = {
            "controller": "pnn_hipad_official_pid_bridge_v1",
            "bridge_dt": self.dt,
            "bridge_temporal_hz": self.temporal_hz,
            "bridge_spatial_step_m": self.spatial_step_m,
            "bridge_temporal_shape": list(temporal.shape),
            "bridge_spatial_shape": list(spatial.shape),
            "bridge_dense_arc_length_m": float(arc[-1]),
        }
        return temporal, spatial, metadata
