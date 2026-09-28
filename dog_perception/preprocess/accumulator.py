"""Multi-sweep accumulation with a time-lag channel.

Past sweeps are stored in the (gravity aligned) LIO world frame and moved
into the current det frame on demand, so static structure lines up and only
moving objects leave a "smear" that the velocity head reads together with the
dt channel -- the same input CenterPoint sees on nuScenes (10 sweeps @20 Hz,
dt in [0, 0.45] s). With XT32 @10 Hz, 5 sweeps give dt in [0, 0.4] s and
~5 x 64k = 320k points, which matches nuScenes' ~34k x 10 closely.
"""
from collections import deque

import numpy as np

from ..geometry import inv_T, transform_points


class SweepAccumulator:
    def __init__(self, num_sweeps=5, max_time_span=0.55):
        self.num_sweeps = num_sweeps
        self.max_time_span = max_time_span
        self._sweeps = deque(maxlen=num_sweeps)

    def reset(self):
        self._sweeps.clear()

    def __len__(self):
        return len(self._sweeps)

    def push(self, stamp, xyz_world, features):
        """xyz_world: (N, 3) float64 world points; features: (N, C) e.g. intensity."""
        features = np.asarray(features, dtype=np.float32).reshape(len(xyz_world), -1)
        self._sweeps.append((float(stamp), np.asarray(xyz_world, dtype=np.float64), features))

    def build(self, T_world_det, stamp):
        """Return (N, 3 + C + 1) float32: xyz in det frame, features, dt=stamp - t_sweep."""
        T_det_world = inv_T(T_world_det)
        out = []
        # newest sweep first: when a pillar/voxel overflows max_points, the
        # voxelizer keeps the earliest points, i.e. the freshest ones (as in
        # det3d's nuScenes loader at test time)
        for t_s, xyz_w, feat in reversed(self._sweeps):
            dt = stamp - t_s
            if dt < -1e-6 or dt > self.max_time_span:
                continue
            xyz_d = transform_points(T_det_world, xyz_w).astype(np.float32)
            lag = np.full((len(xyz_d), 1), max(dt, 0.0), dtype=np.float32)
            out.append(np.hstack([xyz_d, feat, lag]))
        if not out:
            c = self._sweeps[0][2].shape[1] if self._sweeps else 1
            return np.zeros((0, 4 + c), dtype=np.float32)
        return np.concatenate(out, axis=0)
