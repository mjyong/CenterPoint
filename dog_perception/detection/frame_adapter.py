"""Det frame (D) <-> network input frame (M).

nuScenes-pretrained CenterPoint has learned absolute-z priors for a lidar
mounted 1.84 m above the ground (LIDAR_TOP), with the vehicle heading along
+y. Instead of fighting those priors, the dog's gravity-aligned det frame
(ground at z=0, x forward) is mapped onto a *virtual nuScenes lidar*:

    x_M = Rz(+90 deg) x_D + [0, 0, z_offset],   z_offset = -1.84

The same mapping is used for fine-tuning data, so the pretrained z / height
heads keep working after fine-tuning too.
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import rot_z, wrap_angle

NUSC_LIDAR_HEIGHT = 1.84019


@dataclass
class ModelFrame:
    nusc_axes: bool = True
    z_offset: float = -NUSC_LIDAR_HEIGHT

    @property
    def yaw(self):
        return np.pi / 2 if self.nusc_axes else 0.0

    @property
    def R(self):
        return rot_z(self.yaw)

    def points_to_model(self, pts):
        out = np.array(pts, dtype=np.float32, copy=True)
        out[:, :3] = pts[:, :3] @ self.R.T.astype(np.float32)
        out[:, 2] += self.z_offset
        return out

    def points_from_model(self, pts):
        out = np.array(pts, dtype=np.float32, copy=True)
        xyz = pts[:, :3].astype(np.float64)
        xyz[:, 2] -= self.z_offset
        out[:, :3] = xyz @ self.R
        return out

    def boxes_from_model(self, boxes, vel):
        """Standard boxes + velocities in M -> D."""
        b = np.array(boxes, dtype=np.float64, copy=True)
        b[:, 2] -= self.z_offset
        b[:, :3] = b[:, :3] @ self.R
        b[:, 6] = wrap_angle(b[:, 6] - self.yaw)
        return b, np.asarray(vel, dtype=np.float64) @ self.R[:2, :2]

    def boxes_to_model(self, boxes, vel):
        b = np.array(boxes, dtype=np.float64, copy=True)
        b[:, :3] = b[:, :3] @ self.R.T
        b[:, 2] += self.z_offset
        b[:, 6] = wrap_angle(b[:, 6] + self.yaw)
        return b, np.asarray(vel, dtype=np.float64) @ self.R[:2, :2].T
