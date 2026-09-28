"""Removal of the robot's own body/legs/wheels and near-field spray.

Must run in the body frame *before* gravity alignment: the legs move with
the body, not with gravity. It also runs before deskew because it only
needs the static extrinsic, which keeps it cheap.
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import transform_points


@dataclass
class SelfFilterConfig:
    # Axis-aligned boxes in the body frame: [xmin, xmax, ymin, ymax, zmin, zmax].
    # Size them to the full swing envelope of legs/wheels, not the standing pose.
    body_boxes: tuple = ((-0.60, 0.60, -0.40, 0.40, -0.80, 0.30),)
    min_range: float = 0.4      # lidar-frame range [m]; also kills XT32 near-field mixed pixels
    max_range: float = 120.0
    # Near-field sparse-outlier removal (legs produce isolated "spray" points
    # at the edges of the envelope). Only applied inside outlier_max_range to
    # keep the cost low and never touch sparse far-away targets.
    outlier_radius: float = 0.2
    outlier_min_neighbors: int = 2
    outlier_max_range: float = 3.0


def points_in_boxes_aabb(xyz, boxes):
    mask = np.zeros(len(xyz), dtype=bool)
    for b in boxes:
        mask |= (
            (xyz[:, 0] >= b[0]) & (xyz[:, 0] <= b[1])
            & (xyz[:, 1] >= b[2]) & (xyz[:, 1] <= b[3])
            & (xyz[:, 2] >= b[4]) & (xyz[:, 2] <= b[5])
        )
    return mask


def points_in_boxes_oriented(xyz, boxes):
    """boxes: (M, 7) [cx, cy, cz, dx, dy, dz, yaw] (yaw about z)."""
    mask = np.zeros(len(xyz), dtype=bool)
    for cx, cy, cz, dx, dy, dz, yaw in np.asarray(boxes).reshape(-1, 7):
        c, s = np.cos(yaw), np.sin(yaw)
        rel = xyz - np.array([cx, cy, cz])
        lx = c * rel[:, 0] + s * rel[:, 1]
        ly = -s * rel[:, 0] + c * rel[:, 1]
        mask |= (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) & (np.abs(rel[:, 2]) <= dz / 2)
    return mask


def neighbor_counts(xyz, radius):
    """Approximate neighbour count per point using a voxel hash.

    Counts the points inside the 3x3x3 block of ``radius``-sized voxels around
    each point (a cube of side 3*radius), minus the point itself.
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.floor(xyz / radius).astype(np.int64)
    keys -= keys.min(axis=0)
    keys += 1
    dims = keys.max(axis=0) + 2
    s1, s2 = dims[1] * dims[2], dims[2]
    code = keys[:, 0] * s1 + keys[:, 1] * s2 + keys[:, 2]
    ucode, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    total = np.zeros(len(ucode), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nb = ucode + dx * s1 + dy * s2 + dz
                pos = np.clip(np.searchsorted(ucode, nb), 0, len(ucode) - 1)
                total += np.where(ucode[pos] == nb, counts[pos], 0)
    return total[inv] - 1


class SelfFilter:
    def __init__(self, cfg, T_body_lidar):
        self.cfg = cfg
        self.T_body_lidar = np.asarray(T_body_lidar, dtype=np.float64)

    def __call__(self, xyz_lidar, dynamic_boxes=None):
        """Return a keep-mask for (N, 3) lidar-frame points.

        ``dynamic_boxes``: optional (M, 7) oriented boxes in the body frame,
        e.g. per-leg boxes built from joint encoders, for robots whose leg
        envelope is too large for a static mask.
        """
        cfg = self.cfg
        rng = np.linalg.norm(xyz_lidar, axis=1)
        keep = (rng >= cfg.min_range) & (rng <= cfg.max_range)

        xyz_body = transform_points(self.T_body_lidar, xyz_lidar)
        keep &= ~points_in_boxes_aabb(xyz_body, cfg.body_boxes)
        if dynamic_boxes is not None and len(dynamic_boxes):
            keep &= ~points_in_boxes_oriented(xyz_body, dynamic_boxes)

        if cfg.outlier_min_neighbors > 0 and cfg.outlier_max_range > 0:
            near = keep & (rng < cfg.outlier_max_range)
            idx = np.nonzero(near)[0]
            if len(idx):
                cnt = neighbor_counts(xyz_lidar[idx], cfg.outlier_radius)
                keep[idx[cnt < cfg.outlier_min_neighbors]] = False
        return keep
