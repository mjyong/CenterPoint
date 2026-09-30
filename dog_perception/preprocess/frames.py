"""Detection frame ("det" frame, D).

D is gravity aligned (roll = pitch = 0), follows the body heading (yaw), and
has its origin on the ground below the body. Detectors are sensitive to the
absolute z of points and to the tilt of the ground, so this removes the
+-10 deg gait pitch/roll and the body bounce from the network input.

The LIO world frame is already gravity aligned, so building D costs one 4x4
per scan; it is merged into the accumulation transform, not an extra pass.
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import make_T, rot_z, yaw_of


@dataclass
class DetFrameConfig:
    base_height: float = 0.45          # body origin height above ground when standing [m]
    auto_ground: bool = False          # refine the ground offset from the point cloud
    ground_ring: tuple = (1.5, 8.0)    # radial band used to sample the ground [m]
    ground_band: float = 0.35          # |z| window around the expected ground [m]
    ground_min_points: int = 200
    ground_alpha: float = 0.2          # EMA gain
    ground_max_correction: float = 0.3


def det_frame_from_body(T_world_body, base_height, ground_offset=0.0):
    """T_world_det from the body pose."""
    R = T_world_body[:3, :3]
    origin = T_world_body[:3, 3] - np.array([0.0, 0.0, base_height - ground_offset])
    return make_T(rot_z(yaw_of(R)), origin)


class GroundHeightEstimator:
    """Tracks the residual ground height in D (crouching, slopes, stairs)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.offset = 0.0

    def update(self, xyz_det):
        cfg = self.cfg
        r = np.hypot(xyz_det[:, 0], xyz_det[:, 1])
        m = (r > cfg.ground_ring[0]) & (r < cfg.ground_ring[1]) & (np.abs(xyz_det[:, 2]) < cfg.ground_band)
        if m.sum() < cfg.ground_min_points:
            return self.offset
        residual = float(np.median(xyz_det[m, 2]))
        self.offset = float(np.clip(self.offset + cfg.ground_alpha * residual,
                                    -cfg.ground_max_correction, cfg.ground_max_correction))
        return self.offset


def estimate_base_height(xyz_body, ring=(2.0, 15.0), limits=(0.1, 3.0)):
    """Body-origin height above the ground from one (roughly level) sweep.

    The ground is the densest low surface around the robot: take the points
    close to the 5th z-percentile inside a ring and use their median. Returns
    None when the ring holds too few points.
    """
    r = np.hypot(xyz_body[:, 0], xyz_body[:, 1])
    z = xyz_body[(r > ring[0]) & (r < ring[1]) & np.isfinite(xyz_body[:, 2]), 2]
    if len(z) < 500:
        return None
    z5 = np.percentile(z, 5)
    ground = np.median(z[(z > z5 - 0.1) & (z < z5 + 0.25)])
    return float(np.clip(-ground, *limits))
