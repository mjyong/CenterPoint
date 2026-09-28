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
