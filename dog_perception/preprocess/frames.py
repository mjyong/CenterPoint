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

from ..geometry import make_T, rot_z, rot_zyx, yaw_of


@dataclass
class DetFrameConfig:
    base_height: float = 0.45          # body origin height above ground when standing [m]
    auto_ground: bool = False          # refine the ground offset from the point cloud
    # also level D on the ground (slow roll / pitch). For pose sources without a
    # usable attitude, e.g. INSPVAX where only the heading is trusted: the body
    # tilt and the lidar mounting tilt then stay in the cloud (1-2 deg on the
    # sample bags, 0.3-0.6 m of ground height error at 20 m)
    auto_level: bool = False
    ground_ring: tuple = (1.5, 8.0)    # radial band used to sample the ground [m]
    ground_band: float = 0.35          # |z| window around the expected ground [m]
    ground_min_points: int = 200
    ground_alpha: float = 0.2          # EMA gain
    ground_max_correction: float = 0.3
    ground_max_tilt: float = np.radians(5.0)


def det_frame_from_body(T_world_body, base_height, ground_offset=0.0, tilt=(0.0, 0.0)):
    """T_world_det from the body pose. ``tilt`` = (roll, pitch) of the ground in
    the unlevelled D, removed by turning D's axes onto the ground."""
    R = T_world_body[:3, :3]
    origin = T_world_body[:3, 3] - np.array([0.0, 0.0, base_height - ground_offset])
    R_d = rot_z(yaw_of(R))
    if tilt[0] or tilt[1]:
        R_d = R_d @ rot_zyx(0.0, tilt[1], tilt[0])
    return make_T(R_d, origin)


class GroundHeightEstimator:
    """Tracks the residual ground height in D (crouching, slopes, stairs) and,
    with ``auto_level``, the residual ground roll / pitch."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.offset = 0.0
        self.tilt = np.zeros(2)     # (roll, pitch) [rad]

    def update(self, xyz_det):
        cfg = self.cfg
        r = np.hypot(xyz_det[:, 0], xyz_det[:, 1])
        m = (r > cfg.ground_ring[0]) & (r < cfg.ground_ring[1]) & (np.abs(xyz_det[:, 2]) < cfg.ground_band)
        if m.sum() < cfg.ground_min_points:
            return self.offset
        if cfg.auto_ground:
            residual = float(np.median(xyz_det[m, 2]))
            self.offset = float(np.clip(self.offset + cfg.ground_alpha * residual,
                                        -cfg.ground_max_correction, cfg.ground_max_correction))
        if cfg.auto_level:
            slope = fit_ground_slope(xyz_det[m])
            if slope is not None:
                # plane z = a x + b y + c  ->  roll = atan(b), pitch = -atan(a)
                res = np.array([np.arctan(slope[1]), -np.arctan(slope[0])])
                self.tilt = np.clip(self.tilt + cfg.ground_alpha * res, -cfg.ground_max_tilt, cfg.ground_max_tilt)
        return self.offset


def fit_ground_slope(xyz, trim=0.08, min_points=100):
    """Trimmed least-squares plane z = a x + b y + c. Returns (a, b) or None.

    The second pass drops points more than ``trim`` off the first plane (object
    feet, curbs, low vegetation that fall inside the ground band)."""
    A = np.column_stack([xyz[:, 0], xyz[:, 1], np.ones(len(xyz))])
    z = xyz[:, 2]
    keep = np.ones(len(z), dtype=bool)
    for _ in range(2):
        if keep.sum() < min_points:
            return None
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        keep = np.abs(z - A @ coef) < trim
    return coef[:2]


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


def median_base_height(heights):
    """Per-sweep ``estimate_base_height`` values (None allowed) -> (median or None, summary).

    One sweep sees the ground around one spot (a kerb, a road lower than the pavement, a slope),
    so single values scatter: first-sweep estimates of the standing kygi robot ranged
    0.59-1.05 m over 29 scenes. The median over a scene does not depend on where it starts."""
    h = np.array([x for x in heights if x is not None], dtype=np.float64)
    if len(h) == 0:
        return None, dict(n=0)
    return float(np.median(h)), dict(n=int(len(h)), p10=round(float(np.percentile(h, 10)), 3),
                                     p90=round(float(np.percentile(h, 90)), 3))
