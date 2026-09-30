"""Stage 1: raw XT32 sweep -> detector-ready multi-sweep cloud in the det frame.

Per sweep:
    self-filter (body frame) -> deskew to scan end (IMU-rate poses)
    -> world frame -> push into the sweep buffer
Per frame:
    det frame from the body pose at scan end (gravity aligned, yaw only)
    -> accumulate the last N sweeps into it with a dt channel
"""
import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry import make_T, transform_points
from ..pose_buffer import PoseOutOfRange
from .accumulator import SweepAccumulator
from .deskew import deskew_points
from .frames import DetFrameConfig, GroundHeightEstimator, det_frame_from_body
from .self_filter import SelfFilter, SelfFilterConfig


@dataclass
class LidarScan:
    """One XT32 revolution.

    ``point_times`` are absolute per-point times [s]; the Hesai ROS driver
    publishes them in the ``timestamp`` field. ``stamp`` defaults to the last
    point time (scan end).
    """
    xyz: np.ndarray                 # (N, 3) lidar frame
    intensity: np.ndarray           # (N,)
    point_times: np.ndarray = None  # (N,)
    stamp: float = None

    def __post_init__(self):
        if self.stamp is None:
            if self.point_times is None or len(self.point_times) == 0:
                raise ValueError("need either stamp or point_times")
            self.stamp = float(np.max(self.point_times))


def dedup_returns(scan, resolution=1e-3):
    """Remove repeated points (same xyz on a ``resolution`` grid, first one kept)
    and non-finite points (they would be dropped right after anyway)."""
    fin = np.nonzero(np.isfinite(scan.xyz).all(axis=1))[0]
    lim = (1 << 20) - 1                       # 21 bits per axis: +-1 km at 1 mm
    q = np.clip(np.round(scan.xyz[fin] / resolution), -lim, lim).astype(np.int64) + lim
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, first = np.unique(key, return_index=True)
    if len(first) == len(scan.xyz):
        return scan
    idx = np.sort(fin[first])
    return LidarScan(xyz=scan.xyz[idx], intensity=np.asarray(scan.intensity)[idx],
                     point_times=None if scan.point_times is None else np.asarray(scan.point_times)[idx],
                     stamp=scan.stamp)


@dataclass
class PreprocessConfig:
    T_body_lidar: np.ndarray = field(default_factory=lambda: make_T(t=[0.2, 0.0, 0.15]))
    num_sweeps: int = 5
    max_time_span: float = 0.55
    deskew: bool = True
    time_resolution: float = 1e-4
    # dual-return mode (Hesai "last + strongest") repeats a point whenever both
    # returns coincide; drop exact duplicates (1 mm grid) so point density
    # matches the single-return data the model was trained on
    dedup_returns: bool = True
    intensity_scale: float = 1.0    # nuScenes models expect raw 0..255 intensity
    self_filter: SelfFilterConfig = field(default_factory=SelfFilterConfig)
    det_frame: DetFrameConfig = field(default_factory=DetFrameConfig)


@dataclass
class PreprocessedFrame:
    stamp: float
    points: np.ndarray       # (N, 5) float32 [x, y, z, intensity, dt] in det frame
    T_world_det: np.ndarray
    T_world_body: np.ndarray
    num_raw: int
    num_kept: int
    timings: dict


class Preprocessor:
    def __init__(self, cfg, pose_buffer):
        self.cfg = cfg
        self.poses = pose_buffer
        self.self_filter = SelfFilter(cfg.self_filter, cfg.T_body_lidar)
        self.accumulator = SweepAccumulator(cfg.num_sweeps, cfg.max_time_span)
        self.ground = GroundHeightEstimator(cfg.det_frame) if cfg.det_frame.auto_ground else None

    def reset(self):
        self.accumulator.reset()

    def sweep_to_body(self, scan, dynamic_boxes=None):
        """Self-filter + deskew one sweep. Returns (xyz_body@stamp, intensity)."""
        cfg = self.cfg
        if cfg.dedup_returns:
            scan = dedup_returns(scan)
        finite = np.isfinite(scan.xyz).all(axis=1)
        keep = finite.copy()
        keep[finite] = self.self_filter(scan.xyz[finite], dynamic_boxes)
        xyz = scan.xyz[keep]
        inten = np.asarray(scan.intensity, dtype=np.float32)[keep] * cfg.intensity_scale

        if cfg.deskew and scan.point_times is not None:
            xyz_b = deskew_points(xyz, np.asarray(scan.point_times)[keep], self.poses,
                                  scan.stamp, cfg.T_body_lidar, cfg.time_resolution)
        else:
            xyz_b = transform_points(cfg.T_body_lidar, xyz)
        return xyz_b, inten

    def process(self, scan, dynamic_boxes=None):
        """Returns a ``PreprocessedFrame`` or None if no pose is available yet."""
        t0 = time.perf_counter()
        try:
            xyz_b, inten = self.sweep_to_body(scan, dynamic_boxes)
            T_wb = self.poses.pose_at(scan.stamp)
        except PoseOutOfRange:
            return None
        t1 = time.perf_counter()

        self.accumulator.push(scan.stamp, transform_points(T_wb, xyz_b), inten[:, None])

        offset = self.ground.offset if self.ground is not None else 0.0
        T_wd = det_frame_from_body(T_wb, self.cfg.det_frame.base_height, offset)
        points = self.accumulator.build(T_wd, scan.stamp)
        if self.ground is not None:
            # estimate from the newest sweep only (dt == 0)
            self.ground.update(points[points[:, -1] == 0, :3])
        t2 = time.perf_counter()

        return PreprocessedFrame(
            stamp=scan.stamp, points=points, T_world_det=T_wd, T_world_body=T_wb,
            num_raw=len(scan.xyz), num_kept=len(xyz_b),
            timings={"deskew_ms": 1e3 * (t1 - t0), "accumulate_ms": 1e3 * (t2 - t1)},
        )
