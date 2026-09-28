"""On-disk recording format used by the offline tools.

    seq_dir/
      meta.json            {"T_body_lidar": 4x4, "base_height": 0.45, ...}
      scans/000000.npz     xyz (N,3) f32, intensity (N,) f32, t (N,) f64 absolute, stamp f64
      imu.npy              (M, 4)  t, wx, wy, wz        body-frame gyro [rad/s]
      odom.npy             (K, 11) t, x, y, z, qx, qy, qz, qw, vx, vy, vz   LIO world frame
                           (velocity columns NaN if the LIO does not publish a twist)

``tools/dog/rosbag_to_sequence.py`` converts ROS1/ROS2 bags into this.
"""
import glob
import json
import os

import numpy as np

from .geometry import R_to_quat, quat_to_R
from .preprocess.pipeline import LidarScan


def write_sequence(seq_dir, scans, imu=None, odom=None, meta=None):
    os.makedirs(os.path.join(seq_dir, "scans"), exist_ok=True)
    for i, s in enumerate(scans):
        np.savez_compressed(os.path.join(seq_dir, "scans", "%06d.npz" % i), xyz=s.xyz.astype(np.float32),
                            intensity=np.asarray(s.intensity, np.float32), t=np.asarray(s.point_times, np.float64),
                            stamp=np.float64(s.stamp))
    if imu is not None:
        np.save(os.path.join(seq_dir, "imu.npy"), np.asarray(imu, np.float64))
    if odom is not None:
        np.save(os.path.join(seq_dir, "odom.npy"), np.asarray(odom, np.float64))
    meta = dict(meta or {})
    if "T_body_lidar" in meta:
        meta["T_body_lidar"] = np.asarray(meta["T_body_lidar"]).tolist()
    with open(os.path.join(seq_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def odom_row(t, R, p, v=None):
    v = np.full(3, np.nan) if v is None else np.asarray(v)
    return np.concatenate([[t], p, R_to_quat(R), v])


def load_scan(path):
    d = np.load(path)
    return LidarScan(xyz=d["xyz"], intensity=d["intensity"], point_times=d["t"], stamp=float(d["stamp"]))


class SequenceReader:
    def __init__(self, seq_dir):
        self.dir = seq_dir
        with open(os.path.join(seq_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.scan_files = sorted(glob.glob(os.path.join(seq_dir, "scans", "*.npz")))
        p = os.path.join(seq_dir, "imu.npy")
        self.imu = np.load(p) if os.path.exists(p) else np.zeros((0, 4))
        p = os.path.join(seq_dir, "odom.npy")
        self.odom = np.load(p) if os.path.exists(p) else np.zeros((0, 11))

    @property
    def T_body_lidar(self):
        return np.asarray(self.meta.get("T_body_lidar", np.eye(4)))

    def __len__(self):
        return len(self.scan_files)

    def events(self, odom_latency=0.0, start=0, stop=None):
        """Time-ordered ('imu', t, gyro) / ('odom', t, R, p, v) / ('scan', LidarScan).

        Scans are emitted at their end stamp; odometry for time t is emitted at
        ``t + odom_latency`` (0 replays the best case, a positive value mimics
        the LIO processing delay so the IMU propagation path is exercised).
        """
        files = self.scan_files[start:stop]
        stamps = []
        for f in files:
            with np.load(f) as d:
                stamps.append(float(d["stamp"]))
        ev = [(t, 0, ("imu", t, row[1:4])) for t, row in zip(self.imu[:, 0], self.imu)]
        for row in self.odom:
            v = None if np.any(np.isnan(row[8:11])) else row[8:11]
            ev.append((row[0] + odom_latency, 1, ("odom", row[0], quat_to_R(row[4:8]), row[1:4], v)))
        ev += [(s, 2, ("scan", f)) for s, f in zip(stamps, files)]
        ev.sort(key=lambda e: (e[0], e[1]))
        t_lo = stamps[0] - 1.0 if stamps else -np.inf
        t_hi = stamps[-1] if stamps else np.inf
        for t, _, e in ev:
            if t < t_lo or t > t_hi + 1e-6:
                continue
            if e[0] == "scan":
                yield ("scan", load_scan(e[1]))
            else:
                yield e
