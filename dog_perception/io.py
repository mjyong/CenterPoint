"""On-disk recording format used by the offline tools.

    seq_dir/
      meta.json            {"T_body_lidar": 4x4, "base_height": 0.45, ...}
                           optional "lidar_time_offset": seconds added to every lidar time when
                           reading (lidar clock -> IMU / localization clock); see
                           tools/dog/time_offset_check.py
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


class SequenceWriter:
    """Writes a sequence one scan at a time, so a long bag never has to fit in memory. IMU / odom
    rows are collected (small) and written by ``close`` together with ``meta.json``; a sequence
    without ``meta.json`` was not finished.

    With ``pool`` (a ``concurrent.futures`` thread pool) the scans are compressed and written in its
    threads (zlib releases the GIL: compression took half of the converter's time), at most
    ``max_pending`` at a time so memory stays bounded; the files are the same."""

    def __init__(self, seq_dir, pool=None, max_pending=16):
        self.dir = seq_dir
        os.makedirs(os.path.join(seq_dir, "scans"), exist_ok=True)
        self.n_scans = 0
        self.imu, self.odom = [], []
        self.pool, self.max_pending, self._pending = pool, max_pending, []

    def add_scan(self, s):
        args = (os.path.join(self.dir, "scans", "%06d.npz" % self.n_scans),)
        arrays = dict(xyz=s.xyz.astype(np.float32), intensity=np.asarray(s.intensity, np.float32),
                      t=np.asarray(s.point_times, np.float64), stamp=np.float64(s.stamp))
        self.n_scans += 1
        if self.pool is None:
            np.savez_compressed(*args, **arrays)
            return
        self._pending.append(self.pool.submit(np.savez_compressed, *args, **arrays))
        while len(self._pending) > self.max_pending:
            self._pending.pop(0).result()

    def flush(self):
        """Wait for the scans still being written (raises their errors)."""
        while self._pending:
            self._pending.pop(0).result()

    def close(self, meta=None, imu=None, odom=None):
        self.flush()
        imu = self.imu if imu is None else imu
        odom = self.odom if odom is None else odom
        if len(imu):
            np.save(os.path.join(self.dir, "imu.npy"), np.asarray(imu, np.float64).reshape(-1, 4))
        if len(odom):
            np.save(os.path.join(self.dir, "odom.npy"), np.asarray(odom, np.float64).reshape(-1, 11))
        meta = dict(meta or {})
        if "T_body_lidar" in meta:
            meta["T_body_lidar"] = np.asarray(meta["T_body_lidar"]).tolist()
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def write_sequence(seq_dir, scans, imu=None, odom=None, meta=None):
    w = SequenceWriter(seq_dir)
    for s in scans:
        w.add_scan(s)
    w.close(meta, [] if imu is None else imu, [] if odom is None else odom)


def odom_row(t, R, p, v=None):
    v = np.full(3, np.nan) if v is None else np.asarray(v)
    return np.concatenate([[t], p, R_to_quat(R), v])


def pose_bad(meta, stamp):
    """True when ``stamp`` lies in one of the sequence's ``pose_refine.bad_intervals`` (neither the
    INS nor the lidar odometry could be trusted there, tools/dog/refine_poses.py): such frames are
    left out of auto-labelling and the exported datasets."""
    return any(g[0] <= stamp <= g[1] for g in (meta.get("pose_refine") or {}).get("bad_intervals", []))


def load_scan(path):
    d = np.load(path)
    return LidarScan(xyz=d["xyz"], intensity=d["intensity"], point_times=d["t"], stamp=float(d["stamp"]))


class SequenceReader:
    """``lidar_time_offset`` overrides ``meta["lidar_time_offset"]`` (default 0): it is added to
    the scan stamps and point times that ``events`` yields, so every tool downstream (preview,
    auto-label, dataset export) sees the lidar on the localization clock. The files keep the
    times as recorded."""

    def __init__(self, seq_dir, lidar_time_offset=None):
        self.dir = seq_dir
        with open(os.path.join(seq_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.lidar_time_offset = float(self.meta.get("lidar_time_offset", 0.0)
                                       if lidar_time_offset is None else lidar_time_offset)
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

    def events(self, odom_latency=0.0, start=0, stop=None, indices=None):
        """Time-ordered ('imu', t, gyro) / ('odom', t, R, p, v) / ('scan', LidarScan).

        Scans are emitted at their end stamp; odometry for time t is emitted at
        ``t + odom_latency`` (0 replays the best case, a positive value mimics
        the LIO processing delay so the IMU propagation path is exercised).
        ``indices`` (sorted scan indices) loads only those scans.
        """
        files = self.scan_files[start:stop] if indices is None else [self.scan_files[i] for i in indices]
        dt = self.lidar_time_offset
        stamps = []
        for f in files:
            with np.load(f) as d:
                stamps.append(float(d["stamp"]) + dt)
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
                yield ("scan", load_scan(e[1]).shifted(dt))
            else:
                yield e
