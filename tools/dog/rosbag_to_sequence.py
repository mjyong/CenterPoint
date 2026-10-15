"""Convert a ROS1 / ROS2 bag (Hesai XT32 + IMU + LIO odometry) into the sequence format.

Needs only ``pip install rosbags`` (no ROS installation).

    python tools/dog/rosbag_to_sequence.py --bag rec_001/ --out data/rec_001 \
        --lidar-topic /lidar_points --imu-topic /imu/data --odom-topic /Odometry \
        --extrinsic-t 0.2 0 0.15 --extrinsic-rpy 0 0 0 --base-height 0.45

Frames: "body" is the frame the LIO odometry reports (usually the IMU frame,
e.g. FAST-LIO), so ``--extrinsic-*`` is lidar-in-body (FAST-LIO's
extrinsic_T / extrinsic_R) and the IMU gyro must be expressed in it too.
With ``--inspvax-topic`` (kygi665 RTK unit) the IMU is read as deg/s in the
INS axes, the lidar extrinsic defaults to the measured 180 deg yaw and the
sequence is marked for ground levelling (see ``dog_perception/inspvax.py``).
Per-point time: Hesai ROS driver 2.0 publishes an absolute float64
``timestamp`` field (``--time-mode absolute``); drivers that publish offsets
from the header stamp use ``relative`` (seconds) or ``relative_ns``.

Long bags: scans are written as they are read (memory stays small). With
``--segments`` (the ``segments.csv`` of ``tools/dog/bag_survey.py``, rows with
keep = 1) one pass over the bag writes one sequence per scene to
``<out>/<name>``: scans whose header stamp is in ``[t_start, t_end)`` and IMU / localization
rows padded by ``--pad`` seconds on both sides (deskew and the pose buffer need
them). All scenes of a bag share one ENU origin (the first good INS fix), so
their world frames line up; ``--only`` converts a subset. ``--t-start`` /
``--t-end`` (header stamps, seconds) cut a single window instead.

    python tools/dog/rosbag_to_sequence.py --bag /data/rosbag2_... --inspvax-topic /rte/rdap/rtk/inspvax \
        --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
        --segments data/survey/<bag>/segments.csv --out data/seq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.geometry import make_T, quat_to_R, rot_zyx, transform_points  # noqa: E402
from dog_perception.io import SequenceWriter, odom_row  # noqa: E402
from dog_perception.preprocess import estimate_base_height  # noqa: E402
from dog_perception.ros_utils import scan_from_msg, stamp_of  # noqa: E402

class Target:
    """One output sequence: scans in [t0, t1), IMU / localization rows in [t0 - pad, t1 + pad)."""

    def __init__(self, name, out_dir, t0=-np.inf, t1=np.inf, pad=1.0, max_scans=None):
        self.name, self.out_dir, self.t0, self.t1, self.pad, self.max_scans = name, out_dir, t0, t1, pad, max_scans
        self.writer = None
        self.first_xyz = None
        self.pos_types = {}

    def wants_scan(self, t):
        full = self.max_scans is not None and self.writer is not None and self.writer.n_scans >= self.max_scans
        return self.t0 <= t < self.t1 and not full

    def wants_row(self, t):
        return self.t0 - self.pad <= t < self.t1 + self.pad

    def w(self):
        if self.writer is None:
            self.writer = SequenceWriter(self.out_dir)
        return self.writer


def read_segments(path, only=None):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("keep", "1")).strip() not in ("0", "")]
    if only:
        rows = [r for r in rows if r["name"] in set(only)]
    return [(r["name"], float(r["t_start"]), float(r["t_end"]), r.get("split", "")) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="ROS2 bag directory or ROS1 .bag")
    ap.add_argument("--out", required=True, help="sequence directory (with --segments: the parent of one per scene)")
    ap.add_argument("--lidar-topic", required=True)
    ap.add_argument("--imu-topic", required=True)
    ap.add_argument("--odom-topic", default=None, help="nav_msgs/Odometry. Omit when --inspvax-topic is set")
    ap.add_argument("--inspvax-topic", default=None,
                    help="kygi665_msgs/INSPVAX used as localization instead of odometry")
    ap.add_argument("--time-field", default="timestamp")
    ap.add_argument("--time-mode", default="absolute", choices=["absolute", "relative", "relative_ns"])
    ap.add_argument("--odom-twist", default="none", choices=["none", "body", "world"],
                    help="how to read Odometry.twist (FAST-LIO leaves it empty -> none)")
    ap.add_argument("--extrinsic-t", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--extrinsic-rpy", type=float, nargs=3, default=None,
                    help="radians; default 0 0 0, or the measured kygi value with --inspvax-topic")
    ap.add_argument("--base-height", type=float, default=0.45)
    ap.add_argument("--estimate-base-height", action="store_true",
                    help="replace --base-height by the ground height seen in the first sweep "
                         "(always done with --inspvax-topic)")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--segments", default=None, help="segments.csv from tools/dog/bag_survey.py (rows with keep=1)")
    ap.add_argument("--only", nargs="*", default=None, help="with --segments: convert only these scene names")
    ap.add_argument("--t-start", type=float, default=None, help="header stamp [s]: first scan to keep")
    ap.add_argument("--t-end", type=float, default=None, help="header stamp [s]: keep scans before this")
    ap.add_argument("--pad", type=float, default=1.0, help="[s] IMU / localization kept around each window")
    args = ap.parse_args()
    if not args.odom_topic and not args.inspvax_topic:
        ap.error("need --odom-topic or --inspvax-topic")
    kygi = bool(args.inspvax_topic)

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.inspvax import KYGI_LIDAR_RPY, EnuOrigin, gyro_from_kygi_imu, pose_from_inspvax, register_kygi

    if args.extrinsic_rpy is None:
        args.extrinsic_rpy = list(KYGI_LIDAR_RPY) if kygi else [0.0, 0.0, 0.0]

    if args.segments:
        segs = read_segments(args.segments, args.only)
        if not segs:
            raise SystemExit("no segments with keep=1 in %s" % args.segments)
        targets = [Target(n, os.path.join(args.out, n), a, b, args.pad) for n, a, b, _ in segs]
        splits = {n: sp for n, _, _, sp in segs}
    else:
        t0 = -np.inf if args.t_start is None else args.t_start
        t1 = np.inf if args.t_end is None else args.t_end
        targets = [Target(os.path.basename(os.path.normpath(args.out)), args.out, t0, t1, args.pad, args.max_scans)]
        splits = {}
    lo, hi = min(t.t0 for t in targets), max(t.t1 for t in targets)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if args.inspvax_topic:
        register_kygi(typestore)

    origin = None
    n_read = 0
    topics = {args.lidar_topic, args.imu_topic}
    topics.add(args.inspvax_topic if args.inspvax_topic else args.odom_topic)
    with AnyReader([Path(args.bag)], default_typestore=typestore) as reader:
        print("topics:")
        for c in reader.connections:
            print("  %s  %s  %d" % (c.topic, c.msgtype, c.msgcount))
        conns = [c for c in reader.connections if c.topic in topics]
        missing = topics - {c.topic for c in conns}
        if missing:
            raise SystemExit("topics not in bag: %s (have: %s)" % (missing, sorted({c.topic for c in reader.connections})))
        for conn, _, raw in reader.messages(connections=conns):
            if conn.topic == args.lidar_topic:
                # cheap pre-check on the header stamp (scan start) before decoding the cloud
                ht = _header_stamp(raw)
                if ht is not None and not any(t.t0 - 0.5 <= ht < t.t1 + 0.5 for t in targets):
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                scan = scan_from_msg(msg, args.time_field, args.time_mode)
                if n_read == 0 and scan.point_times is None:
                    print("WARNING: no per-point time field '%s': deskew will be disabled for this data"
                          % args.time_field)
                key = ht if ht is not None else scan.stamp     # windows are in header stamps, as in bag_survey.py
                for t in targets:
                    if t.wants_scan(key):
                        t.w().add_scan(scan)
                        if t.first_xyz is None:
                            t.first_xyz = scan.xyz
                n_read += 1
                if n_read % 1000 == 0:
                    print("  %d scans decoded, t = %.1f" % (n_read, scan.stamp), flush=True)
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == args.imu_topic:
                t_msg = stamp_of(msg.header)
                if not lo - args.pad <= t_msg < hi + args.pad:
                    continue
                w = msg.angular_velocity
                g = gyro_from_kygi_imu(msg) if kygi else (w.x, w.y, w.z)
                for t in targets:
                    if t.wants_row(t_msg):
                        t.w().imu.append((t_msg, g[0], g[1], g[2]))
            elif args.inspvax_topic and conn.topic == args.inspvax_topic:
                if origin is None and int(msg.ins_status.data) == 3:
                    origin = EnuOrigin(msg.latitude, msg.longitude, msg.height)
                got = pose_from_inspvax(msg, origin) if origin is not None else None
                if got is None:
                    continue
                tm, R, p, v = got
                for t in targets:
                    if t.wants_row(tm):
                        t.w().odom.append(odom_row(tm, R, p, v))
                        k = int(msg.pos_type.data)
                        t.pos_types[k] = t.pos_types.get(k, 0) + 1
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                R = quat_to_R([q.x, q.y, q.z, q.w])
                v = None
                if args.odom_twist != "none":
                    lv = msg.twist.twist.linear
                    v = np.array([lv.x, lv.y, lv.z])
                    if args.odom_twist == "body":
                        v = R @ v
                tm = stamp_of(msg.header)
                for t in targets:
                    if t.wants_row(tm):
                        t.w().odom.append(odom_row(tm, R, [p.x, p.y, p.z], v))

    T_bl = make_T(rot_zyx(args.extrinsic_rpy[2], args.extrinsic_rpy[1], args.extrinsic_rpy[0]), args.extrinsic_t)
    written = 0
    for t in targets:
        if t.writer is None or t.writer.n_scans == 0:
            print("%s: no scans in [%.3f, %.3f) - skipped" % (t.name, t.t0, t.t1))
            continue
        if not t.writer.odom:
            print("%s: no localization rows (INS solution not good, or empty odom topic) - skipped" % t.name)
            continue
        meta = dict(T_body_lidar=T_bl, base_height=args.base_height, lidar="XT32", source=str(args.bag))
        if np.isfinite(t.t0) or np.isfinite(t.t1):
            meta["segment"] = dict(name=t.name, t_start=t.t0 if np.isfinite(t.t0) else None,
                                   t_end=t.t1 if np.isfinite(t.t1) else None, split=splits.get(t.name, ""))
        if origin is not None:
            meta.update(lat0=origin.lat, lon0=origin.lon, h0=origin.h, inspvax_pos_type=t.pos_types)
        if kygi:
            meta["det_frame"] = dict(auto_ground=True, auto_level=True)    # INS attitude is heading only
        if (args.estimate_base_height or origin is not None) and t.first_xyz is not None:
            xyz = t.first_xyz[np.isfinite(t.first_xyz).all(1)]
            h = estimate_base_height(transform_points(T_bl, xyz))
            if h is not None:
                meta["base_height"] = h
        odom = np.asarray(t.writer.odom, np.float64).reshape(-1, 11)
        print("%s: odom t %.3f .. %.3f  span %.1fs  pos_type %s  base_height %.2f" % (
            t.name, odom[0, 0], odom[-1, 0], odom[-1, 0] - odom[0, 0], t.pos_types or "-", meta["base_height"]))
        n_imu = len(t.writer.imu)
        t.writer.close(meta)
        written += 1
        print("wrote %d scans, %d imu, %d odom -> %s" % (t.writer.n_scans, n_imu, len(odom), t.out_dir))
    if not written:
        raise SystemExit("nothing written")


def _header_stamp(raw):
    try:
        if raw[1] != 1:                     # only little-endian CDR
            return None
        import struct
        sec, nsec = struct.unpack_from("<iI", raw, 4)
        return sec + nsec * 1e-9
    except Exception:
        return None


if __name__ == "__main__":
    main()
