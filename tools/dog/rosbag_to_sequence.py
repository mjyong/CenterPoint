"""Convert a ROS1 / ROS2 bag (Hesai XT32 + IMU + LIO odometry) into the sequence format.

Needs only ``pip install rosbags`` (no ROS installation).

    python tools/dog/rosbag_to_sequence.py --bag rec_001/ --out data/rec_001 \
        --lidar-topic /lidar_points --imu-topic /imu/data --odom-topic /Odometry \
        --extrinsic-t 0.2 0 0.15 --extrinsic-rpy 0 0 0 --base-height 0.45

Frames: "body" is the frame the LIO odometry reports (usually the IMU frame,
e.g. FAST-LIO), so ``--extrinsic-*`` is lidar-in-body (FAST-LIO's
extrinsic_T / extrinsic_R) and the IMU gyro must be expressed in it too.
With ``--inspvax-topic`` (kygi665 RTK unit) the IMU is read as deg/s in the
INS axes, the lidar extrinsic defaults to the measured 180 deg yaw, the INS rows are moved from
the INS point to the body origin (``--ins-lever-arm``, default the measured 0.54 m behind the lidar),
the heading is ``--azimuth-offset`` minus the azimuth (180 deg while the dual antennas were swapped,
as on 0921, 0 once fixed; ``auto`` picks the one that makes the robot walk forwards),
the sequence is marked for ground levelling (see ``dog_perception/inspvax.py``), and the base height
is the median ground height over every ``--base-height-every``-th sweep of the sequence.
Per-point time: Hesai ROS driver 2.0 publishes an absolute float64
``timestamp`` field (``--time-mode absolute``), livox_ros_driver2 absolute
nanoseconds (``absolute_ns``); drivers that publish offsets from the header
stamp use ``relative`` (seconds) or ``relative_ns``.

``--lidar`` picks the kygi665 lidar preset (``dog_perception/lidars.py``): topic, time format,
noise tags, mounting and self filter. ``xt32`` is the Hesai on top (0921), ``mid360`` the Livox at
the front (1016; mounting estimated from the data until calibrated, marked in meta.json); without
``--lidar`` the preset whose topic is ``--lidar-topic`` is used, else ``xt32`` with INSPVAX.
Camera streams (``--camera-topics``, default every CompressedVideo / CompressedImage topic) are
copied unchanged (CDR, ROS 2 message definitions as schemas) into ``<seq>/cameras.mcap`` for the
scene's time window, from ``--camera-pad`` seconds before it so the video starts on a key frame;
``run_sequence.py`` and ``review_labels.py mcap`` put them next to the cloud and the boxes.

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
from dog_perception.lidars import LIDARS, register_compressed_video  # noqa: E402
from dog_perception.preprocess import estimate_base_height, median_base_height  # noqa: E402
from dog_perception.ros_utils import scan_from_msg, stamp_of  # noqa: E402

class Target:
    """One output sequence: scans in [t0, t1), IMU / localization rows in [t0 - pad, t1 + pad)."""

    def __init__(self, name, out_dir, t0=-np.inf, t1=np.inf, pad=1.0, max_scans=None, pool=None):
        self.name, self.out_dir, self.t0, self.t1, self.pad, self.max_scans = name, out_dir, t0, t1, pad, max_scans
        self.pool = pool
        self.writer = None
        self.heights = []        # base height seen by every k-th sweep
        self.pos_types = {}
        self.cams = None         # mcap Writer of <out_dir>/cameras.mcap
        self.cam_channels = {}
        self.cam_count = 0

    def wants_scan(self, t):
        full = self.max_scans is not None and self.writer is not None and self.writer.n_scans >= self.max_scans
        return self.t0 <= t < self.t1 and not full

    def wants_row(self, t):
        return self.t0 - self.pad <= t < self.t1 + self.pad

    def w(self):
        if self.writer is None:
            self.writer = SequenceWriter(self.out_dir, self.pool)
        return self.writer

    def add_camera(self, conn, msgdef, t, raw):
        if self.cams is None:
            from mcap.writer import CompressionType, Writer

            os.makedirs(self.out_dir, exist_ok=True)
            self._cam_fh = open(os.path.join(self.out_dir, "cameras.mcap"), "wb")
            self.cams = Writer(self._cam_fh, compression=CompressionType.NONE)    # H.264 / JPEG: nothing to gain
            self.cams.start(profile="ros2", library="tools/dog/rosbag_to_sequence.py")
        ch = self.cam_channels.get(conn.topic)
        if ch is None:
            sid = self.cams.register_schema(conn.msgtype, "ros2msg", msgdef.encode())
            ch = self.cam_channels[conn.topic] = self.cams.register_channel(conn.topic, "cdr", sid)
        ns = int(round(t * 1e9))
        self.cams.add_message(ch, ns, bytes(raw), ns)
        self.cam_count += 1

    def close_cameras(self):
        if self.cams is not None:
            self.cams.finish()
            self._cam_fh.close()


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
    ap.add_argument("--lidar", default=None, choices=sorted(LIDARS),
                    help="kygi665 lidar preset (dog_perception/lidars.py); default: from --lidar-topic")
    ap.add_argument("--lidar-topic", default=None, help="default: the --lidar preset's topic")
    ap.add_argument("--imu-topic", required=True)
    ap.add_argument("--camera-topics", nargs="*", default=["auto"],
                    help="camera topics copied to <seq>/cameras.mcap; auto: every foxglove CompressedVideo / "
                         "sensor_msgs CompressedImage topic; none: no cameras")
    ap.add_argument("--camera-pad", type=float, default=1.5,
                    help="[s] camera messages kept before each scene (H.264 needs a key frame to start)")
    ap.add_argument("--odom-topic", default=None, help="nav_msgs/Odometry. Omit when --inspvax-topic is set")
    ap.add_argument("--inspvax-topic", default=None,
                    help="kygi665_msgs/INSPVAX used as localization instead of odometry")
    ap.add_argument("--time-field", default=None, help="default: the preset's (timestamp)")
    ap.add_argument("--time-mode", default=None, choices=["absolute", "absolute_ns", "relative", "relative_ns"],
                    help="default: the preset's (xt32: absolute, mid360: absolute_ns)")
    ap.add_argument("--odom-twist", default="none", choices=["none", "body", "world"],
                    help="how to read Odometry.twist (FAST-LIO leaves it empty -> none)")
    ap.add_argument("--extrinsic-t", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--extrinsic-rpy", type=float, nargs=3, default=None,
                    help="radians; default 0 0 0, or the measured kygi value with --inspvax-topic")
    ap.add_argument("--ins-lever-arm", type=float, nargs=3, default=None,
                    help="[m] INSPVAX position point in the body frame; its rows are moved to the body origin "
                         "(default: the value measured on 0921 with the antennas swapped, else the IMU centre)")
    ap.add_argument("--azimuth-offset", default="auto", choices=["auto", "180", "0"],
                    help="[deg] heading = offset - azimuth: 180 with the dual antennas swapped (0921), 0 once "
                         "fixed; auto: whichever makes the robot walk forwards")
    ap.add_argument("--base-height", type=float, default=0.45)
    ap.add_argument("--estimate-base-height", action="store_true",
                    help="replace --base-height by the median ground height over the sweeps "
                         "(always done with --inspvax-topic)")
    ap.add_argument("--base-height-every", type=int, default=10,
                    help="estimate the base height on every k-th sweep of each sequence")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--segments", default=None, help="segments.csv from tools/dog/bag_survey.py (rows with keep=1)")
    ap.add_argument("--only", nargs="*", default=None, help="with --segments: convert only these scene names")
    ap.add_argument("--t-start", type=float, default=None, help="header stamp [s]: first scan to keep")
    ap.add_argument("--t-end", type=float, default=None, help="header stamp [s]: keep scans before this")
    ap.add_argument("--pad", type=float, default=1.0, help="[s] IMU / localization kept around each window")
    ap.add_argument("--write-threads", type=int, default=0,
                    help="threads compressing and writing scans while the bag is read (0: min(8, cores), 1: none)")
    args = ap.parse_args()
    if not args.odom_topic and not args.inspvax_topic:
        ap.error("need --odom-topic or --inspvax-topic")
    kygi = bool(args.inspvax_topic)
    if args.lidar is None:
        args.lidar = next((k for k, v in LIDARS.items() if v.topic == args.lidar_topic), "xt32" if kygi else None)
    preset = LIDARS.get(args.lidar)
    if args.lidar_topic is None:
        if preset is None:
            ap.error("need --lidar-topic or --lidar")
        args.lidar_topic = preset.topic
    args.time_field = args.time_field or (preset.time_field if preset else "timestamp")
    args.time_mode = args.time_mode or (preset.time_mode if preset else "absolute")
    noise_mask = preset.noise_tag_mask if preset else 0
    print("lidar %s on %s (time %s %s%s)" % (preset.name if preset else "generic", args.lidar_topic, args.time_field,
                                              args.time_mode, ", noise tags 0x%02x dropped" % noise_mask if noise_mask else ""))

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.inspvax import (KYGI_AZIMUTH_OFFSET_DEG, EnuOrigin, apply_lever_arm,
                                        default_lever_arm, forward_speed, gyro_from_kygi_imu, pose_from_inspvax,
                                        register_kygi, turn_heading)

    provisional = False
    if args.extrinsic_rpy is None:
        if preset is not None and (kygi or args.lidar != "xt32"):
            args.extrinsic_rpy, provisional = list(preset.rpy), preset.provisional
            if not any(args.extrinsic_t):
                args.extrinsic_t = list(preset.t)
        else:
            args.extrinsic_rpy = [0.0, 0.0, 0.0]
    T_bl = make_T(rot_zyx(args.extrinsic_rpy[2], args.extrinsic_rpy[1], args.extrinsic_rpy[0]), args.extrinsic_t)
    if provisional:
        print("NOTE: %s mounting rpy %s is an estimate from the data (dog_perception/lidars.py); pass the "
              "calibration with --extrinsic-rpy / --extrinsic-t when known" % (preset.name, args.extrinsic_rpy))
    est_h = args.estimate_base_height or kygi

    n_threads = min(8, os.cpu_count() or 1) if args.write_threads <= 0 else args.write_threads
    pool = None
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(n_threads)

    if args.segments:
        segs = read_segments(args.segments, args.only)
        if not segs:
            raise SystemExit("no segments with keep=1 in %s" % args.segments)
        targets = [Target(n, os.path.join(args.out, n), a, b, args.pad, pool=pool) for n, a, b, _ in segs]
        splits = {n: sp for n, _, _, sp in segs}
    else:
        t0 = -np.inf if args.t_start is None else args.t_start
        t1 = np.inf if args.t_end is None else args.t_end
        targets = [Target(os.path.basename(os.path.normpath(args.out)), args.out, t0, t1, args.pad, args.max_scans,
                          pool=pool)]
        splits = {}
    lo, hi = min(t.t0 for t in targets), max(t.t1 for t in targets)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if args.inspvax_topic:
        register_kygi(typestore)
    register_compressed_video(typestore)
    cam_lo = min(args.camera_pad, 30.0)

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
        cams = camera_connections(reader.connections, args.camera_topics)
        msgdefs = {c.msgtype: typestore.generate_msgdef(c.msgtype, ros_version=2)[0] for c in cams}
        cam_ids = {c.id for c in cams}
        if cams:
            print("cameras -> <seq>/cameras.mcap: %s" % ", ".join(c.topic for c in cams))
        for conn, t_log, raw in reader.messages(connections=conns + cams):
            if conn.id in cam_ids:
                msg = reader.deserialize(raw, conn.msgtype)
                ts_ = getattr(msg, "timestamp", None) or getattr(getattr(msg, "header", None), "stamp", None)
                t_cam = ts_.sec + ts_.nanosec * 1e-9 if ts_ is not None else t_log * 1e-9
                for t in targets:
                    if t.t0 - cam_lo <= t_cam < t.t1 + args.pad:
                        t.add_camera(conn, msgdefs[conn.msgtype], t_cam, raw)
                continue
            if conn.topic == args.lidar_topic:
                # cheap pre-check on the header stamp (scan start) before decoding the cloud
                ht = _header_stamp(raw)
                if ht is not None and not any(t.t0 - 0.5 <= ht < t.t1 + 0.5 for t in targets):
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                scan = scan_from_msg(msg, args.time_field, args.time_mode, noise_tag_mask=noise_mask)
                if n_read == 0 and scan.point_times is None:
                    print("WARNING: no per-point time field '%s': deskew will be disabled for this data"
                          % args.time_field)
                key = ht if ht is not None else scan.stamp     # windows are in header stamps, as in bag_survey.py
                for t in targets:
                    if t.wants_scan(key):
                        t.w().add_scan(scan)
                        if est_h and (t.writer.n_scans - 1) % args.base_height_every == 0:
                            t.heights.append(estimate_base_height(transform_points(T_bl, scan.xyz)))
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

    az_off = None
    if kygi:
        rows = [np.asarray(t.writer.odom, np.float64).reshape(-1, 11) for t in targets if t.writer is not None]
        rows = np.concatenate(rows) if rows else np.zeros((0, 11))
        if args.azimuth_offset == "auto":
            fwd, how = forward_speed(rows), "> 0.5 m/s"     # rows were read with KYGI_AZIMUTH_OFFSET_DEG
            if fwd is None:     # a slow recording: walking > 0.15 m/s for at least 2 s (turning in place
                fwd, how = forward_speed(rows, 0.15, min_rows=200), "> 0.15 m/s, less certain"  # moves the INS sideways)
            az_off = KYGI_AZIMUTH_OFFSET_DEG if fwd is None or fwd >= 0 else (KYGI_AZIMUTH_OFFSET_DEG + 180.0) % 360
            print("INS azimuth offset %.0f deg (%s; median forward speed %s m/s%s)" % (
                az_off, "antennas swapped as on 0921" if az_off == 180 else "antennas fixed",
                "-" if fwd is None else "%+.2f" % fwd, "" if fwd is None else " while moving " + how))
            if fwd is None:
                print("WARNING: the robot hardly moves; offset %.0f is the 0921 default. Pass --azimuth-offset as "
                      "decided on the other recordings of the same day" % az_off)
        else:
            az_off = float(args.azimuth_offset)
        if args.ins_lever_arm is None:
            if args.lidar == "xt32":
                args.ins_lever_arm = list(default_lever_arm(az_off))
            else:       # measured for the XT32 origin only; another lidar is another body origin
                args.ins_lever_arm = [0.0, 0.0, 0.0]
                print("NOTE: no INS lever arm known for the %s origin: 0 (pass --ins-lever-arm, the IMU centre "
                      "in the body frame, once the lidar-to-IMU extrinsic is known)" % preset.name)
        print("INS lever arm %s m (body frame)" % args.ins_lever_arm)

    written = 0
    for t in targets:
        t.close_cameras()
        if t.writer is None or t.writer.n_scans == 0:
            print("%s: no scans in [%.3f, %.3f) - skipped" % (t.name, t.t0, t.t1))
            continue
        if not t.writer.odom:
            print("%s: no localization rows (INS solution not good, or empty odom topic) - skipped" % t.name)
            continue
        meta = dict(T_body_lidar=T_bl, base_height=args.base_height, lidar=preset.name if preset else "generic",
                    source=str(args.bag))
        if provisional:
            meta["extrinsic_provisional"] = True
        if preset is not None and preset.self_filter:
            meta["self_filter"] = preset.self_filter
        if t.cam_count:
            meta["cameras"] = sorted(t.cam_channels)
        if np.isfinite(t.t0) or np.isfinite(t.t1):
            meta["segment"] = dict(name=t.name, t_start=t.t0 if np.isfinite(t.t0) else None,
                                   t_end=t.t1 if np.isfinite(t.t1) else None, split=splits.get(t.name, ""))
        if origin is not None:
            meta.update(lat0=origin.lat, lon0=origin.lon, h0=origin.h, inspvax_pos_type=t.pos_types)
        if kygi:
            meta["det_frame"] = dict(auto_ground=True, auto_level=True)    # INS attitude is heading only
            meta["ins_lever_arm"] = [float(a) for a in args.ins_lever_arm]  # already applied to odom.npy
            meta["ins_azimuth_offset_deg"] = az_off                         # heading = offset - azimuth
            meta["imu_axes"] = "body"                                        # gyro_from_kygi_imu
        h, info = median_base_height(t.heights)
        if h is not None:
            meta["base_height"] = h
            meta["base_height_sweeps"] = dict(info, every=args.base_height_every)
        odom = np.asarray(t.writer.odom, np.float64).reshape(-1, 11)
        if kygi:
            odom = apply_lever_arm(turn_heading(odom, np.radians(az_off - KYGI_AZIMUTH_OFFSET_DEG)), args.ins_lever_arm)
            fwd = forward_speed(odom)
            if fwd is not None and fwd < 0:
                print("WARNING %s: the robot walks backwards in the body frame (median %.2f m/s): wrong "
                      "--azimuth-offset for this recording (see dog_perception/inspvax.py)" % (t.name, fwd))
        print("%s: odom t %.3f .. %.3f  span %.1fs  pos_type %s  base_height %.2f" % (
            t.name, odom[0, 0], odom[-1, 0], odom[-1, 0] - odom[0, 0], t.pos_types or "-", meta["base_height"]))
        n_imu = len(t.writer.imu)
        t.writer.close(meta, odom=odom)
        written += 1
        print("wrote %d scans, %d imu, %d odom%s -> %s" % (
            t.writer.n_scans, n_imu, len(odom), ", %d camera messages" % t.cam_count if t.cam_count else "",
            t.out_dir))
    if pool is not None:
        pool.shutdown()
    if not written:
        raise SystemExit("nothing written")


CAMERA_TYPES = ("foxglove_msgs/msg/CompressedVideo", "sensor_msgs/msg/CompressedImage")


def camera_connections(connections, topics):
    """Bag connections of the camera topics: ``["auto"]`` = every CAMERA_TYPES topic, ``["none"]`` or
    ``[]`` = none, else the named topics."""
    if not topics or topics == ["none"]:
        return []
    if topics == ["auto"]:
        return [c for c in connections if c.msgtype in CAMERA_TYPES]
    missing = set(topics) - {c.topic for c in connections}
    if missing:
        raise SystemExit("camera topics not in bag: %s" % sorted(missing))
    return [c for c in connections if c.topic in set(topics)]


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
