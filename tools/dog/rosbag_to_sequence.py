"""Convert a ROS1 / ROS2 bag (Hesai XT32 + IMU + LIO odometry) into the sequence format.

Needs only ``pip install rosbags`` (no ROS installation).

    python tools/dog/rosbag_to_sequence.py --bag rec_001/ --out data/rec_001 \
        --lidar-topic /lidar_points --imu-topic /imu/data --odom-topic /Odometry \
        --extrinsic-t 0.2 0 0.15 --extrinsic-rpy 0 0 0 --base-height 0.45

Frames: "body" is the frame the LIO odometry reports (usually the IMU frame,
e.g. FAST-LIO), so ``--extrinsic-*`` is lidar-in-body (FAST-LIO's
extrinsic_T / extrinsic_R) and the IMU gyro must be expressed in it too.
Per-point time: Hesai ROS driver 2.0 publishes an absolute float64
``timestamp`` field (``--time-mode absolute``); drivers that publish offsets
from the header stamp use ``relative`` (seconds) or ``relative_ns``.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.geometry import make_T, quat_to_R, rot_zyx  # noqa: E402
from dog_perception.io import odom_row, write_sequence  # noqa: E402
from dog_perception.ros_utils import scan_from_msg, stamp_of  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="ROS2 bag directory or ROS1 .bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lidar-topic", required=True)
    ap.add_argument("--imu-topic", required=True)
    ap.add_argument("--odom-topic", required=True)
    ap.add_argument("--time-field", default="timestamp")
    ap.add_argument("--time-mode", default="absolute", choices=["absolute", "relative", "relative_ns"])
    ap.add_argument("--odom-twist", default="none", choices=["none", "body", "world"],
                    help="how to read Odometry.twist (FAST-LIO leaves it empty -> none)")
    ap.add_argument("--extrinsic-t", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--extrinsic-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="radians")
    ap.add_argument("--base-height", type=float, default=0.45)
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    scans, imu, odom = [], [], []
    topics = {args.lidar_topic, args.imu_topic, args.odom_topic}
    with AnyReader([Path(args.bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        missing = topics - {c.topic for c in conns}
        if missing:
            raise SystemExit("topics not in bag: %s (have: %s)" % (missing, sorted({c.topic for c in reader.connections})))
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == args.lidar_topic:
                if args.max_scans is not None and len(scans) >= args.max_scans:
                    continue
                scans.append(scan_from_msg(msg, args.time_field, args.time_mode))
            elif conn.topic == args.imu_topic:
                w = msg.angular_velocity
                imu.append((stamp_of(msg.header), w.x, w.y, w.z))
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                R = quat_to_R([q.x, q.y, q.z, q.w])
                v = None
                if args.odom_twist != "none":
                    l = msg.twist.twist.linear
                    v = np.array([l.x, l.y, l.z])
                    if args.odom_twist == "body":
                        v = R @ v
                odom.append(odom_row(stamp_of(msg.header), R, [p.x, p.y, p.z], v))

    if scans and scans[0].point_times is None:
        print("WARNING: no per-point time field '%s': deskew will be disabled for this data" % args.time_field)
    T_bl = make_T(rot_zyx(args.extrinsic_rpy[2], args.extrinsic_rpy[1], args.extrinsic_rpy[0]), args.extrinsic_t)
    write_sequence(args.out, scans, np.asarray(imu).reshape(-1, 4), np.asarray(odom).reshape(-1, 11),
                   dict(T_body_lidar=T_bl, base_height=args.base_height, lidar="XT32", source=str(args.bag)))
    print("wrote %d scans, %d imu, %d odom -> %s" % (len(scans), len(imu), len(odom), args.out))


if __name__ == "__main__":
    main()
