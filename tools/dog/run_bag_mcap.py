"""Stream one ROS2 bag through CenterPoint and write a Foxglove MCAP.

Localization comes from kygi665 INSPVAX (local ENU). Lidar and IMU are read
from the bag in log-time order, so nothing is written except the MCAP.

    python tools/dog/run_bag_mcap.py \
        --bag d20car_predict/rosbag0921/rosbag2_2026_09_21-16_41_27 \
        --ckpt work_dirs/pretrained/nusc_voxel.pth \
        --mcap /public/zjst/robot/rosbag0921/rosbag2_2026_09_21-16_41_27/centerpoint_voxel.mcap
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.geometry import make_T, transform_points  # noqa: E402
from dog_perception.inspvax import EnuOrigin, pose_from_inspvax, register_kygi  # noqa: E402
from dog_perception.mcap_viz import McapViz  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig  # noqa: E402
from dog_perception.ros_utils import scan_from_msg, stamp_of  # noqa: E402

VOXEL_CFG = "configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py"


def estimate_base_height(xyz):
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    band = (r > 4.0) & (r < 30.0) & np.isfinite(xyz[:, 2])
    if band.sum() < 1000:
        return 1.5
    return float(np.clip(-np.percentile(xyz[band, 2], 10), 0.4, 3.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--lidar-topic", default="/robot/topic/rdap/top_point_cloud2")
    ap.add_argument("--imu-topic", default="/rte/rdap/rtk/imus/data_raw")
    ap.add_argument("--inspvax-topic", default="/rte/rdap/rtk/inspvax")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=VOXEL_CFG)
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.detection.centerpoint import CenterPointDetector

    typestore = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    det = CenterPointDetector(args.config, args.ckpt, device=args.device)
    pipe = None
    pending = []
    viz = McapViz(args.mcap)
    origin = None
    n_scan = n_used = n_det = 0
    pos_types = {}
    net_ms = []

    with AnyReader([Path(args.bag)], default_typestore=typestore) as reader:
        print("topics:")
        for c in reader.connections:
            print("  %s  %s  %d" % (c.topic, c.msgtype, c.msgcount))
        want = {args.lidar_topic, args.imu_topic, args.inspvax_topic}
        conns = [c for c in reader.connections if c.topic in want]
        missing = want - {c.topic for c in conns}
        if missing:
            raise SystemExit("topics not in bag: %s" % sorted(missing))
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == args.imu_topic:
                w = msg.angular_velocity
                ev = ("imu", stamp_of(msg.header), (w.x, w.y, w.z))
                pending.append(ev) if pipe is None else pipe.on_imu(ev[1], ev[2])
                continue
            if conn.topic == args.inspvax_topic:
                if origin is None and int(msg.ins_status.data) == 3:
                    origin = EnuOrigin(msg.latitude, msg.longitude, msg.height)
                    print("ENU origin lat %.8f lon %.8f h %.3f" % (origin.lat, origin.lon, origin.h))
                got = pose_from_inspvax(msg, origin) if origin is not None else None
                if got is None:
                    continue
                t, R, p, v = got
                pos_types[int(msg.pos_type.data)] = pos_types.get(int(msg.pos_type.data), 0) + 1
                if pipe is None:
                    pending.append(("odom", t, R, p, v))
                else:
                    pipe.on_odometry(t, R, p, v)
                continue
            if args.max_scans is not None and n_scan >= args.max_scans:
                continue
            scan = scan_from_msg(msg, "timestamp", "absolute")
            n_scan += 1
            if pipe is None:
                h = estimate_base_height(scan.xyz)
                print("base_height %.2f m (from first cloud)" % h)
                cfg = PreprocessConfig(T_body_lidar=make_T(), num_sweeps=5, deskew=True,
                                       det_frame=DetFrameConfig(base_height=h))
                pipe = PerceptionPipeline(cfg, det)
                for ev in pending:
                    if ev[0] == "imu":
                        pipe.on_imu(ev[1], ev[2])
                    else:
                        pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
                pending.clear()
            frame = pipe.pre.process(scan)
            if frame is None:
                continue
            d = det(frame.points, frame.stamp)
            cur = frame.points[frame.points[:, -1] == 0]
            xyz_w = transform_points(frame.T_world_det, cur[:, :3]) if len(cur) else np.zeros((0, 3))
            viz.add(frame.stamp, xyz_w, cur[:, 3] if len(cur) else np.zeros(0),
                    d.transform(frame.T_world_det, "world"), frame.T_world_body)
            n_used += 1
            n_det += len(d)
            net_ms.append(det.last_timing.get("network_ms", 0.0))
            if n_used % 20 == 0:
                print("frame %d  dets %d  network %.1f ms" % (n_used, len(d), net_ms[-1]), flush=True)

    viz.close()
    med = float(np.median(net_ms)) if net_ms else float("nan")
    print("pos_type %s" % (pos_types or "-"))
    print("scans %d used %d  detections %d  network median %.1f ms" % (n_scan, n_used, n_det, med))
    print("mcap %s  %d frames" % (args.mcap, viz.frames))


if __name__ == "__main__":
    main()
