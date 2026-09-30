"""Stream one ROS2 bag through the pipeline and write a Foxglove MCAP.

Localization comes from kygi665 INSPVAX (local ENU). Lidar, IMU and INS are
read from the bag in log-time order; only the MCAP is written:

    /lidar        current sweep (map frame)
    /detections   detections above the per-class report thresholds
    /tracks       confirmed tracks with id / class / speed (coasting ones faded)
    /pose         robot pose

    python tools/dog/run_bag_mcap.py \
        --bag samples/rosbag2_2026_09_21-16_41_27 \
        --ckpt work_dirs/pretrained/nusc_pp.pth \
        --mcap work_dirs/rosbag2_2026_09_21-16_41_27.mcap

The det3d config is picked from the checkpoint (pillar / voxel) unless
``--config`` is given; a checkpoint that does not fit the config is refused.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS, PRESET_CONFIGS  # noqa: E402
from dog_perception.geometry import make_T, rot_zyx, transform_points  # noqa: E402
from dog_perception.inspvax import EnuOrigin, pose_from_inspvax, register_kygi  # noqa: E402
from dog_perception.mcap_viz import McapViz  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig, estimate_base_height  # noqa: E402
from dog_perception.ros_utils import scan_from_msg, stamp_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--lidar-topic", default="/robot/topic/rdap/top_point_cloud2")
    ap.add_argument("--imu-topic", default="/rte/rdap/rtk/imus/data_raw")
    ap.add_argument("--inspvax-topic", default="/rte/rdap/rtk/inspvax")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="det3d config or preset (%s); default: from --ckpt"
                    % ", ".join(PRESET_CONFIGS))
    ap.add_argument("--extrinsic-t", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="lidar position in the INS/body frame [m]")
    ap.add_argument("--extrinsic-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="radians")
    ap.add_argument("--base-height", type=float, default=None,
                    help="body origin height above ground; default: estimated from the first sweep")
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.detection import build_detector
    from dog_perception.detection.centerpoint import checkpoint_arch

    config = args.config or checkpoint_arch(args.ckpt)
    if config is None:
        raise SystemExit("cannot tell the architecture of %s; pass --config" % args.ckpt)
    print("config: %s" % PRESET_CONFIGS.get(config, config))
    det = build_detector(config, checkpoint=args.ckpt, device=args.device)
    T_bl = make_T(rot_zyx(args.extrinsic_rpy[2], args.extrinsic_rpy[1], args.extrinsic_rpy[0]), args.extrinsic_t)

    typestore = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    pipe, pending = None, []
    viz = McapViz(args.mcap)
    origin = None
    stats = dict(scans=0, frames=0, raw=0, filtered=0, reported=0, tracks=0)
    pos_types, net_ms = {}, []

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
                pos_types[int(msg.pos_type.data)] = pos_types.get(int(msg.pos_type.data), 0) + 1
                pending.append(("odom",) + tuple(got)) if pipe is None else pipe.on_odometry(*got)
                continue
            if args.max_scans is not None and stats["scans"] >= args.max_scans:
                continue
            scan = scan_from_msg(msg, "timestamp", "absolute")
            stats["scans"] += 1
            if pipe is None:
                h = args.base_height
                if h is None:
                    h = estimate_base_height(transform_points(T_bl, scan.xyz[np.isfinite(scan.xyz).all(1)])) or 0.45
                print("base_height %.2f m%s" % (h, "" if args.base_height else " (estimated from the first sweep)"))
                cfg = PreprocessConfig(T_body_lidar=T_bl, num_sweeps=5,
                                       det_frame=DetFrameConfig(base_height=h, auto_ground=True))
                pipe = PerceptionPipeline(cfg, det)
                for ev in pending:
                    (pipe.on_imu if ev[0] == "imu" else pipe.on_odometry)(*ev[1:])
                pending.clear()
            out = pipe.on_scan(scan)
            if out is None:
                continue
            frame = out.frame
            shown = out.detections.above(DEFAULT_SCORE_THRESHOLDS)
            cur = frame.points[frame.points[:, -1] == 0]
            xyz_w = transform_points(frame.T_world_det, cur[:, :3]) if len(cur) else np.zeros((0, 3))
            viz.add(frame.stamp, xyz_w, cur[:, 3] if len(cur) else np.zeros(0),
                    shown.transform(frame.T_world_det, "world"), frame.T_world_body, out.tracks)
            confirmed = [t for t in out.tracks if not t.coasting]
            stats["frames"] += 1
            stats["raw"] += det.last_raw_count
            stats["filtered"] += len(out.detections)
            stats["reported"] += len(shown)
            stats["tracks"] += len(confirmed)
            net_ms.append(det.last_timing.get("network_ms", 0.0))
            if stats["frames"] % 10 == 0:
                print("frame %d  raw %d  filtered %d  reported %d  tracks %d  network %.1f ms"
                      % (stats["frames"], det.last_raw_count, len(out.detections), len(shown), len(confirmed),
                         net_ms[-1]), flush=True)

    viz.close()
    n = max(stats["frames"], 1)
    print("pos_type %s" % (pos_types or "-"))
    print("scans %d, frames %d | per frame: raw %.1f -> filtered %.1f -> reported %.1f, confirmed tracks %.1f"
          % (stats["scans"], stats["frames"], stats["raw"] / n, stats["filtered"] / n, stats["reported"] / n,
             stats["tracks"] / n))
    print("network median %.1f ms" % (float(np.median(net_ms)) if net_ms else float("nan")))
    print("mcap %s  %d frames" % (args.mcap, viz.frames))


if __name__ == "__main__":
    main()
