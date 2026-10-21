"""Replay a recorded sequence through the four-stage pipeline.

Runs one or both CenterPoint variants on the *same* preprocessed frames and
saves, per detector, everything the other tools need:

    <out>/<name>/results.pkl    per frame: stamp, T_world_det, detections (det frame),
                                tracks, predictions, timings
    <out>/<name>/track_log.npz  tracker log for train_predictor.py
    <out>/frames/*.npy          (--save-frames) preprocessed clouds for auto_label / export
    <out>/<scene>_<first>.mcap  Foxglove view of the first detector (skip with --no-mcap), with the
                                sequence's camera streams when it has them (cameras.mcap)

    python tools/dog/run_sequence.py --seq data/rec_001 --detectors pillar,voxel \
        --ckpt-pillar pp.pth --ckpt-voxel voxel.pth --out work_dirs/rec_001
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS  # noqa: E402
from dog_perception.geometry import transform_points  # noqa: E402
from dog_perception.io import SequenceReader, pose_bad  # noqa: E402
from dog_perception.mcap_viz import McapViz  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.prediction.dataset import TrackLogger  # noqa: E402

PRESETS = {
    "pillar": "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py",
    "voxel": "configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--detectors", default="pillar")
    ap.add_argument("--config-pillar", default=None, help="default: from --ckpt-pillar (nuScenes or dog model)")
    ap.add_argument("--config-voxel", default=None, help="default: from --ckpt-voxel")
    ap.add_argument("--ckpt-pillar", default=None)
    ap.add_argument("--ckpt-voxel", default=None)
    ap.add_argument("--tta", action="store_true", help="double-flip TTA (offline teacher)")
    ap.add_argument("--base-height", type=float, default=None)
    ap.add_argument("--num-sweeps", type=int, default=5)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--odom-latency", type=float, default=0.0)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--mcap", default=None,
                    help="Foxglove MCAP of the first detector (default: <out>/<scene>_<first>.mcap): "
                         "cloud, reported detections, tracks, pose")
    ap.add_argument("--no-mcap", action="store_true", help="do not write the Foxglove MCAP")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dog_perception.detection import PRESET_CONFIGS
    from dog_perception.detection.centerpoint import CenterPointDetector, checkpoint_arch

    for n in ("pillar", "voxel"):
        ckpt = getattr(args, "ckpt_" + n)
        if getattr(args, "config_" + n) is None:
            arch = checkpoint_arch(ckpt) if ckpt else None
            setattr(args, "config_" + n, PRESET_CONFIGS.get(arch, PRESETS[n]))

    reader = SequenceReader(args.seq)
    base_h = args.base_height if args.base_height is not None else reader.meta.get("base_height", 0.45)
    pcfg = reader.preprocess_config(base_h, num_sweeps=args.num_sweeps, deskew=not args.no_deskew)

    names = args.detectors.split(",")
    dets = {n: CenterPointDetector(getattr(args, "config_" + n), getattr(args, "ckpt_" + n),
                                   device=args.device, tta=args.tta) for n in names}
    # one pipeline drives preprocessing; each detector gets its own tracker / predictor
    pipes = {n: PerceptionPipeline(pcfg, dets[n]) for n in names}
    main_pipe = pipes[names[0]]
    results = {n: [] for n in names}
    logs = {n: TrackLogger() for n in names}
    if args.save_frames:
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    scene = os.path.basename(os.path.normpath(args.seq))
    mcap_path = None if args.no_mcap else (
        args.mcap or os.path.join(args.out, "%s_%s.mcap" % (scene, names[0])))
    if mcap_path:
        os.makedirs(os.path.dirname(os.path.abspath(mcap_path)), exist_ok=True)
    viz = McapViz(mcap_path) if mcap_path else None

    for ev in reader.events(odom_latency=args.odom_latency):
        if ev[0] == "imu":
            main_pipe.on_imu(ev[1], ev[2])
            continue
        if ev[0] == "odom":
            main_pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
            continue
        scan = ev[1]
        frame = main_pipe.pre.process(scan)
        if frame is None:
            continue
        if args.save_frames:
            np.save(os.path.join(args.out, "frames", "%.6f.npy" % frame.stamp), frame.points)
        shown = None
        for n in names:
            d = dets[n](frame.points, frame.stamp)
            out = run_back_half(pipes[n], frame, d)
            results[n].append(dict(stamp=frame.stamp, T_world_det=frame.T_world_det,
                                   pose_bad=pose_bad(reader.meta, frame.stamp),
                                   T_world_body=frame.T_world_body, detections=d, tracks=out[0],
                                   predictions=out[1], timings=dict(frame.timings, **dets[n].last_timing)))
            logs[n].add(frame.stamp, out[0], frame.T_world_body, out[1])
            if n == names[0]:
                shown = (d.above(DEFAULT_SCORE_THRESHOLDS), out[0])
        if viz is not None and shown is not None:
            cur = frame.points[frame.points[:, -1] == 0]
            xyz_w = transform_points(frame.T_world_det, cur[:, :3]) if len(cur) else np.zeros((0, 3))
            viz.add(frame.stamp, xyz_w, cur[:, 3] if len(cur) else np.zeros(0),
                    shown[0].transform(frame.T_world_det, "world"), frame.T_world_body, shown[1])
        if len(results[names[0]]) % 20 == 0:
            print("frame %d  " % len(results[names[0]]) +
                  "  ".join("%s: %d dets %.1f ms" % (n, len(results[n][-1]["detections"]),
                                                     results[n][-1]["timings"].get("network_ms", 0)) for n in names))

    for n in names:
        d = os.path.join(args.out, n)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.pkl"), "wb") as f:
            pickle.dump(results[n], f)
        logs[n].save(os.path.join(d, "track_log.npz"))
        lat = [r["timings"].get("network_ms", np.nan) for r in results[n][1:]]
        print("[%s] %d frames, network %.1f ms (median)" % (n, len(results[n]), np.nanmedian(lat) if lat else np.nan))
    if viz is not None:
        if reader.cameras and results[names[0]]:      # the recording's cameras next to the boxes
            n_cam = viz.copy_from(reader.cameras, results[names[0]][0]["stamp"] - 1.5, results[names[0]][-1]["stamp"])
            print("mcap: %d camera messages from %s" % (n_cam, reader.cameras))
        viz.close()
        print("mcap %s  %d frames" % (mcap_path, viz.frames))


def run_back_half(pipe, frame, dets):
    """Tracking + prediction for an already preprocessed / detected frame."""
    dets_w = dets.transform(frame.T_world_det, "world")
    tracks = pipe.tracker.step(dets_w, frame.stamp)
    preds = pipe.predictor(pipe.tracker) if pipe.predictor is not None else []
    return tracks, preds


if __name__ == "__main__":
    main()
