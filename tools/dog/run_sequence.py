"""Replay a recorded sequence through the four-stage pipeline.

Runs one or both CenterPoint variants on the *same* preprocessed frames and
saves, per detector, everything the other tools need:

    <out>/<name>/results.pkl    per frame: stamp, T_world_det, detections (det frame),
                                tracks, predictions, timings
    <out>/<name>/track_log.npz  tracker log for train_predictor.py
    <out>/frames/*.npy          (--save-frames) preprocessed clouds for auto_label / export

    python tools/dog/run_sequence.py --seq data/rec_001 --detectors pillar,voxel \
        --ckpt-pillar pp.pth --ckpt-voxel voxel.pth --out work_dirs/rec_001
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.io import SequenceReader  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.prediction.dataset import TrackLogger  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig  # noqa: E402

PRESETS = {
    "pillar": "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py",
    "voxel": "configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--detectors", default="pillar")
    ap.add_argument("--config-pillar", default=PRESETS["pillar"])
    ap.add_argument("--config-voxel", default=PRESETS["voxel"])
    ap.add_argument("--ckpt-pillar", default=None)
    ap.add_argument("--ckpt-voxel", default=None)
    ap.add_argument("--tta", action="store_true", help="double-flip TTA (offline teacher)")
    ap.add_argument("--base-height", type=float, default=None)
    ap.add_argument("--num-sweeps", type=int, default=5)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--odom-latency", type=float, default=0.0)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dog_perception.detection.centerpoint import CenterPointDetector

    reader = SequenceReader(args.seq)
    base_h = args.base_height if args.base_height is not None else reader.meta.get("base_height", 0.45)
    pcfg = PreprocessConfig(T_body_lidar=reader.T_body_lidar, num_sweeps=args.num_sweeps,
                            deskew=not args.no_deskew, det_frame=DetFrameConfig(base_height=base_h))

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
        for n in names:
            d = dets[n](frame.points, frame.stamp)
            out = run_back_half(pipes[n], frame, d)
            results[n].append(dict(stamp=frame.stamp, T_world_det=frame.T_world_det,
                                   T_world_body=frame.T_world_body, detections=d, tracks=out[0],
                                   predictions=out[1], timings=dict(frame.timings, **dets[n].last_timing)))
            logs[n].add(frame.stamp, out[0], frame.T_world_body, out[1])
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


def run_back_half(pipe, frame, dets):
    """Tracking + prediction for an already preprocessed / detected frame."""
    dets_w = dets.transform(frame.T_world_det, "world")
    tracks = pipe.tracker.step(dets_w, frame.stamp)
    preds = pipe.predictor(pipe.tracker) if pipe.predictor is not None else []
    return tracks, preds


if __name__ == "__main__":
    main()
