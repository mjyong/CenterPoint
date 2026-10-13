"""Offboard auto-labelling of a recorded sequence (teacher -> tracks -> refined labels).

Teacher: CenterPoint-Voxel with double-flip TTA (nuScenes weights first, your
fine-tuned voxel model in later rounds). Either run it here or reuse the
output of run_sequence.py (--results). Output ``labels.pkl``:

    {"sequence": str, "frames": [{"stamp", "T_world_det",
        "objects": [{"track_id", "label", "box" (7, det frame), "velocity" (2, det frame),
                     "score", "review", "interpolated"}]}]}

``labels_tracks.csv`` lists every track (low-confidence ones first, flagged
``review``) for the human pass: look at the labels in Foxglove and fill its
``action`` column, then apply it with tools/dog/review_labels.py and feed the
result to export_dataset.py.

    python tools/dog/auto_label.py --seq data/rec_001 --ckpt-voxel nusc_voxel.pth --tta --out data/rec_001/labels.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.autolabel import AutoLabelConfig, auto_label  # noqa: E402
from dog_perception.detection.boxes import CLASSES, Detections  # noqa: E402
from dog_perception.geometry import inv_T  # noqa: E402
from dog_perception.label_review import write_track_table  # noqa: E402


def teacher_frames(args):
    """Returns list of (stamp, T_world_det, Detections in det frame)."""
    if args.results:
        with open(args.results, "rb") as f:
            res = pickle.load(f)
        return [(r["stamp"], r["T_world_det"], r["detections"]) for r in res]

    from dog_perception.detection.centerpoint import CenterPointDetector
    from dog_perception.io import SequenceReader
    from dog_perception.pipeline import PerceptionPipeline
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig

    reader = SequenceReader(args.seq)
    pcfg = PreprocessConfig(T_body_lidar=reader.T_body_lidar,
                            det_frame=DetFrameConfig(base_height=reader.meta.get("base_height", 0.45),
                                                     **reader.meta.get("det_frame", {})))
    det = CenterPointDetector(args.config_voxel, args.ckpt_voxel, device=args.device, tta=args.tta,
                              score_threshold=0.1)
    pipe = PerceptionPipeline(pcfg, det)
    out = []
    for ev in reader.events():
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            frame = pipe.pre.process(ev[1])
            if frame is not None:
                out.append((frame.stamp, frame.T_world_det, det(frame.points, frame.stamp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--results", default=None, help="results.pkl from run_sequence.py (skip the teacher)")
    ap.add_argument("--config-voxel", default="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py")
    ap.add_argument("--ckpt-voxel", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-track-len", type=int, default=5)
    ap.add_argument("--min-mean-score", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = teacher_frames(args)
    world = [(s, d.transform(T, "world")) for s, T, d in frames]
    labels = auto_label(world, AutoLabelConfig(min_track_len=args.min_track_len, min_mean_score=args.min_mean_score))

    out_frames = []
    for stamp, T_wd, _ in frames:
        objs = labels.get(round(float(stamp), 6), [])
        if objs:
            d = Detections(np.stack([o["box"] for o in objs]), np.stack([o["velocity"] for o in objs]),
                           np.array([o["score"] for o in objs]), np.array([o["label"] for o in objs]))
            d = d.transform(inv_T(T_wd), "det")
        entries = [dict(track_id=o["track_id"], label=CLASSES[o["label"]], box=d.boxes[i], velocity=d.velocities[i],
                        score=o["score"], review=o["review"], interpolated=o["interpolated"], moving=o["moving"])
                   for i, o in enumerate(objs)]
        out_frames.append(dict(stamp=stamp, T_world_det=T_wd, objects=entries))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result = dict(sequence=args.seq or args.results, frames=out_frames)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    table = os.path.splitext(args.out)[0] + "_tracks.csv"
    rows = write_track_table(table, result)
    n = sum(len(f["objects"]) for f in out_frames)
    print("labelled %d frames, %d boxes, %d tracks (%d flagged for review) -> %s, %s"
          % (len(out_frames), n, len(rows), sum(r["review"] for r in rows), args.out, table))


if __name__ == "__main__":
    main()
