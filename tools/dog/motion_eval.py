"""How well the tracker's online motion state (moving / stopped / static) follows the labels.

Re-runs the tracker on saved detections (``run_sequence.py`` results.pkl: the teacher's, or a
fine-tuned model's) for each scene and compares, frame by frame, with the auto-labels' ``moving``
flags (dog_perception/tracking/motion_eval.py). Every combination of ``--on-speed`` and
``--on-frames`` is one row, to choose ``TrackerConfig.motion_on_speed`` / ``motion_on_frames``:
few false "moving" on parked vehicles (the robot stops for nothing) against a short delay on
vehicles that start to move (the robot stops late).

    python tools/dog/motion_eval.py \
        --scene data/labels/0921_164127_s000/labels.pkl data/runs/0921_164127_s000/voxel/results.pkl \
        --scene ... --on-speed 0.5 0.7 1.0 --on-frames 2 3 5
"""
import argparse
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import CLASSES, Detections  # noqa: E402
from dog_perception.tracking import TrackerConfig  # noqa: E402
from dog_perception.tracking.motion_eval import evaluate_motion  # noqa: E402


def load_scene(labels_path, results_path, classes):
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    gt_by_stamp = {round(fr["stamp"], 6): fr for fr in labels["frames"]}
    frames = []
    for r in results:
        fr = gt_by_stamp.get(round(r["stamp"], 6))
        bad = bool(r.get("pose_bad")) or fr is None or fr.get("pose_bad")
        dets = Detections(frame="world", stamp=r["stamp"]) if bad else r["detections"].transform(r["T_world_det"], "world")
        gt = []
        if not bad:
            T = fr["T_world_det"]
            for o in fr["objects"]:
                if o["label"] in classes:
                    gt.append(dict(key=(labels_path, o["track_id"]), label=CLASSES.index(o["label"]),
                                   xy=(T[:3, :3] @ np.asarray(o["box"][:3]) + T[:3, 3])[:2],
                                   moving=bool(o.get("moving", False))))
        frames.append((r["stamp"], dets, gt))
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs=2, action="append", required=True, metavar=("LABELS", "RESULTS"))
    ap.add_argument("--classes", nargs="+", default=["vehicle"])
    ap.add_argument("--on-speed", type=float, nargs="+", default=[0.7], help="[m/s] TrackerConfig.motion_on_speed")
    ap.add_argument("--on-frames", type=int, nargs="+", default=[3], help="TrackerConfig.motion_on_frames")
    ap.add_argument("--off-speed", type=float, default=0.3)
    ap.add_argument("--off-time", type=float, default=1.0)
    ap.add_argument("--dist", type=float, default=2.0, help="[m] track <-> label match")
    args = ap.parse_args()

    scenes = [load_scene(lp, rp, args.classes) for lp, rp in args.scene]
    print("%d scenes, %d frames" % (len(scenes), sum(len(s) for s in scenes)))
    print("| on speed | on frames | moving recall | false moving | static tracks flagged | false episodes / h "
          "| moving stretches | missed | latency median / p90 s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for on in args.on_speed:
        for n in args.on_frames:
            cfg = replace(TrackerConfig(), motion_on_speed=on, motion_on_frames=n, motion_off_speed=args.off_speed,
                          motion_off_time=args.off_time)
            m = evaluate_motion(scenes, cfg, args.classes, args.dist)
            print("| %.2f | %d | %.3f (%d) | %.4f (%d) | %d / %d | %.1f | %d | %d | %.2f / %.2f |"
                  % (on, n, m["moving_recall"], m["moving_frames"], m["false_moving"], m["static_frames"],
                     m["static_flagged"], m["static_tracks"], m["false_episodes_per_h"], m["segments"],
                     m["missed"], m["latency_median"], m["latency_p90"]))


if __name__ == "__main__":
    main()
