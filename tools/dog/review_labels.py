"""Track-level review of auto-labels: look in Foxglove, fix whole tracks in a CSV, apply.

    # 1) labels + point clouds (+ the teacher's raw boxes) -> Foxglove
    python tools/dog/review_labels.py mcap --labels data/labels/rec_001/labels.pkl \
        --frames work_dirs/rec_001/frames --results work_dirs/rec_001/voxel/results.pkl \
        --out data/labels/rec_001/rec_001_review.mcap

    # 2) edit the "action" column of data/labels/rec_001/labels_tracks.csv
    #    (written by auto_label.py; regenerate with the "table" command; vehicles only:
    #    table --classes vehicle --out data/labels/rec_001/labels_tracks_vehicle.csv)

    # 3) apply -> labels_reviewed.pkl, then export_dataset.py
    python tools/dog/review_labels.py apply --labels data/labels/rec_001/labels.pkl \
        --edits data/labels/rec_001/labels_tracks.csv --out data/labels/rec_001/labels_reviewed.pkl

    # how many tracks each review rule would flag (labels from auto_label.py with det_score)
    python tools/dog/review_labels.py confidence --vehicles-only --labels data/labels/*/labels.pkl \
        --spot-check data/reports/review_spot_check.csv

Actions: ok, drop, vehicle / pedestrian / cyclist, car / truck / bus (vehicle subclass), flip, static, ignore; combine with
"+"; "from" / "to" (seconds since the recording start) limit an action to part
of a track. See dog_perception/label_review.py.
"""
import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.label_review import apply_edits, read_edits, write_review_mcap, write_track_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="write the track table CSV")
    t.add_argument("--labels", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--classes", choices=("sub", "coarse", "vehicle"), default="sub",
                   help="training classes for the train_as column (as export_dataset.py); vehicle: vehicles only")
    m = sub.add_parser("mcap", help="labels (+ point clouds, teacher boxes) -> Foxglove MCAP")
    m.add_argument("--labels", required=True)
    m.add_argument("--frames", default=None, help="run_sequence.py --save-frames output (<run>/frames)")
    m.add_argument("--results", default=None, help="run_sequence.py results.pkl: teacher boxes in grey")
    m.add_argument("--all-sweeps", action="store_true", help="draw the accumulated sweeps (larger file)")
    m.add_argument("--vehicles-only", action="store_true", help="draw only the vehicle labels")
    m.add_argument("--cameras", default=None,
                   help="camera MCAP to copy in (the sequence's cameras.mcap, e.g. data/seq/<scene>/cameras.mcap)")
    m.add_argument("--out", default=None,
                   help="default: <labels dir>/<scene>_review.mcap, scene = the directory name")
    c = sub.add_parser("confidence", help="tracks and boxes each review rule would flag, over many scenes")
    c.add_argument("--labels", nargs="+", required=True, help="labels.pkl files (auto_label.py, with det_score)")
    c.add_argument("--vehicles-only", action="store_true")
    c.add_argument("--review-score", type=float, default=0.5, help="mean score at or above which a track is sure")
    c.add_argument("--scores", type=float, nargs="+", default=[0.5, 0.6, 0.7], help="confident detection scores")
    c.add_argument("--frames", type=int, nargs="+", default=[3, 5, 10], help="confident detections needed")
    c.add_argument("--spot-check", default=None,
                   help="CSV of the tracks the default rule (0.6, 5) no longer flags, for a look in Foxglove")
    a = sub.add_parser("apply", help="apply the edited track table")
    a.add_argument("--labels", required=True)
    a.add_argument("--edits", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "confidence":
        confidence(args)
        return
    with open(args.labels, "rb") as f:
        labels = pickle.load(f)
    if args.cmd == "table":
        rows = write_track_table(args.out, labels, args.classes)
        print("%d tracks (%d flagged for review, %d to decide: train_as = ignore) -> %s"
              % (len(rows), sum(r["review"] for r in rows), sum(r["train_as"] == "ignore" for r in rows), args.out))
    elif args.cmd == "mcap":
        results = None
        if args.results:
            with open(args.results, "rb") as f:
                results = pickle.load(f)
        out = args.out
        if not out:
            d = os.path.dirname(os.path.abspath(args.labels))
            out = os.path.join(d, os.path.basename(d) + "_review.mcap")
        if args.vehicles_only:
            labels = dict(labels, frames=[dict(f, objects=[o for o in f["objects"] if o["label"] == "vehicle"])
                                          for f in labels["frames"]])
        n = write_review_mcap(out, labels, args.frames, results, current_sweep_only=not args.all_sweeps,
                              cameras=args.cameras)
        print("%d messages -> %s (3D panel: display frame world or det; topics /lidar /labels /teacher, "
              "Log panel on /log lists the tracks to review)" % (n, out))
    else:
        out, counts = apply_edits(labels, read_edits(args.edits))
        with open(args.out, "wb") as f:
            pickle.dump(out, f)
        left = len({o["track_id"] for fr in out["frames"] for o in fr["objects"] if o["review"]})
        print("applied %s; %d tracks still flagged (ignore boxes in export_dataset.py) -> %s"
              % (counts, left, args.out))


def confidence(args):
    """Prints, for each (confident score, confident detections) pair, the tracks and boxes still flagged
    (mean score below --review-score and fewer confident detections), next to the mean-only rule."""
    tracks = []
    for path in args.labels:
        with open(path, "rb") as f:
            labels = pickle.load(f)
        scene = os.path.basename(os.path.dirname(os.path.abspath(path)))
        by_id = defaultdict(list)
        for fr in labels["frames"]:
            for o in fr["objects"]:
                if not args.vehicles_only or o["label"] == "vehicle":
                    by_id[o["track_id"]].append(o)
        for tid, objs in by_id.items():
            if "det_score" not in objs[0]:
                raise SystemExit("%s has no det_score: re-run auto_label.py" % path)
            det = np.array([o["det_score"] for o in objs if not o["interpolated"]])
            tracks.append(dict(scene=scene, track_id=tid, mean_score=float(objs[0]["score"]), det=det,
                               boxes=len(objs), length_m=float(np.median([o["box"][3] for o in objs])),
                               min_range_m=float(min(np.hypot(*o["box"][:2]) for o in objs))))
    low = [t for t in tracks if t["mean_score"] < args.review_score]
    print("%d tracks, %d boxes; mean score < %.2f: %d tracks, %d boxes"
          % (len(tracks), sum(t["boxes"] for t in tracks), args.review_score, len(low), sum(t["boxes"] for t in low)))
    print("still flagged (tracks / boxes) when a track with N detections >= S is sure:")
    print("| S \\ N | " + " | ".join(str(n) for n in args.frames) + " |")
    print("|---|" + "---|" * len(args.frames))
    for sc in args.scores:
        cells = []
        for n in args.frames:
            left = [t for t in low if np.sum(t["det"] >= sc) < n]
            cells.append("%d / %d" % (len(left), sum(t["boxes"] for t in left)))
        print("| %.2f | %s |" % (sc, " | ".join(cells)))
    if args.spot_check:
        freed = [t for t in low if np.sum(t["det"] >= 0.6) >= 5]
        os.makedirs(os.path.dirname(os.path.abspath(args.spot_check)), exist_ok=True)
        with open(args.spot_check, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "track_id", "mean_score", "confident_frames", "max_score", "boxes", "length_m",
                        "min_range_m", "real_vehicle"])
            for t in sorted(freed, key=lambda t: (t["scene"], t["track_id"])):
                w.writerow([t["scene"], t["track_id"], "%.3f" % t["mean_score"], int(np.sum(t["det"] >= 0.6)),
                            "%.3f" % (t["det"].max() if len(t["det"]) else 0), t["boxes"], "%.1f" % t["length_m"],
                            "%.1f" % t["min_range_m"], ""])
        print("%d tracks no longer flagged with (0.6, 5) -> %s (fill real_vehicle: y / n)" % (len(freed), args.spot_check))


if __name__ == "__main__":
    main()
