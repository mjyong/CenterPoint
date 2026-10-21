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

Actions: ok, drop, vehicle / pedestrian / cyclist, car / truck / bus (vehicle subclass), flip, static, ignore; combine with
"+"; "from" / "to" (seconds since the recording start) limit an action to part
of a track. See dog_perception/label_review.py.
"""
import argparse
import os
import pickle
import sys

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
    a = sub.add_parser("apply", help="apply the edited track table")
    a.add_argument("--labels", required=True)
    a.add_argument("--edits", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
