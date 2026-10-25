"""Reviewed training labels made fuller (dog_perception/label_merge.py): gaps of parked vehicles filled
where the cloud shows them, and what fresh two-teacher auto-labels see that the review never covered
added as ignore boxes. For training scenes only; validation labels stay as the review left them.

    python tools/dog/merge_labels.py --reviewed data/labels/<scene>/labels_full.pkl --frames data/runs/<scene>/frames \
        --teachers data/labels_dual/<scene>/labels.pkl --out data/labels/<scene>/labels_merged.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.label_merge import add_teacher_ignores, fill_static_gaps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", required=True, help="labels after the box review (labels_full.pkl)")
    ap.add_argument("--frames", required=True, help="the scene's saved detector inputs (run_sequence.py --save-frames)")
    ap.add_argument("--teachers", default=None, help="auto-labels of the two teachers (auto_label.py with both results)")
    ap.add_argument("--min-points", type=int, default=5, help="current-sweep points a gap frame needs to be filled")
    ap.add_argument("--no-fill", action="store_true", help="do not fill the gaps of parked vehicles")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.reviewed, "rb") as f:
        labels = pickle.load(f)

    def cloud_of(stamp):
        path = os.path.join(args.frames, "%.6f.npy" % stamp)
        return np.load(path) if os.path.exists(path) else None

    report = {}
    if not args.no_fill:
        labels, report["fill"] = fill_static_gaps(labels, cloud_of, min_points=args.min_points)
    if args.teachers:
        with open(args.teachers, "rb") as f:
            extra = pickle.load(f)
        labels, report["teachers"] = add_teacher_ignores(labels, extra)
    labels["source"] = "merged: %s" % labels.get("source")
    with open(args.out, "wb") as f:
        pickle.dump(labels, f)
    print("%s -> %s %s" % (args.reviewed, args.out, report))


if __name__ == "__main__":
    main()
