"""Labels + preprocessed frames -> DogDataset (det3d) for fine-tuning.

Frames come from ``run_sequence.py --save-frames`` (det-frame clouds, the
exact detector input) and labels from ``auto_label.py`` (optionally hand
corrected). Both are mapped into the network frame with ``ModelFrame`` so
fine-tuning sees what deployment sees. Splits are by recording.

Class names: the dog configs' head classes car / truck / bus / pedestrian /
cyclist (the labels' ``subclass``; a vehicle without one, e.g. from labels made
before subclasses existed, gets one from its size and is counted), or with
``--classes coarse`` vehicle / pedestrian / cyclist for the older *_3cls configs.

    python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/rec_001/labels.pkl \
        --recording work_dirs/rec_002 data/rec_002/labels.pkl --val-recordings rec_002 --out data/dog --gt-db
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import standard_to_det3d, subclass_from_size  # noqa: E402
from dog_perception.detection.frame_adapter import ModelFrame  # noqa: E402


def gt_name(o, classes="sub", counts=None):
    """Training class name of one label object (see the module docstring)."""
    if classes == "coarse" or o["label"] != "vehicle":
        return o["label"]
    if o.get("subclass"):
        return o["subclass"]
    if counts is not None:
        counts["vehicle subclass from size"] = counts.get("vehicle subclass from size", 0) + 1
    return subclass_from_size(o["box"][3], o["box"][5])


def export_recording(frames_dir, labels, out_root, name, frame=None, skip_review=False, classes="sub", counts=None):
    frame = frame or ModelFrame()
    os.makedirs(os.path.join(out_root, "lidar", name), exist_ok=True)
    infos = []
    for f in labels["frames"]:
        src = os.path.join(frames_dir, "%.6f.npy" % f["stamp"])
        if f.get("pose_bad") or not os.path.exists(src):      # pose not trusted (refine_poses.py): left out
            continue
        objs = [o for o in f["objects"] if not (skip_review and o["review"])]
        pts = frame.points_to_model(np.load(src))
        rel = os.path.join("lidar", name, "%.6f.npy" % f["stamp"])
        np.save(os.path.join(out_root, rel), pts.astype(np.float32))
        if objs:
            boxes = np.stack([o["box"] for o in objs])
            vel = np.stack([o["velocity"] for o in objs])
            boxes, vel = frame.boxes_to_model(boxes, vel)
            gt = standard_to_det3d(boxes, vel).astype(np.float32)
        else:
            gt = np.zeros((0, 9), np.float32)
        infos.append(dict(token="%s_%.6f" % (name, f["stamp"]), lidar_path=rel, gt_boxes=gt,
                          gt_names=np.array([gt_name(o, classes, counts) for o in objs]), sequence=name,
                          timestamp=f["stamp"]))
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", nargs=2, action="append", metavar=("RUN_DIR", "LABELS"), required=True,
                    help="run_sequence.py output dir (with frames/) and its labels.pkl")
    ap.add_argument("--val-recordings", nargs="*", default=[])
    ap.add_argument("--skip-review", action="store_true", help="drop tracks still flagged for review")
    ap.add_argument("--classes", choices=("sub", "coarse"), default="sub",
                    help="sub: car/truck/bus/pedestrian/cyclist (configs/dog/*_xt32.py); "
                         "coarse: vehicle/pedestrian/cyclist (the older *_3cls configs)")
    ap.add_argument("--out", default="data/dog")
    ap.add_argument("--gt-db", action="store_true", help="also build the GT-sampling database")
    args = ap.parse_args()

    train, val, counts = [], [], {}
    for run_dir, lab_path in args.recording:
        name = os.path.basename(os.path.normpath(run_dir))
        with open(lab_path, "rb") as f:
            labels = pickle.load(f)
        infos = export_recording(os.path.join(run_dir, "frames"), labels, args.out, name,
                                 skip_review=args.skip_review, classes=args.classes, counts=counts)
        (val if name in args.val_recordings else train).extend(infos)
        print("%s: %d frames" % (name, len(infos)))
    for split, infos in (("train", train), ("val", val)):
        with open(os.path.join(args.out, "infos_%s.pkl" % split), "wb") as f:
            pickle.dump(infos, f)
    print("train %d / val %d frames -> %s" % (len(train), len(val), args.out))
    names, n = np.unique(np.concatenate([i["gt_names"] for i in train + val] or [[]]), return_counts=True)
    print("boxes: " + ", ".join("%s %d" % kv for kv in zip(names, n)))
    if counts:
        print("note: %d vehicle boxes had no subclass and got one from their size (car / truck / bus); "
              "check them, or re-run auto_label.py with a teacher that has car / truck / bus heads"
              % counts["vehicle subclass from size"])

    if args.gt_db and train:
        from det3d.datasets.utils.create_gt_database import create_groundtruth_database
        create_groundtruth_database("DOG", args.out, os.path.join(args.out, "infos_train.pkl"), nsweeps=5,
                                    db_path=None, dbinfo_path=None)


if __name__ == "__main__":
    main()
