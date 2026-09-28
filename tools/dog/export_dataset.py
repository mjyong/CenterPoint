"""Labels + preprocessed frames -> DogDataset (det3d) for fine-tuning.

Frames come from ``run_sequence.py --save-frames`` (det-frame clouds, the
exact detector input) and labels from ``auto_label.py`` (optionally hand
corrected). Both are mapped into the network frame with ``ModelFrame`` so
fine-tuning sees what deployment sees. Splits are by recording.

    python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/rec_001/labels.pkl \
        --recording work_dirs/rec_002 data/rec_002/labels.pkl --val-recordings rec_002 --out data/dog --gt-db
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import standard_to_det3d  # noqa: E402
from dog_perception.detection.frame_adapter import ModelFrame  # noqa: E402


def export_recording(frames_dir, labels, out_root, name, frame=None, skip_review=False):
    frame = frame or ModelFrame()
    os.makedirs(os.path.join(out_root, "lidar", name), exist_ok=True)
    infos = []
    for f in labels["frames"]:
        src = os.path.join(frames_dir, "%.6f.npy" % f["stamp"])
        if not os.path.exists(src):
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
                          gt_names=np.array([o["label"] for o in objs]), sequence=name, timestamp=f["stamp"]))
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", nargs=2, action="append", metavar=("RUN_DIR", "LABELS"), required=True,
                    help="run_sequence.py output dir (with frames/) and its labels.pkl")
    ap.add_argument("--val-recordings", nargs="*", default=[])
    ap.add_argument("--skip-review", action="store_true", help="drop tracks still flagged for review")
    ap.add_argument("--out", default="data/dog")
    ap.add_argument("--gt-db", action="store_true", help="also build the GT-sampling database")
    args = ap.parse_args()

    train, val = [], []
    for run_dir, lab_path in args.recording:
        name = os.path.basename(os.path.normpath(run_dir))
        with open(lab_path, "rb") as f:
            labels = pickle.load(f)
        infos = export_recording(os.path.join(run_dir, "frames"), labels, args.out, name, skip_review=args.skip_review)
        (val if name in args.val_recordings else train).extend(infos)
        print("%s: %d frames" % (name, len(infos)))
    for split, infos in (("train", train), ("val", val)):
        with open(os.path.join(args.out, "infos_%s.pkl" % split), "wb") as f:
            pickle.dump(infos, f)
    print("train %d / val %d frames -> %s" % (len(train), len(val), args.out))

    if args.gt_db and train:
        from det3d.datasets.utils.create_gt_database import create_groundtruth_database
        create_groundtruth_database("DOG", args.out, os.path.join(args.out, "infos_train.pkl"), nsweeps=5,
                                    db_path=None, dbinfo_path=None)


if __name__ == "__main__":
    main()
