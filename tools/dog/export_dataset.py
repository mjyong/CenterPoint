"""Labels + preprocessed frames -> DogDataset (det3d) for fine-tuning.

Frames come from ``run_sequence.py --save-frames`` (det-frame clouds, the
exact detector input) and labels from ``auto_label.py`` (optionally hand
corrected). Both are mapped into the network frame with ``ModelFrame`` so
fine-tuning sees what deployment sees. Splits are by recording.

Class names: the dog configs' head classes car / truck / bus / pedestrian /
cyclist (the labels' ``subclass``; a vehicle without one, e.g. from labels made
before subclasses existed, gets one from its size and is counted), with
``--classes coarse`` vehicle / pedestrian / cyclist for the older *_3cls configs,
or with ``--classes vehicle`` car / truck by box size for the vehicle-only
*_veh configs (``vehicle_head``; pedestrians and cyclists are left out).

Tracks still flagged for review (mean score < 0.5, not yet reviewed) are by
default written as ignore boxes (``gt_ignore``): no loss either way and not
scored, so a real but uncertain vehicle is not taught as background
(``--review-policy drop`` leaves them out, ``keep`` trains them as positives).
Tracks a reviewer marked ``ignore`` are ignore boxes too. With ``--classes
vehicle``, an unreviewed vehicle whose teacher subclass disagrees with its
size (a "truck" the size of a car) is also ignored.

Occlusion and range (``dog_perception/visibility.py``): the points inside each
box are counted in the exported frame; a box with no point in any sweep is
left out, one with fewer than 5 (vehicles) in the current sweep is an ignore box.

    python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/rec_001/labels.pkl \
        --recording work_dirs/rec_002 data/rec_002/labels.pkl --val-recordings rec_002 --out data/dog --gt-db
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # parallel

from dog_perception.detection.boxes import standard_to_det3d  # noqa: E402
from dog_perception.detection.frame_adapter import ModelFrame  # noqa: E402
from dog_perception.label_review import train_label  # noqa: E402
from dog_perception.visibility import annotate_points  # noqa: E402
from parallel import run_each  # noqa: E402


def export_recording(frames_dir, labels, out_root, name, frame=None, review_policy="ignore", classes="sub",
                     counts=None):
    frame = frame or ModelFrame()
    os.makedirs(os.path.join(out_root, "lidar", name), exist_ok=True)
    infos = []
    for f in labels["frames"]:
        src = os.path.join(frames_dir, "%.6f.npy" % f["stamp"])
        if f.get("pose_bad") or not os.path.exists(src):      # pose not trusted (refine_poses.py): left out
            continue
        cloud = np.load(src)
        objs, names, ignore = [], [], []
        # counted here, on the exact input, so boxes moved by a reviewer are counted where they are now
        for o in annotate_points(f["objects"], cloud):
            n, ign = train_label(o, classes, review_policy, counts)
            if n is not None:
                objs.append(o)
                names.append(n)
                ignore.append(ign)
        pts = frame.points_to_model(cloud)
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
                          gt_names=np.array(names, dtype="<U16"), gt_ignore=np.array(ignore, dtype=bool),
                          sequence=name, timestamp=f["stamp"]))
    return infos


def export_one(job):
    """(run dir, labels path, out root, review policy, classes) -> (name, infos, counts); for run_each."""
    run_dir, lab_path, out_root, review_policy, classes = job
    name = os.path.basename(os.path.normpath(run_dir))
    with open(lab_path, "rb") as f:
        labels = pickle.load(f)
    counts = {}
    infos = export_recording(os.path.join(run_dir, "frames"), labels, out_root, name,
                             review_policy=review_policy, classes=classes, counts=counts)
    return name, infos, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", nargs=2, action="append", metavar=("RUN_DIR", "LABELS"), required=True,
                    help="run_sequence.py output dir (with frames/) and its labels.pkl")
    ap.add_argument("--val-recordings", nargs="*", default=[])
    ap.add_argument("--test-recordings", nargs="*", default=[],
                    help="written to infos_test.pkl (never used for training or for choosing a model)")
    ap.add_argument("--review-policy", choices=("ignore", "drop", "keep"), default="ignore",
                    help="tracks still flagged for review: ignore boxes (default), left out, or positives")
    ap.add_argument("--skip-review", action="store_true", help="same as --review-policy drop")
    ap.add_argument("--classes", choices=("sub", "coarse", "vehicle"), default="sub",
                    help="sub: car/truck/bus/pedestrian/cyclist (configs/dog/*_xt32.py); "
                         "coarse: vehicle/pedestrian/cyclist (the older *_3cls configs); "
                         "vehicle: car/truck by size, vehicles only (the *_veh configs)")
    ap.add_argument("--out", default="data/dog")
    ap.add_argument("--gt-db", action="store_true", help="also build the GT-sampling database")
    ap.add_argument("--jobs", type=int, default=0, help="recordings exported at once (0: one per core)")
    args = ap.parse_args()
    policy = "drop" if args.skip_review else args.review_policy

    train, val, test, counts = [], [], [], {}
    jobs = [(run_dir, lab_path, args.out, policy, args.classes) for run_dir, lab_path in args.recording]
    for name, infos, c in run_each(export_one, jobs, args.jobs):
        (test if name in args.test_recordings else val if name in args.val_recordings else train).extend(infos)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        print("%s: %d frames" % (name, len(infos)))
    for split, infos in (("train", train), ("val", val), ("test", test)):
        if split == "test" and not args.test_recordings:
            continue
        with open(os.path.join(args.out, "infos_%s.pkl" % split), "wb") as f:
            pickle.dump(infos, f)
    print("train %d / val %d / test %d frames -> %s" % (len(train), len(val), len(test), args.out))
    for split, infos in (("train", train), ("val", val), ("test", test)):
        if not infos:
            continue
        names = np.concatenate([i["gt_names"] for i in infos])
        ign = np.concatenate([i["gt_ignore"] for i in infos])
        print("%s boxes: " % split + (", ".join("%s %d (+%d ignored)" % (n, np.sum((names == n) & ~ign),
                                                                          np.sum((names == n) & ign))
                                                for n in np.unique(names)) or "none"))
    for k in sorted(c for c in counts if "disagrees" in c or "points" in c):
        print("note: %d boxes: %s" % (counts[k], k))
    if counts.get("vehicle subclass from size"):
        print("note: %d vehicle boxes had no subclass and got one from their size (car / truck / bus); "
              "check them, or re-run auto_label.py with a teacher that has car / truck / bus heads"
              % counts["vehicle subclass from size"])

    if args.gt_db and train:
        from det3d.datasets.utils.create_gt_database import create_groundtruth_database
        create_groundtruth_database("DOG", args.out, os.path.join(args.out, "infos_train.pkl"), nsweeps=5,
                                    db_path=None, dbinfo_path=None)


if __name__ == "__main__":
    main()
