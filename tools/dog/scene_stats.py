"""What each scene's labels hold, for choosing the scenes to review (docs/DATA_COLLECTION_PLAN.md 3.1, 3.4):
vehicle tracks, the ones that move inside 30 m / 15 m of the robot (by subclass), parked vehicles that drive off.

    python tools/dog/scene_stats.py --labels data/labels_dual/1104_*/labels.pkl --out data/reports/T37/scene_stats.csv

A track "moves inside R" when at least ``--min-frames`` of its frames (default 10, 1 s at 10 Hz) are moving (the
auto label's ``moving`` flag, or a speed of at least ``--speed``) with the box centre within R metres of the robot
(det frame, the robot at the origin). A "start" is a track whose first ``--min-frames`` frames do not move and
which moves inside 30 m later (a parked vehicle driving off). Interpolated boxes count; frames whose pose is not
trusted (``pose_bad``) do not. Prints one row per scene and the totals; ``--out`` writes the rows as CSV.
"""
import argparse
import csv
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

COLUMNS = ("scene", "seconds", "frames", "vehicle_tracks", "moving_30m", "moving_15m", "moving_30m_car",
           "moving_30m_truck", "moving_30m_bus", "starts", "review_tracks")


def scene_stats(labels, scene, label="vehicle", speed=1.0, min_frames=10, near=30.0, close=15.0):
    """One row (dict with COLUMNS) for a labels.pkl dict (tools/dog/auto_label.py)."""
    frames = [f for f in labels["frames"] if not f.get("pose_bad")]
    tracks = defaultdict(list)                       # track_id -> [(moving, range m, subclass, review)]
    for f in frames:
        for o in f["objects"]:
            if o["label"] != label:
                continue
            moving = bool(o.get("moving")) or float(np.hypot(*o["velocity"][:2])) >= speed
            tracks[o["track_id"]].append((moving, float(np.hypot(*o["box"][:2])), o.get("subclass"),
                                          bool(o.get("review"))))
    row = dict(scene=scene, frames=len(frames), vehicle_tracks=len(tracks), moving_30m=0, moving_15m=0,
               moving_30m_car=0, moving_30m_truck=0, moving_30m_bus=0, starts=0,
               review_tracks=sum(any(r for *_, r in t) for t in tracks.values()))
    stamps = [f["stamp"] for f in frames]
    row["seconds"] = round(float(stamps[-1] - stamps[0]), 1) if len(stamps) > 1 else 0.0
    for t in tracks.values():
        near_moving = sum(m and r <= near for m, r, _, _ in t)
        if near_moving < min_frames:
            continue
        row["moving_30m"] += 1
        row["moving_15m"] += sum(m and r <= close for m, r, _, _ in t) >= min_frames
        sub = Counter(s for _, _, s, _ in t).most_common(1)[0][0]
        if sub in ("car", "truck", "bus"):
            row["moving_30m_" + sub] += 1
        row["starts"] += len(t) >= 2 * min_frames and not any(m for m, *_ in t[:min_frames])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="labels.pkl files; the scene is the parent directory")
    ap.add_argument("--label", default="vehicle")
    ap.add_argument("--speed", type=float, default=1.0, help="[m/s] moving when the auto label's flag is not set")
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.labels):
        with open(path, "rb") as f:
            labels = pickle.load(f)
        scene = os.path.basename(os.path.dirname(os.path.abspath(path)))
        rows.append(scene_stats(labels, scene, args.label, args.speed, args.min_frames))
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")
    total = {c: (round(sum(r[c] for r in rows), 1) if c != "scene" else "total") for c in COLUMNS}
    print("| " + " | ".join(str(total[c]) for c in COLUMNS) + " |")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows + [total])


if __name__ == "__main__":
    main()
