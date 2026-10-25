"""The lidar's blind sector behind the robot, and whether detectors fire more often at its edges.

The XT32 on the dog sees nothing in a wedge behind it (0921: |azimuth| > 141 deg, 0 deg = straight
ahead): something on the robot's back blocks it. A structure cut off by the wedge (a wall, a row of
parked cars) ends abruptly at its edge, which a detector may take for the end of a vehicle. This tool

1. measures the wedge from the saved detector inputs (current sweep, robot body frame): the azimuth
   bins that hold (almost) no point beyond 1 m in nearly every frame;
2. counts, per model, detections >= --min-score within --max-range and how many of them no label
   explains (no label box, ignore boxes included, within 2 m), split by zone: inside the wedge, the
   --edge band of the visible side next to it, and elsewhere; near (< 10 m) and far;
3. lists the false ones in the edge band and in the wedge (scene, stamp, score, range, azimuth), to
   look at in Foxglove or SUSTechPOINTS.

    python tools/dog/sector_check.py --scenes 0921_164127_s000 0921_164127_s001 \
        --labels data/labels/{scene}/labels_sustech.pkl --frames data/runs/{scene}/frames \
        --results v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl teacher=data/runs/{scene}/voxel/results.pkl

Azimuth is in the robot body frame from results.pkl (T_world_body), 0 deg along body +y (forward on
the kygi robot), positive towards body +x.
"""
import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import CLASSES  # noqa: E402

CURRENT_SWEEP_LAG = 0.05


def body_azimuth(xy):
    """Degrees, 0 along body +y (forward), positive towards body +x."""
    return np.degrees(np.arctan2(xy[..., 0], xy[..., 1]))


def blind_sector(frames, every=10, min_range=1.0, empty_share=0.9):
    """(lo, hi) azimuth intervals in degrees with (almost) no current-sweep point beyond ``min_range``
    in at least ``empty_share`` of the sampled frames. ``frames``: [(points (N, >=3) det frame, T_body_det)]."""
    hist = []
    for pts, T_body_det in frames[::every]:
        if pts.shape[1] > 4:
            pts = pts[pts[:, 4] < CURRENT_SWEEP_LAG]
        p = pts[:, :3] @ T_body_det[:3, :3].T + T_body_det[:3, 3]
        r = np.hypot(p[:, 0], p[:, 1])
        h, _ = np.histogram(body_azimuth(p[r > min_range, :2]), bins=360, range=(-180, 180))
        hist.append(h)
    if not hist:
        return []
    hist = np.array(hist)
    floor = 0.01 * np.median(hist[hist > 0]) if (hist > 0).any() else 0
    empty = (hist <= floor).mean(0) >= empty_share
    out, start = [], None
    for i in range(361):
        on = i < 360 and empty[i]
        if on and start is None:
            start = i
        if not on and start is not None:
            out.append((start - 180.0, i - 180.0))
            start = None
    # a wedge across +-180 deg is one interval
    if len(out) > 1 and out[0][0] == -180.0 and out[-1][1] == 180.0:
        out = [(out[-1][0], out[0][1] + 360.0)] + out[1:-1]
    return out


def in_intervals(az, intervals, pad=0.0):
    a = np.asarray(az, float)
    hit = np.zeros(a.shape, bool)
    for lo, hi in intervals:
        for shift in (-360.0, 0.0, 360.0):
            hit |= (a + shift >= lo - pad) & (a + shift <= hi + pad)
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--labels", required=True, help="labels per scene, {scene} stands for the scene name")
    ap.add_argument("--frames", default=None, help="saved detector inputs per scene ({scene}); measures the wedge")
    ap.add_argument("--results", nargs="+", required=True, metavar="NAME=PATH", help="results.pkl per model ({scene})")
    ap.add_argument("--sector", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the wedge in degrees instead of measuring it, e.g. 141 219")
    ap.add_argument("--edge", type=float, default=20.0, help="[deg] width of the edge band on the visible side")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--match-dist", type=float, default=2.0)
    ap.add_argument("--classes", nargs="+", default=["vehicle"], choices=CLASSES)
    ap.add_argument("--list", type=int, default=30, help="false detections at the edge / in the wedge to list")
    args = ap.parse_args()
    want = [CLASSES.index(c) for c in args.classes]
    specs = [kv.split("=", 1) for kv in args.results]

    if args.sector:
        sector = [tuple(args.sector)]
    else:
        frames = []
        first = specs[0][1]
        for scene in args.scenes:
            rows = pickle.load(open(first.replace("{scene}", scene), "rb"))
            for row in rows[::10]:
                path = os.path.join(args.frames.replace("{scene}", scene), "%.6f.npy" % row["stamp"])
                if os.path.exists(path):
                    T = np.linalg.inv(np.asarray(row["T_world_body"])) @ np.asarray(row["T_world_det"])
                    frames.append((np.load(path), T))
        sector = blind_sector(frames, every=1)
        print("blind sector (deg, 0 = forward, from %d frames): %s" % (len(frames), sector or "none found"))
    if not sector:
        return
    edge = [(lo - args.edge, lo) for lo, _ in sector] + [(hi, hi + args.edge) for _, hi in sector]

    labels = {s: {round(f["stamp"], 6): f for f in pickle.load(open(args.labels.replace("{scene}", s), "rb"))["frames"]}
              for s in args.scenes}
    zones = ("elsewhere", "edge", "blind")
    print("\n| model | range | zone | detections | false | false share |")
    print("|---|---|---|---|---|---|")
    listing = []
    for name, path in specs:
        det, fp = Counter(), Counter()
        for scene in args.scenes:
            for row in pickle.load(open(path.replace("{scene}", scene), "rb")):
                f = labels[scene].get(round(row["stamp"], 6))
                if f is None or row.get("pose_bad"):
                    continue
                T = np.linalg.inv(np.asarray(row["T_world_body"])) @ np.asarray(row["T_world_det"])
                gt = np.array([o["box"] for o in f["objects"] if o["label"] in args.classes]).reshape(-1, 7)
                d = row["detections"]
                k = np.nonzero((d.scores >= args.min_score) & np.isin(d.labels, want))[0]
                if not len(k):
                    continue
                xy = d.boxes[k, :2] @ T[:2, :2].T + T[:2, 3]
                rng = np.hypot(xy[:, 0], xy[:, 1])
                az = body_azimuth(xy)
                zone = np.where(in_intervals(az, sector), 2, np.where(in_intervals(az, edge), 1, 0))
                for j, kk in enumerate(k):
                    if rng[j] > args.max_range:
                        continue
                    key = (zones[zone[j]], "< 10 m" if rng[j] < 10 else "10-%d m" % args.max_range)
                    det[key] += 1
                    false = not len(gt) or np.hypot(*(gt[:, :2] - d.boxes[kk, :2]).T).min() > args.match_dist
                    fp[key] += false
                    if false and zone[j]:
                        listing.append((name, scene, "%.6f" % row["stamp"], zones[zone[j]], round(float(d.scores[kk]), 2),
                                        round(float(rng[j]), 1), round(float(az[j]))))
        for rng_key in ("< 10 m", "10-%d m" % args.max_range):
            for z in zones:
                n, m = det[(z, rng_key)], fp[(z, rng_key)]
                print("| %s | %s | %s | %d | %d | %s |" % (name, rng_key, z, n, m, "%.0f%%" % (100 * m / n) if n else "-"))
    by_model = defaultdict(list)
    for row in listing:
        by_model[row[0]].append(row)
    for name, rows in by_model.items():
        print("\n%s: false detections at the edge / in the wedge (scene, stamp, zone, score, range m, azimuth deg), "
              "highest scores first" % name)
        for row in sorted(rows, key=lambda r: -r[4])[:args.list]:
            print("  ", *row[1:])


if __name__ == "__main__":
    main()
