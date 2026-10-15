"""Train / val split over several recordings: static scenes, a spatial hold-out, overlap.

    python tools/dog/split_check.py --seq data/seq/0921_* --out data/survey/split \
        --holdout 0921_165659 [--holdout-s 90] [--radius 10] [--apply]

Reads each sequence's meta.json (segment name / split, ENU origin lat0 / lon0 / h0) and odom.npy
(INS positions and velocity), puts all paths into one ENU frame (the origin of the first sequence)
and decides, in this order:

1. static val: a val scene where the robot moves (> 0.3 m/s) less than ``--min-moving`` of the
   time -> drop. A lying or standing robot repeats one view ~300 times, so the metric would mostly
   measure that one view.
2. hold-out (``--holdout <recording>``): the shortest run of consecutive train scenes of that
   recording, at least ``--holdout-s`` long, whose path stays farthest from all other train paths,
   becomes val; the scene on each side of it -> drop (buffer: the same objects seconds apart).
   Ties go to the run that needs fewer buffer seconds, then to the longer path.
3. overlap: a val scene and a train scene overlap when more than ``--max-share`` of either path
   lies within ``--radius`` of the other (same street, same parked cars, minutes apart). For each
   val scene with overlaps, the cheaper side is dropped: the val scene if the overlapping train
   scenes are longer than it, otherwise those train scenes.

Output in ``--out``: split_check.csv (per scene: recording, seconds, path, moving share, split
before / after, reason, share of the path within --radius of the other side), split_check.png
(map: train blue, val orange, drop grey) and split_check.json (totals before / after, val share
within --radius of train). Nothing changes without ``--apply``; with it, ``<out>/split.csv`` (name,
split = train | val | drop) is written and each meta.json gets segment.split = train | val | drop
(the survey's value is kept once as segment.split_survey). The split.csv is what the batch
auto-label and the dataset export read (docs/BAG_SPLIT.md section 4).
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.inspvax import EnuOrigin  # noqa: E402
from dog_perception.io import SequenceReader  # noqa: E402

MOVING_SPEED = 0.3      # [m/s]


def resample(xy, step=1.0):
    """Points every ``step`` metres along a path (one point for a robot that did not move), so
    shares are shares of path length, not of time."""
    if len(xy) == 0:
        return xy
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if s[-1] < step:
        return xy.mean(axis=0, keepdims=True)
    q = np.r_[np.arange(0.0, s[-1], step), s[-1]]
    return np.column_stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])])


def near_share(xy, other, radius):
    """Share of the points ``xy`` within ``radius`` of any point of ``other``."""
    if len(xy) == 0 or len(other) == 0:
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(other).query(xy)
    return float(np.mean(d <= radius))


def recording_of(name, meta):
    m = re.match(r"(.+)_s\d+$", name)
    return m.group(1) if m else os.path.basename(os.path.normpath(meta.get("source", name)))


def load(seq_dirs):
    scenes, ref = [], None
    for d in sorted(seq_dirs):
        if not os.path.exists(os.path.join(d, "meta.json")):
            print("%s: no meta.json (unfinished conversion) - skipped" % d)
            continue
        rd = SequenceReader(d)
        m, seg = rd.meta, rd.meta.get("segment", {})
        if "lat0" not in m:
            raise SystemExit("%s: no lat0 / lon0 in meta.json (only INSPVAX sequences share a map frame)" % d)
        if ref is None:
            ref = EnuOrigin(m["lat0"], m["lon0"], m["h0"])
        off = ref.enu(m["lat0"], m["lon0"], m["h0"])[:2]    # this bag's origin in the common frame
        od = rd.odom
        t0, t1 = seg.get("t_start"), seg.get("t_end")
        if t0 is not None and t1 is not None:
            od = od[(od[:, 0] >= t0) & (od[:, 0] < t1)]
        if len(od) < 2:
            print("%s: no localization inside the scene - skipped" % d)
            continue
        xy = od[:, 1:3] + off
        if np.isfinite(od[:, 8:10]).all():
            speed = np.hypot(od[:, 8], od[:, 9])
        else:
            speed = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(od[:, 0]), 1e-3)]
        name = seg.get("name") or os.path.basename(os.path.normpath(d))
        split = seg.get("split_survey", seg.get("split")) or "train"
        scenes.append(dict(name=name, dir=d, rec=recording_of(name, m), t0=float(od[0, 0]),
                           dur=float((t1 - t0) if t0 is not None and t1 is not None else od[-1, 0] - od[0, 0]),
                           path=float(np.sum(np.hypot(*np.diff(xy, axis=0).T))),
                           moving=float(np.mean(speed > MOVING_SPEED)), before=split, after=split, reason="",
                           xy=resample(xy), near=np.nan))
    scenes.sort(key=lambda s: (s["rec"], s["t0"]))
    return scenes


def pts(scenes):
    return np.concatenate([s["xy"] for s in scenes]) if scenes else np.zeros((0, 2))


def pick_holdout(scenes, rec, target_s, radius, min_moving):
    """(block, buffers, share of the block path near the remaining train) or None."""
    run = [s for s in scenes if s["rec"] == rec]
    best = None
    for i in range(len(run)):
        for j in range(i, len(run)):
            block = run[i:j + 1]
            if any(s["after"] != "train" or s["moving"] < min_moving for s in block):
                break
            if sum(s["dur"] for s in block) < target_s:
                continue
            buffers = ([run[i - 1]] if i > 0 else []) + ([run[j + 1]] if j + 1 < len(run) else [])
            buffers = [b for b in buffers if b["after"] == "train"]
            rest = [s for s in scenes if s["after"] == "train" and s not in block and s not in buffers]
            share = near_share(pts(block), pts(rest), radius)
            key = (round(share, 3), sum(b["dur"] for b in buffers), -sum(s["path"] for s in block))
            if best is None or key < best[0]:
                best = (key, block, buffers, share)
            break                          # longer runs from i only add scenes
    return None if best is None else best[1:]


def overlaps(a, b, radius, max_share):
    return near_share(a["xy"], b["xy"], radius) > max_share or near_share(b["xy"], a["xy"], radius) > max_share


def decide(scenes, holdout=None, holdout_s=90.0, radius=10.0, max_share=0.1, min_moving=0.3):
    for s in scenes:
        if s["after"] == "val" and s["moving"] < min_moving:
            s["after"], s["reason"] = "drop", "static val scene (moving %.0f%% of the time)" % (100 * s["moving"])
    if holdout:
        got = pick_holdout(scenes, holdout, holdout_s, radius, min_moving)
        if got is None:
            print("hold-out: no run of train scenes of %s reaches %.0f s" % (holdout, holdout_s))
        else:
            block, buffers, share = got
            for s in block:
                s["after"], s["reason"] = "val", "hold-out block (%.0f%% of its path within %.0f m of train)" % (
                    100 * share, radius)
            for s in buffers:
                s["after"], s["reason"] = "drop", "buffer next to the hold-out block"
    for v in [s for s in scenes if s["after"] == "val"]:
        hits = [t for t in scenes if t["after"] == "train" and overlaps(v, t, radius, max_share)]
        if not hits:
            continue
        names = ", ".join(t["name"] for t in hits)
        if sum(t["dur"] for t in hits) > v["dur"]:
            v["after"], v["reason"] = "drop", "overlaps train %s (cheaper to drop this val scene)" % names
        else:
            for t in hits:
                t["after"], t["reason"] = "drop", "overlaps val %s" % v["name"]
    val, train = [s for s in scenes if s["after"] == "val"], [s for s in scenes if s["after"] == "train"]
    for s in scenes:
        other = train if s["after"] == "val" else val
        s["near"] = near_share(s["xy"], pts(other), radius) if s["after"] != "drop" else np.nan
    return val, train


def totals(scenes, key):
    out = {}
    for sp in ("train", "val", "drop"):
        sel = [s for s in scenes if s[key] == sp]
        out[sp] = dict(scenes=len(sel), seconds=round(sum(s["dur"] for s in sel), 1),
                       moving_seconds=round(sum(s["dur"] * s["moving"] for s in sel), 1),
                       path_m=round(sum(s["path"] for s in sel), 1))
    return out


def draw(path, scenes, radius):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted({s["rec"] for s in scenes})
    tag = {r: chr(ord("A") + i) for i, r in enumerate(recs)}
    fig, ax = plt.subplots(figsize=(12, 9))
    color = dict(train="tab:blue", val="tab:orange", drop="0.65")
    for s in scenes:
        xy = s["xy"]
        ax.plot(xy[:, 0], xy[:, 1], "-" if len(xy) > 1 else "o", c=color[s["after"]], lw=2.2 if s["after"] == "val" else 1.4)
        m = xy[len(xy) // 2]
        ax.text(m[0], m[1], "%s%s" % (tag[s["rec"]], s["name"][-3:]), fontsize=7)
    for sp, c in color.items():
        ax.plot([], [], c=c, lw=2, label=sp)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(title="  ".join("%s = %s" % (tag[r], r) for r in recs), fontsize=8, title_fontsize=8)
    t = totals(scenes, "after")
    ax.set_title("train %.0f s, val %.0f s (moving %.0f s), dropped %.0f s; overlap radius %.0f m" % (
        t["train"]["seconds"], t["val"]["seconds"], t["val"]["moving_seconds"], t["drop"]["seconds"], radius))
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def apply(scenes, out):
    with open(os.path.join(out, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "reason"])
        for s in scenes:
            w.writerow([s["name"], s["after"], s["reason"]])
    for s in scenes:
        p = os.path.join(s["dir"], "meta.json")
        meta = json.load(open(p))
        seg = meta.setdefault("segment", dict(name=s["name"]))
        seg.setdefault("split_survey", seg.get("split", ""))
        seg["split"] = s["after"]
        with open(p + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(p + ".tmp", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True, help="sequence directories of all recordings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default=None, help="recording to take an extra val block from, e.g. 0921_165659")
    ap.add_argument("--holdout-s", type=float, default=90.0, help="[s] minimum length of the hold-out block")
    ap.add_argument("--radius", type=float, default=10.0, help="[m] paths closer than this count as the same place")
    ap.add_argument("--max-share", type=float, default=0.1, help="share of a path near the other side that is allowed")
    ap.add_argument("--min-moving", type=float, default=0.3, help="val scenes moving less of the time are dropped")
    ap.add_argument("--apply", action="store_true", help="write <out>/split.csv and segment.split into meta.json")
    args = ap.parse_args()

    scenes = load(args.seq)
    if not scenes:
        raise SystemExit("no sequences")
    os.makedirs(args.out, exist_ok=True)
    before = totals(scenes, "before")
    decide(scenes, args.holdout, args.holdout_s, args.radius, args.max_share, args.min_moving)
    after = totals(scenes, "after")

    cols = ["name", "recording", "seconds", "path_m", "moving_share", "split_before", "split_after", "near_other_share",
            "reason"]
    rows = [dict(name=s["name"], recording=s["rec"], seconds=round(s["dur"], 1), path_m=round(s["path"], 1),
                 moving_share=round(s["moving"], 2), split_before=s["before"], split_after=s["after"],
                 near_other_share=None if np.isnan(s["near"]) else round(s["near"], 3), reason=s["reason"])
            for s in scenes]
    with open(os.path.join(args.out, "split_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    val, train = [s for s in scenes if s["after"] == "val"], [s for s in scenes if s["after"] == "train"]
    summary = dict(radius_m=args.radius, max_share=args.max_share, holdout=args.holdout, holdout_s=args.holdout_s,
                   before=before, after=after,
                   val_path_within_radius_of_train=round(near_share(pts(val), pts(train), args.radius), 3),
                   val_share_of_moving=round(after["val"]["moving_seconds"] /
                                             max(after["val"]["moving_seconds"] + after["train"]["moving_seconds"], 1e-9),
                                             3),
                   applied=bool(args.apply))
    with open(os.path.join(args.out, "split_check.json"), "w") as f:
        json.dump(summary, f, indent=2)
    draw(os.path.join(args.out, "split_check.png"), scenes, args.radius)

    print("%-20s %6s %6s %6s  %-5s -> %-5s %6s  %s" % ("scene", "s", "path", "moving", "was", "now", "near", "reason"))
    for r in rows:
        print("%-20s %6.1f %6.1f %6.2f  %-5s -> %-5s %6s  %s" % (
            r["name"], r["seconds"], r["path_m"], r["moving_share"], r["split_before"], r["split_after"],
            "-" if r["near_other_share"] is None else "%.2f" % r["near_other_share"], r["reason"]))
    for k in ("before", "after"):
        t = summary[k]
        print("%-6s train %4d s (%d scenes)  val %4d s, moving %4d s (%d scenes)  drop %4d s" % (
            k, t["train"]["seconds"], t["train"]["scenes"], t["val"]["seconds"], t["val"]["moving_seconds"],
            t["val"]["scenes"], t["drop"]["seconds"]))
    print("val path within %.0f m of train: %.1f%%; val = %.1f%% of the moving data" % (
        args.radius, 100 * summary["val_path_within_radius_of_train"], 100 * summary["val_share_of_moving"]))
    if args.apply:
        apply(scenes, args.out)
        print("wrote %s and segment.split in %d meta.json" % (os.path.join(args.out, "split.csv"), len(scenes)))
    else:
        print("nothing written (add --apply)")


if __name__ == "__main__":
    main()
