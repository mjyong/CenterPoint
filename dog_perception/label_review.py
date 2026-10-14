"""Track-level review of auto-labels (``labels.pkl`` from tools/dog/auto_label.py).

A reviewer watches the labels in Foxglove (``write_review_mcap``) and edits
one row per track in the track table (``write_track_table``): the
``action`` column takes

    ok                          accept the track as is (clears the review flag)
    drop                        remove it
    vehicle|pedestrian|cyclist  change the class
    car|truck|bus               a vehicle of that subclass (the dog models' vehicle heads; the
                                ``subclass`` column shows the teacher's vote)
    flip                        turn the heading by 180 deg
    static                      the object does not move: zero velocity, one position, one heading

Several actions combine with ``+`` (``cyclist+flip``). ``from`` / ``to``
limit the action to part of the track (seconds since the recording start,
as in the table's ``t_first`` / ``t_last``), e.g. to drop the frames after
a track jumped to another object; copy the row to give the rest of the
track its own action (rows are applied top to bottom). Any action other than drop clears the
review flag; tracks still flagged after review are dropped by
``export_dataset.py --skip-review``. Missing objects cannot be added here;
use a 3D annotation tool for that (see docs/DOG_PIPELINE.md).
"""
import copy
import csv
from collections import defaultdict

import numpy as np

from .detection.boxes import CLASSES, SUBCLASS_TO_CLASS, Detections
from .geometry import inv_T, wrap_angle

VEHICLE_SUBCLASSES = tuple(k for k, v in SUBCLASS_TO_CLASS.items() if v == "vehicle")
ACTIONS = ("ok", "drop", "flip", "static") + CLASSES + VEHICLE_SUBCLASSES
TABLE_COLUMNS = ["track_id", "label", "subclass", "review", "mean_score", "frames", "interpolated", "t_first",
                 "t_last", "min_range_m", "moving_share", "max_speed", "action", "from", "to", "note"]


def _tracks(labels):
    """track_id -> list of (frame index, object index)."""
    out = defaultdict(list)
    for fi, f in enumerate(labels["frames"]):
        for oi, o in enumerate(f["objects"]):
            out[o["track_id"]].append((fi, oi))
    return out


def track_table(labels):
    """One row per track, flagged tracks first."""
    t0 = labels["frames"][0]["stamp"] if labels["frames"] else 0.0
    rows = []
    for tid, refs in _tracks(labels).items():
        objs = [labels["frames"][fi]["objects"][oi] for fi, oi in refs]
        stamps = [labels["frames"][fi]["stamp"] for fi, _ in refs]
        speed = np.array([np.hypot(*o["velocity"]) for o in objs])
        rows.append(dict(
            track_id=tid, label=objs[0]["label"], subclass=objs[0].get("subclass") or "",
            review=int(any(o["review"] for o in objs)),
            mean_score="%.3f" % np.mean([o["score"] for o in objs]), frames=len(objs),
            interpolated=sum(bool(o["interpolated"]) for o in objs),
            t_first="%.2f" % (min(stamps) - t0), t_last="%.2f" % (max(stamps) - t0),
            min_range_m="%.1f" % min(np.hypot(*o["box"][:2]) for o in objs),
            moving_share="%.2f" % np.mean([bool(o.get("moving", False)) for o in objs]),
            max_speed="%.2f" % speed.max(), action="", **{"from": "", "to": ""}, note=""))
    rows.sort(key=lambda r: (-r["review"], float(r["mean_score"]), r["track_id"]))
    return rows


def write_track_table(path, labels):
    rows = track_table(labels)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def read_edits(path):
    """{track_id: [(actions list, t_from or None, t_to or None)]} for rows with an action."""
    edits = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            acts = [a.strip().lower() for a in (row.get("action") or "").replace(";", "+").split("+") if a.strip()]
            if not acts:
                continue
            bad = [a for a in acts if a not in ACTIONS]
            if bad:
                raise ValueError("track %s: unknown action %s (use %s)" % (row["track_id"], bad, ", ".join(ACTIONS)))
            num = lambda v: float(v) if v not in (None, "") else None   # noqa: E731
            edits[int(row["track_id"])].append((acts, num(row.get("from")), num(row.get("to"))))
    return dict(edits)


def _world(f, objs):
    d = Detections(np.stack([o["box"] for o in objs]), np.stack([o["velocity"] for o in objs]),
                   np.ones(len(objs)), np.zeros(len(objs), np.int64))
    return d.transform(f["T_world_det"], "world")


def apply_edits(labels, edits):
    """Returns (new labels, counts). ``labels`` is not modified."""
    out = copy.deepcopy(labels)
    t0 = out["frames"][0]["stamp"] if out["frames"] else 0.0
    counts = defaultdict(int)
    tracks = _tracks(out)
    drop = set()
    for tid, acts, t_from, t_to in ((tid, *e) for tid, rows in edits.items() for e in rows):
        refs = [(fi, oi) for fi, oi in tracks.get(tid, []) if (fi, oi) not in drop
                and (t_from is None or out["frames"][fi]["stamp"] - t0 >= t_from - 1e-6)
                and (t_to is None or out["frames"][fi]["stamp"] - t0 <= t_to + 1e-6)]
        if not refs:
            counts["no frames"] += 1
            continue
        if "drop" in acts:
            drop.update(refs)
            counts["drop"] += 1
            continue
        if "static" in acts:
            _make_static(out, refs)
        for fi, oi in refs:
            o = out["frames"][fi]["objects"][oi]
            for a in acts:
                if a in CLASSES:
                    if a != o["label"]:     # a vehicle keeps its subclass when only confirmed
                        o["subclass"] = None if a == "vehicle" else a
                    o["label"] = a
                elif a in VEHICLE_SUBCLASSES:
                    o["label"], o["subclass"] = "vehicle", a
                elif a == "flip":
                    o["box"][6] = wrap_angle(o["box"][6] + np.pi)
            o["review"] = False
            o["reviewed"] = True
        for a in acts:
            counts[a] += 1
    for fi, f in enumerate(out["frames"]):
        f["objects"] = [o for oi, o in enumerate(f["objects"]) if (fi, oi) not in drop]
    return out, dict(counts)


def _make_static(out, refs):
    """One world position (median) and heading (circular mean) for the given frames, zero velocity."""
    world = [_world(out["frames"][fi], [out["frames"][fi]["objects"][oi]]) for fi, oi in refs]
    xy = np.median(np.stack([w.boxes[0, :2] for w in world]), 0)
    yaws = np.array([w.boxes[0, 6] for w in world])
    ref = yaws[0]
    yaws = np.where(np.abs(wrap_angle(yaws - ref)) > np.pi / 2, yaws + np.pi, yaws)
    yaw = float(np.arctan2(np.sin(yaws).sum(), np.cos(yaws).sum()))
    for (fi, oi), w in zip(refs, world):
        f = out["frames"][fi]
        b = w.boxes[0].copy()
        b[:2], b[6] = xy, yaw
        d = Detections(b[None], np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64), "world")
        d = d.transform(inv_T(f["T_world_det"]), "det")
        o = f["objects"][oi]
        o["box"], o["velocity"], o["moving"] = d.boxes[0], np.zeros(2), False


# ------------------------------------------------------------------ Foxglove
def write_review_mcap(path, labels, frames_dir=None, results=None, current_sweep_only=True, trail=3.0):
    """Point cloud (``frames_dir/<stamp>.npy``, det frame), label boxes and track trails.

    Frames: ``world`` (localization) -> ``det`` (robot, gravity aligned). Labels and the
    point cloud are in ``det``; trails in ``world``, so static objects should stay put
    when the 3D panel's display frame is ``world``. Flagged tracks are drawn magenta
    with a thick outline, interpolated boxes faint. ``results`` (run_sequence.py
    results.pkl) adds the teacher's raw boxes in grey."""
    import os

    from . import foxglove as fx
    from .clip_viz import CLASS_RGB
    from .detection.filters import DEFAULT_SCORE_THRESHOLDS
    from .detection.boxes import score_threshold_array

    flagged_rgb = fx.hex_rgba("#e11d8f")
    teacher = {}
    if results is not None:
        teacher = {round(float(r["stamp"]), 6): r["detections"] for r in results}
    t0 = labels["frames"][0]["stamp"] if labels["frames"] else 0.0
    history = defaultdict(list)
    announced = set()
    w = fx.McapWriter(path)
    try:
        for f in labels["frames"]:
            t = f["stamp"]
            w.write("/tf", "foxglove.FrameTransforms", t, fx.frame_transforms(t, [("world", "det", f["T_world_det"])]))
            if frames_dir is not None:
                p = os.path.join(frames_dir, "%.6f.npy" % t)
                if os.path.exists(p):
                    pts = np.load(p)
                    if current_sweep_only and pts.shape[1] > 4:
                        pts = pts[pts[:, 4] <= 1e-6]
                    w.write("/lidar", "foxglove.PointCloud", t, fx.point_cloud(t, "det", pts[:, :3],
                                                                               intensity=pts[:, 3]))
            ents = []
            world = _world(f, f["objects"]) if f["objects"] else None
            for k, o in enumerate(f["objects"]):
                c = CLASSES.index(o["label"])
                base = flagged_rgb if o["review"] else CLASS_RGB.get(c, (0.6, 0.6, 0.6, 1))
                alpha = 0.12 if o["interpolated"] else 0.3
                b = o["box"]
                name = o["label"] + ("/" + o["subclass"] if o.get("subclass") not in (None, o["label"]) else "")
                tag = "T%d %s %.2f%s%s" % (o["track_id"], name, o["score"], " REVIEW" if o["review"] else "",
                                         " interp" if o["interpolated"] else "")
                v = np.hypot(*o["velocity"])
                if v > 0.05:
                    tag += " %.1f m/s" % v
                ents.append(fx.entity(
                    "label/%d" % o["track_id"], t, "det", lifetime=0.3,
                    metadata=dict(track_id=o["track_id"], label=o["label"], subclass=o.get("subclass") or "",
                                  score="%.3f" % o["score"],
                                  review=o["review"], t="%.2f" % (t - t0)),
                    cubes=[fx.cube(b[:3], b[3:6], b[6], base[:3] + (alpha,))],
                    lines=[fx.box_wire(b, base[:3] + (1.0,), 0.12 if o["review"] else 0.05)],
                    texts=[fx.text(b[:3] + np.array([0, 0, b[5] / 2 + 0.4]), tag, base[:3] + (1.0,), 11)],
                    arrows=[fx.arrow(b[:3] + np.array([0, 0, b[5] / 2]), np.r_[o["velocity"], 0.0],
                                     base[:3] + (1.0,), 0.08)]))
                hist = history[o["track_id"]]
                hist.append((t, world.boxes[k, :3].copy()))
                while hist and hist[0][0] < t - trail:
                    hist.pop(0)
                if len(hist) > 1:
                    ents.append(fx.entity("trail/%d" % o["track_id"], t, "world", lifetime=0.3,
                                          lines=[fx.line(np.stack([h[1] for h in hist]), base[:3] + (0.8,), 0.06)]))
                if o["review"] and o["track_id"] not in announced:
                    announced.add(o["track_id"])
                    w.write("/log", "foxglove.Log", t, fx.log(t, "t=%.2f s: track %d (%s, score %.2f) needs review" % (
                        t - t0, o["track_id"], o["label"], o["score"]), level=3, name="review"))
            w.write("/labels", "foxglove.SceneUpdate", t, fx.scene_update(ents))
            d = teacher.get(round(float(t), 6))
            if d is not None and len(d):
                keep = d.scores >= score_threshold_array(d.labels, DEFAULT_SCORE_THRESHOLDS)
                tents = [fx.entity("teacher/%d" % i, t, "det", lifetime=0.3,
                                   lines=[fx.box_wire(d.boxes[i], (0.6, 0.6, 0.6, 0.9), 0.03)],
                                   texts=[fx.text(d.boxes[i, :3] - np.array([0, 0, d.boxes[i, 5] / 2 + 0.3]),
                                                  "%.2f" % d.scores[i], (0.7, 0.7, 0.7, 1), 9)])
                         for i in np.nonzero(keep)[0]]
                w.write("/teacher", "foxglove.SceneUpdate", t, fx.scene_update(tents))
    finally:
        w.close()
    return w.messages
