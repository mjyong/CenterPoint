"""Reviewed training labels made fuller without another review: gaps of parked vehicles filled, and what
the teachers see that the review never covered kept out of the background.

The box review (SUSTechPOINTS, then ``propagate_keyframes``) decides about the auto-label's tracks. Two
things stay wrong in the result, and training learns them as background:

1. A parked vehicle the auto-label split into two tracks (the teacher lost it for a while): the frames
   between them have no box although the cloud shows the vehicle. On 0921_165659_s025, 51 of 115 such
   frames held >= 5 current-sweep points in the vehicle's box (docs/DATA_COLLECTION_PLAN.md 2.1).
   ``fill_static_gaps`` puts the box back in those frames: the place has reviewed labels before and
   after, the vehicle stands still, so the box is the reviewed one (``filled``).
2. Vehicles the first teacher missed and a second teacher finds. ``add_teacher_ignores`` adds the objects
   of fresh auto-labels (tools/dog/auto_label.py with both teachers' results) that no label of the frame
   covers, as ignore boxes (``added_by`` "teachers"): neither vehicle nor background.

Frames the reviewed labels leave out (outside the reviewed keyframes) stay out.
"""
from collections import defaultdict

import numpy as np

from .detection.boxes import Detections
from .geometry import inv_T, wrap_angle
from .visibility import box_point_counts


def _world(box, T):
    return Detections(np.asarray(box, float)[None], np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64)).transform(
        np.asarray(T, float), "world").boxes[0]


def _det(box, T):
    return Detections(np.asarray(box, float)[None], np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64)).transform(
        inv_T(np.asarray(T, float)), "det").boxes[0]


def _covers(a, b, dist=2.0, margin=0.5):
    d = np.asarray(b[:2], float) - np.asarray(a[:2], float)
    if np.hypot(*d) <= dist:
        return True
    for box, p in ((a, d), (b, -d)):
        c, s = np.cos(box[6]), np.sin(box[6])
        if abs(c * p[0] + s * p[1]) <= box[3] / 2 + margin and abs(-s * p[0] + c * p[1]) <= box[4] / 2 + margin:
            return True
    return False


def fill_static_gaps(labels, cloud_of, label="vehicle", static_spread=1.0, same_place=1.5, min_points=5):
    """Labels with the gaps of parked ``label`` objects filled where the cloud shows them, and counts.

    ``cloud_of(stamp)``: the frame's detector input (N, >=5; column 4 the sweep lag) or None. A place is
    where reviewed, non-ignored tracks stood still (90 % of their world centers within
    ``static_spread`` of the median); tracks within ``same_place`` m of each other are one place. In a
    frame between the place's first and last label with no box there, the box of the nearest labelled
    frame (in time) is put back when it holds >= ``min_points`` current-sweep points."""
    frames = labels["frames"]
    counts = defaultdict(int)
    tracks = defaultdict(list)                       # track id -> [(frame index, object)]
    for k, f in enumerate(frames):
        for o in f["objects"]:
            if o["label"] == label and not o.get("ignore") and not o.get("review"):
                tracks[o["track_id"]].append((k, o))
    static = {}
    for tid, obs in tracks.items():
        w = np.array([_world(o["box"], frames[k]["T_world_det"]) for k, o in obs])
        if np.percentile(np.hypot(*(w[:, :2] - np.median(w[:, :2], 0)).T), 90) < static_spread:
            static[tid] = (obs, w)
    places, used = [], set()
    for tid in static:
        if tid in used:
            continue
        c = np.median(static[tid][1][:, :2], 0)
        grp = [t for t in static if t not in used and np.hypot(*(np.median(static[t][1][:, :2], 0) - c)) < same_place]
        used.update(grp)
        places.append(grp)
    out = [dict(f, objects=list(f["objects"])) for f in frames]
    for grp in places:
        obs = sorted(((k, o, w) for t in grp for (k, o), w in zip(static[t][0], static[t][1])), key=lambda x: x[0])
        have = {k for k, _, _ in obs}
        centre = np.median(np.array([w[:2] for _, _, w in obs]), 0)
        for k in range(obs[0][0] + 1, obs[-1][0]):
            if k in have:
                continue
            f = out[k]
            if f.get("pose_bad"):
                continue
            here = [_world(o["box"], f["T_world_det"])[:2] for o in f["objects"] if o["label"] == label]
            if any(np.hypot(*(p - centre)) < same_place for p in here):
                continue                                 # another track (or an ignore box) is there
            near = min(obs, key=lambda x: abs(frames[x[0]]["stamp"] - f["stamp"]))
            box = _det(near[2], f["T_world_det"])
            box[6] = wrap_angle(box[6])
            cloud = cloud_of(f["stamp"])
            if cloud is None:
                counts["no cloud"] += 1
                continue
            cur, _ = box_point_counts(cloud, box[None])
            if cur[0] < min_points:
                counts["gap frames hidden (left as they are)"] += 1
                continue
            f["objects"].append(dict(near[1], box=box, velocity=np.zeros(2), interpolated=True, moving=False,
                                     filled=True))
            counts["gap frames filled"] += 1
    counts["parked places"] = len(places)
    counts["parked places split into several tracks"] = sum(len(g) > 1 for g in places)
    return dict(labels, frames=out), dict(counts)


def add_teacher_ignores(labels, extra, classes=("vehicle",), dist=2.0):
    """Labels plus, as ignore boxes, the objects of ``extra`` (auto-labels of the same scene, matched by
    stamp) of ``classes`` that no object of the frame covers, and counts."""
    counts = defaultdict(int)
    at = {round(float(f["stamp"]), 6): f for f in extra["frames"]}
    next_id = 1 + max([o["track_id"] for f in labels["frames"] for o in f["objects"]], default=0)
    ids = {}
    out = []
    for f in labels["frames"]:
        e = at.get(round(float(f["stamp"]), 6))
        objs = list(f["objects"])
        if e is not None and not f.get("pose_bad"):
            have = [np.asarray(o["box"], float) for o in objs if o["label"] in classes]
            for o in e["objects"]:
                if o["label"] not in classes:
                    continue
                if any(_covers(h, o["box"], dist) for h in have):
                    counts["already labelled"] += 1
                    continue
                if o["track_id"] not in ids:
                    ids[o["track_id"]], next_id = next_id, next_id + 1
                objs.append(dict(o, track_id=ids[o["track_id"]], ignore=True, review=True, reviewed=False,
                                 added_by="teachers"))
                counts["added as ignore boxes"] += 1
        out.append(dict(f, objects=objs))
    counts["tracks added"] = len(ids)
    return dict(labels, frames=out), dict(counts)
