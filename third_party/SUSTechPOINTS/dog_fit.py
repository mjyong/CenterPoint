"""CenterPoint dog pipeline (2026-10-24): a box for the points the annotator selected with Ctrl + drag.

main.py's /dog_fit_box calls ``fit``; public/js/editor.js (handleSelectRect) uses the answer and falls
back to SUSTechPOINTS' own fitting when there is none. Coordinates are the scene's lidar frame, as in
the label files. numpy only.

1. The object: the selected points above the ground, and among them the group connected in the
   bird's-eye view that comes closest to the middle of the rectangle (the rectangle also takes in walls,
   bushes and cars next to the object, often more points than the object itself).
2. A model's box when one holds most of the object's points: data/<scene>/dog_dets/<frame>.json lists
   the detections of the models that ran on the frame (tools/dog/sustechpoints_convert.py detections).
   The best-scoring one is used as it is: the network knows how long a truck is when the lidar sees
   only its front, and which way a car faces.
3. Otherwise the smallest box around the object's points: yaw from an L-shape fit (the closeness
   criterion of Zhang et al., "Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners",
   2017), which follows the visible sides of a vehicle where the principal axis of an L-shaped view is
   off by up to 45 deg; bottom on the ground where the selection shows it; of the two headings along
   the box, the one closer to the view's (vehicles pointing up the screen, as for SUSTechPOINTS' own
   boxes; g flips it). A vehicle seen only in part gets a box around the part seen.
"""
import json
import os

import numpy as np

GROUND_CLEARANCE = 0.25     # [m] points this close above the ground are ground
GROUND_BAND = 0.5           # [m] selected points this close to the scene's ground level give the local ground
MAX_HEIGHT = 5.0            # [m] above the ground: higher points are not the object
CELL = 0.3                  # [m] bird's-eye grid that joins the object's points
MIN_POINTS = 5
MIN_SHARE = 0.5             # a model's box must hold this share of the object's points
BOX_MARGIN = 0.3            # [m] added around a model's box when counting the points it holds


def _meta(scene_dir):
    try:
        with open(os.path.join(scene_dir, "dog_meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def ground_level(z, z_ref=None):
    """(ground height, seen) under the selection: the median of the points within GROUND_BAND of ``z_ref``
    (the scene's ground level, from the export; without one, the lowest points) when there are at least
    20, else ``z_ref`` and seen = False (far away the ground is often hidden behind the object)."""
    if z_ref is None:
        z_ref = float(np.percentile(z, 2))
    near = z[np.abs(z - z_ref) <= GROUND_BAND]
    return (float(np.median(near)), True) if len(near) >= 20 else (float(z_ref), False)


def groups(xy, cell=CELL):
    """Group index of each point: groups of occupied grid cells that touch (8 neighbours)."""
    if len(xy) == 0:
        return np.zeros(0, dtype=np.int64)
    ij = np.floor(xy / cell).astype(np.int64)
    cells, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    index = {tuple(c): k for k, c in enumerate(cells)}
    group = -np.ones(len(cells), dtype=np.int64)
    g = 0
    for start in range(len(cells)):
        if group[start] >= 0:
            continue
        stack = [start]
        group[start] = g
        while stack:
            ci, cj = cells[stack.pop()]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    m = index.get((ci + di, cj + dj))
                    if m is not None and group[m] < 0:
                        group[m] = g
                        stack.append(m)
        g += 1
    return group[inv]


def object_mask(xy, center, min_points=MIN_POINTS):
    """Mask of the points of the group (``groups``, at least ``min_points``) with the point closest to
    ``center``, the middle of the selection."""
    g = groups(xy)
    ids, count = np.unique(g, return_counts=True)
    big = ids[count >= min_points]
    if not len(big):
        return np.zeros(len(xy), dtype=bool)
    d = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
    return g == min(big, key=lambda k: d[g == k].min())


def l_shape_yaw(xy, step_deg=1.0, d0=0.01):
    """Yaw in [0, pi/2) of the rectangle whose sides the points hug best (closeness criterion)."""
    angles = np.deg2rad(np.arange(0.0, 90.0, step_deg))
    best, best_score = 0.0, -np.inf
    for a in angles:
        c1 = xy @ [np.cos(a), np.sin(a)]
        c2 = xy @ [-np.sin(a), np.cos(a)]
        d = []
        for c in (c1, c2):
            lo, hi = c - c.min(), c.max() - c
            d.append(lo if np.linalg.norm(lo) < np.linalg.norm(hi) else hi)
        score = float(np.sum(1.0 / np.maximum(np.minimum(d[0], d[1]), d0)))
        if score > best_score:
            best, best_score = a, score
    return float(best)


def tight_box(obj, ground, yaw_hint=None):
    """Smallest box around the object's points (N, 3), L-shape yaw, bottom on the ``ground`` (None: at
    the lowest point)."""
    xy = obj[:, :2]
    a = l_shape_yaw(xy - xy.mean(0))
    e1, e2 = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c1, c2 = xy @ e1, xy @ e2
    ext = [(c1.min(), c1.max()), (c2.min(), c2.max())]
    center = e1 * sum(ext[0]) / 2 + e2 * sum(ext[1]) / 2
    size = [ext[0][1] - ext[0][0], ext[1][1] - ext[1][0]]
    yaw = a if size[0] >= size[1] else a + np.pi / 2
    length, width = max(size), min(size)
    if yaw_hint is not None and abs(np.arctan2(np.sin(yaw - yaw_hint), np.cos(yaw - yaw_hint))) > np.pi / 2:
        yaw += np.pi
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    bottom, top = float(obj[:, 2].min()), float(obj[:, 2].max())
    if ground is not None:
        bottom = min(bottom, ground)
    return {"position": {"x": float(center[0]), "y": float(center[1]), "z": (bottom + top) / 2},
            "scale": {"x": float(length), "y": float(width), "z": top - bottom},
            "rotation": {"x": 0.0, "y": 0.0, "z": yaw}}


def share_inside(obj, psr, margin=BOX_MARGIN):
    """Share of the object's points (N, 3) inside the box ``psr`` grown by ``margin``."""
    p, s, r = psr["position"], psr["scale"], psr["rotation"]
    d = obj[:, :2] - [p["x"], p["y"]]
    c, sn = np.cos(r["z"]), np.sin(r["z"])
    inside = (np.abs(c * d[:, 0] + sn * d[:, 1]) <= s["x"] / 2 + margin) & \
        (np.abs(-sn * d[:, 0] + c * d[:, 1]) <= s["y"] / 2 + margin) & \
        (np.abs(obj[:, 2] - p["z"]) <= s["z"] / 2 + margin)
    return float(inside.mean()) if len(obj) else 0.0


def model_box(obj, dets, min_share=MIN_SHARE):
    """The best-scoring detection that holds at least ``min_share`` of the object's points, or None."""
    held = [d for d in dets if share_inside(obj, d["psr"]) >= min_share]
    return max(held, key=lambda d: d["score"]) if held else None


def fit(scene_dir, frame, points, yaw_hint=None, center=None):
    """{"psr", "obj_type", "source", "points"} for the selected ``points`` ((N, 3) lidar frame), or {}
    when they hold no object (fewer than MIN_POINTS above the ground). ``center``: the middle of the
    selection rectangle (x, y); default the middle of the selected points' extent."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < MIN_POINTS:
        return {}
    if center is None:
        center = (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2
    meta = _meta(scene_dir)
    ground, seen = ground_level(pts[:, 2], meta.get("z_offset"))
    obj = pts[(pts[:, 2] > ground + GROUND_CLEARANCE) & (pts[:, 2] < ground + MAX_HEIGHT)]
    if len(obj) < MIN_POINTS:
        return {}
    obj = obj[object_mask(obj[:, :2], center)]
    if len(obj) < MIN_POINTS:
        return {}
    try:
        with open(os.path.join(scene_dir, "dog_dets", frame + ".json")) as f:
            dets = json.load(f)
    except (OSError, ValueError):
        dets = []
    d = model_box(obj, dets)
    if d is not None:
        return {"psr": d["psr"], "obj_type": d.get("obj_type"), "points": int(len(obj)),
                "source": "%s %.2f" % (d.get("model", "model"), d["score"])}
    return {"psr": tight_box(obj, ground if seen else None, yaw_hint), "obj_type": None, "points": int(len(obj)),
            "source": "fit"}
