"""How much of a labelled object the detector's input actually shows.

A label can sit where the cloud shows little or nothing: the tracker coasts and the auto-labeller
interpolates through a short occlusion (a car passing in front), and the far end of a track has a
handful of returns. Teaching a single-frame detector to fire there teaches it to guess, so
training and scoring use the points inside each box (``label_review.train_label``):

* no point in the detector input (all sweeps): left out, the network sees background there;
* fewer than ``MIN_POINTS`` in the current sweep: an ignore box (the older sweeps may still show
  the object; neither object nor background);
* otherwise a positive.

The tracker still coasts through occlusions online; this only concerns the training targets.
"""
import numpy as np

MIN_POINTS = {"vehicle": 5, "pedestrian": 3, "cyclist": 3}
BOX_MARGIN = 0.25           # [m] added to length and width: box error, motion within one sweep
GROUND_CLEARANCE = 0.2      # [m] returns this close to the box bottom are ground
CURRENT_SWEEP_LAG = 0.05    # [s] column 4 of a frame (the sweep's time lag) below this: current sweep


def box_point_counts(points, boxes, margin=BOX_MARGIN, clearance=GROUND_CLEARANCE):
    """Points inside each box.

    ``points``: (N, >=3) det-frame cloud as the detector sees it (``run_sequence.py --save-frames``;
    column 4, when present, is the sweep's time lag). ``boxes``: (M, 7) standard boxes in the same
    frame. Returns (current sweep, all sweeps), each (M,) int64.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    cur = np.zeros(len(boxes), dtype=np.int64)
    total = np.zeros(len(boxes), dtype=np.int64)
    if len(boxes) == 0 or len(points) == 0:
        return cur, total
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    is_cur = points[:, 4] < CURRENT_SWEEP_LAG if points.shape[1] > 4 else np.ones(len(points), dtype=bool)
    for k, (x, y, z, l, w, h, yaw) in enumerate(boxes):
        half_l, half_w = l / 2 + margin, w / 2 + margin
        r = np.hypot(half_l, half_w)
        near = np.nonzero((np.abs(xyz[:, 0] - x) <= r) & (np.abs(xyz[:, 1] - y) <= r))[0]
        if len(near) == 0:
            continue
        dx, dy = xyz[near, 0] - x, xyz[near, 1] - y
        c, s = np.cos(yaw), np.sin(yaw)
        inside = (np.abs(c * dx + s * dy) <= half_l) & (np.abs(-s * dx + c * dy) <= half_w) & \
            (xyz[near, 2] >= z - h / 2 + clearance) & (xyz[near, 2] <= z + h / 2 + margin)
        total[k] = int(inside.sum())
        cur[k] = int((inside & is_cur[near]).sum())
    return cur, total


def annotate_points(objects, points):
    """Copies of the label ``objects`` (dicts with a det-frame ``box``) with ``points`` (current
    sweep) and ``points_all`` (all sweeps) counted in ``points``."""
    if not objects:
        return []
    cur, total = box_point_counts(points, np.stack([o["box"] for o in objects]))
    return [dict(o, points=int(c), points_all=int(t)) for o, c, t in zip(objects, cur, total)]


def visibility(o):
    """'none' (no point in any sweep), 'few' (< MIN_POINTS in the current sweep), 'ok', or None
    when the object carries no counts."""
    if o.get("points_all") is None:
        return None
    if o["points_all"] == 0:
        return "none"
    if o.get("points", 0) < MIN_POINTS.get(o.get("label"), 5):
        return "few"
    return "ok"
