"""Post-detection sanity filters and per-class reporting thresholds.

The detector keeps everything down to score 0.1 because the tracker's second
association stage uses low-score boxes to *continue* existing tracks. Those
boxes must never be reported as objects. This module holds

* geometric filters that are valid at every score (applied inside the
  detector): too few lidar points in the box, box floating above or sunk
  below the ground, and duplicates of another class on the same object
  (car + bicycle heads firing on one car);
* the per-class score thresholds used to start tracks and to report / draw
  detections (``Detections.above``).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .boxes import CLASSES

# score needed to report a detection or to start a track. Cyclist is the
# noisiest nuScenes class on vegetation / poles for a low-mounted lidar.
DEFAULT_SCORE_THRESHOLDS = {"vehicle": 0.35, "pedestrian": 0.3, "cyclist": 0.4}


@dataclass
class DetectionFilterConfig:
    # minimum points of the *current* sweep inside the (slightly enlarged) box;
    # boxes hallucinated from context in empty space have ~0
    min_points: dict = field(default_factory=lambda: {"vehicle": 5, "pedestrian": 3, "cyclist": 3})
    box_margin: float = 0.1          # relative enlargement of l / w / h for the point count
    # box bottom relative to the det-frame ground (z = 0)
    max_float: float = 0.7           # [m] above ground: boxes on vegetation / walls / awnings
    max_sink: float = 1.2            # [m] below ground
    cross_class_nms: bool = True
    cross_class_margin: float = 0.2  # [m] enlargement of the higher-score box


def points_in_box_counts(xyz, boxes, margin=0.1):
    """Number of points of (N, 3) ``xyz`` inside each (M, 7) standard box."""
    counts = np.zeros(len(boxes), dtype=np.int64)
    if len(xyz) == 0 or len(boxes) == 0:
        return counts
    tree = cKDTree(xyz[:, :2])
    for i, (cx, cy, cz, l, w, h, yaw) in enumerate(boxes):
        l, w, h = l * (1 + margin), w * (1 + margin), h * (1 + margin)
        idx = tree.query_ball_point([cx, cy], 0.5 * np.hypot(l, w))
        if not idx:
            continue
        p = xyz[idx]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        counts[i] = int(np.sum((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2) & (np.abs(p[:, 2] - cz) <= h / 2)))
    return counts


def cross_class_suppress(dets, margin=0.2):
    """Drop a box whose center lies inside a higher-scoring box of another class."""
    order = np.argsort(-dets.scores, kind="stable")
    keep = np.ones(len(dets), dtype=bool)
    b = dets.boxes
    for a_i, i in enumerate(order):
        if not keep[i]:
            continue
        c, s = np.cos(b[i, 6]), np.sin(b[i, 6])
        for j in order[a_i + 1:]:
            if not keep[j] or dets.labels[j] == dets.labels[i]:
                continue
            dx, dy = b[j, 0] - b[i, 0], b[j, 1] - b[i, 1]
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            if abs(lx) <= b[i, 3] / 2 + margin and abs(ly) <= b[i, 4] / 2 + margin:
                keep[j] = False
    return keep


def filter_detections(dets, points_det, cfg=None):
    """Apply the geometric filters. ``points_det``: (N, >=4) det-frame cloud whose
    last column is the sweep time lag (0 = current sweep)."""
    cfg = cfg or DetectionFilterConfig()
    if len(dets) == 0:
        return dets
    keep = np.ones(len(dets), dtype=bool)

    bottom = dets.boxes[:, 2] - dets.boxes[:, 5] / 2
    keep &= (bottom <= cfg.max_float) & (bottom >= -cfg.max_sink)

    if cfg.min_points and points_det is not None:
        cur = points_det[points_det[:, -1] <= 1e-6, :3]
        need = np.array([cfg.min_points.get(CLASSES[l], 0) for l in dets.labels])
        cand = np.nonzero(keep & (need > 0))[0]
        counts = points_in_box_counts(cur, dets.boxes[cand], cfg.box_margin)
        keep[cand[counts < need[cand]]] = False

    dets = dets.select(np.nonzero(keep)[0])
    if cfg.cross_class_nms and len(dets):
        dets = dets.select(np.nonzero(cross_class_suppress(dets, cfg.cross_class_margin))[0])
    return dets
