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
    # box bottom relative to the ground under the box
    max_float: float = 0.7           # [m] above ground: boxes on vegetation / walls / awnings
    max_sink: float = 1.2            # [m] below ground
    # ground under a box: the current sweep's lowest point per ground_cell, and
    # a low percentile of those cells in a ring around the footprint; z = 0 of
    # the det frame when the ring has too few cells. The det frame is only level
    # near the robot: on the moving sample bag the ground 25-40 m away sits
    # 0.2-0.7 m off z = 0 (slopes, non-planar lots)
    local_ground: bool = True
    ground_cell: float = 2.0         # [m]
    ground_ring: float = 4.0         # [m] beyond the footprint radius
    ground_percentile: float = 10.0  # of the cell minima (cells under objects sit higher)
    ground_min_cells: int = 5
    cross_class_nms: bool = True
    cross_class_margin: float = 0.2  # [m] enlargement of the higher-score box


def points_in_box_counts(xyz, boxes, margin=0.1, tree=None):
    """Number of points of (N, 3) ``xyz`` inside each (M, 7) standard box."""
    counts = np.zeros(len(boxes), dtype=np.int64)
    if len(xyz) == 0 or len(boxes) == 0:
        return counts
    tree = tree if tree is not None else cKDTree(xyz[:, :2])
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


def local_ground_heights(xyz, boxes, cell=2.0, ring=4.0, percentile=10.0, min_cells=5):
    """Ground height under each box, computed once per frame on a BEV grid:
    the lowest point of every ``cell`` x ``cell`` column, then a low percentile
    of those minima over a disk of radius ``ring`` around each cell. A box reads
    the cell under its center. Cells under objects sit higher than the ground,
    so a low percentile of the disk still lands on it (a bus covers ~40 % of
    the disk around its center). NaN where the disk has fewer than
    ``min_cells`` occupied cells."""
    from scipy import ndimage

    out = np.full(len(boxes), np.nan)
    if len(xyz) == 0 or len(boxes) == 0:
        return out
    pad = ring + cell
    lo, hi = boxes[:, :2].min(axis=0) - pad, boxes[:, :2].max(axis=0) + pad
    xyz = xyz[np.all((xyz[:, :2] >= lo) & (xyz[:, :2] < hi), axis=1)]   # the cloud reaches > 100 m
    if len(xyz) == 0:
        return out
    ij = np.floor((xyz[:, :2] - lo) / cell).astype(np.int64)
    shape = tuple(np.floor((hi - lo) / cell).astype(np.int64) + 1)
    grid = np.full(shape[0] * shape[1], np.inf)
    np.minimum.at(grid, ij[:, 0] * shape[1] + ij[:, 1], xyz[:, 2])
    grid = grid.reshape(shape)

    r = int(np.ceil(ring / cell))
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    disk = np.hypot(di, dj) * cell <= ring
    occupied = np.isfinite(grid)
    big = 1e6                                                   # empty cells rank last
    ground = ndimage.percentile_filter(np.where(occupied, grid, big), percentile, footprint=disk,
                                       mode="constant", cval=big)
    count = ndimage.correlate(occupied.astype(np.int32), disk.astype(np.int32), mode="constant", cval=0)

    bi = np.floor((boxes[:, :2] - lo) / cell).astype(np.int64)
    inside = (bi[:, 0] >= 0) & (bi[:, 0] < shape[0]) & (bi[:, 1] >= 0) & (bi[:, 1] < shape[1])
    k = np.nonzero(inside)[0]
    g, n = ground[bi[k, 0], bi[k, 1]], count[bi[k, 0], bi[k, 1]]
    # the percentile must land on an occupied cell: at least percentile % of the disk occupied
    ok = (n >= min_cells) & (g < big / 2)
    out[k[ok]] = g[ok]
    return out


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


def _current_sweep(points_det):
    cur = points_det[points_det[:, -1] <= 1e-6, :3] if points_det is not None else np.zeros((0, 3))
    return cur, (cKDTree(cur[:, :2]) if len(cur) else None)


def _bottom_above_ground(dets, cur, cfg):
    ground = np.zeros(len(dets))
    if cfg.local_ground and len(cur):
        g = local_ground_heights(cur, dets.boxes, cfg.ground_cell, cfg.ground_ring, cfg.ground_percentile,
                                 cfg.ground_min_cells)
        ground = np.where(np.isfinite(g), g, 0.0)
    return dets.boxes[:, 2] - dets.boxes[:, 5] / 2 - ground


def filter_measurements(dets, points_det, cfg=None):
    """What the geometric filters test, for every detection: the box bottom above the local ground
    [m] and the current-sweep points in the box grown by ``box_margin`` (None without a cloud).
    Saved with the raw detections (``run_sequence.py --save-raw``), they let the thresholds
    (``max_float``, ``max_sink``, ``min_points``, cross-class NMS) be swept offline without the clouds:
    ``filter_reasons(dets, None, cfg, measured=...)``."""
    cfg = cfg or DetectionFilterConfig()
    cur, tree = _current_sweep(points_det)
    counts = None if points_det is None else points_in_box_counts(cur, dets.boxes, cfg.box_margin, tree)
    return _bottom_above_ground(dets, cur, cfg), counts


def filter_reasons(dets, points_det, cfg=None, measured=None):
    """Why each detection is dropped ('' = kept), in the order the filters apply: "floating" /
    "sunk" (box bottom too far above / below the ground), "few points" (current sweep),
    "duplicate" (inside a higher-scoring box of another class). ``measured``: (bottom, counts) of
    ``filter_measurements``, made with the same ``box_margin`` and ground settings, instead of the cloud."""
    cfg = cfg or DetectionFilterConfig()
    why = np.full(len(dets), "", dtype=object)
    if len(dets) == 0:
        return why
    if measured is None:
        cur, tree = _current_sweep(points_det)
        bottom, counts, have_cloud = _bottom_above_ground(dets, cur, cfg), None, points_det is not None
    else:
        bottom, counts = measured
        have_cloud = counts is not None
    why[~(bottom <= cfg.max_float)] = "floating"
    why[(why == "") & ~(bottom >= -cfg.max_sink)] = "sunk"

    if cfg.min_points and have_cloud:                 # an empty current sweep supports nothing
        need = np.array([cfg.min_points.get(CLASSES[l], 0) for l in dets.labels])
        cand = np.nonzero((why == "") & (need > 0))[0]
        n = counts[cand] if counts is not None else points_in_box_counts(cur, dets.boxes[cand], cfg.box_margin, tree)
        why[cand[n < need[cand]]] = "few points"

    kept = np.nonzero(why == "")[0]
    if cfg.cross_class_nms and len(kept):
        dup = ~cross_class_suppress(dets.select(kept), cfg.cross_class_margin)
        why[kept[dup]] = "duplicate"
    return why


def filter_detections(dets, points_det, cfg=None):
    """Apply the geometric filters. ``points_det``: (N, >=4) det-frame cloud whose
    last column is the sweep time lag (0 = current sweep)."""
    if len(dets) == 0:
        return dets
    return dets.select(np.nonzero(filter_reasons(dets, points_det, cfg) == "")[0])
