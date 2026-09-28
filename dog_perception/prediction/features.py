"""Agent-centric inputs for the learned predictor.

One builder is shared by dataset generation and runtime inference, so the
network sees exactly the same (causal, tracker-filtered) histories online
as it was trained on.

Per focal agent (origin = its current position, +x = its heading):
  agent_hist   (Tp, 6)      [x, y, vx, vy, observed, valid] at t - k*dt
  agent_class  (C,)         one-hot
  nbr_hist     (Nn, Tp, 6)  other agents (and the robot itself) in the same frame
  nbr_static   (Nn, C + 1)  one-hot class + is_robot flag
  nbr_mask     (Nn,)        1 = neighbour slot used
  raster       (1, S, S)    optional static-obstacle occupancy around the agent
"""
from dataclasses import dataclass

import numpy as np

from ..detection.boxes import CLASSES

HIST_FEATS = 6


@dataclass
class FeatureConfig:
    hist_len: int = 20          # 2 s at 0.1 s
    hist_dt: float = 0.1
    fut_len: int = 6            # 3 s at 0.5 s
    fut_dt: float = 0.5
    max_neighbors: int = 16
    neighbor_radius: float = 20.0
    heading_min_speed: float = 0.5
    use_raster: bool = False
    raster_size: int = 64
    raster_res: float = 0.5


def resample_history(hist, t_now, cfg):
    """hist: (K, 6) [t, x, y, vx, vy, observed] sorted by t (world frame).
    Returns (Tp, 6) world-frame [x, y, vx, vy, observed, valid] on the fixed grid."""
    grid = t_now - cfg.hist_dt * np.arange(cfg.hist_len - 1, -1, -1)
    out = np.zeros((cfg.hist_len, HIST_FEATS), dtype=np.float32)
    if hist is None or len(hist) == 0:
        return out
    t = hist[:, 0]
    valid = (grid >= t[0] - 1e-3) & (grid <= t[-1] + 1e-3)
    for c in range(1, 5):
        out[:, c - 1] = np.interp(grid, t, hist[:, c])
    out[:, 4] = np.interp(grid, t, hist[:, 5]) > 0.5
    out[:, 5] = valid
    out[~valid] = 0.0
    return out


def heading_of(hist_row, fallback_yaw, min_speed):
    vx, vy = hist_row[2], hist_row[3]
    return float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > min_speed else float(fallback_yaw)


def to_agent_frame(h, origin, theta):
    """h: (..., 6) world-frame resampled history -> agent frame (in place copy)."""
    c, s = np.cos(theta), np.sin(theta)
    out = h.copy()
    x, y = h[..., 0] - origin[0], h[..., 1] - origin[1]
    out[..., 0] = c * x + s * y
    out[..., 1] = -s * x + c * y
    out[..., 2] = c * h[..., 2] + s * h[..., 3]
    out[..., 3] = -s * h[..., 2] + c * h[..., 3]
    out[..., :4] *= h[..., 5:6]   # zero invalid steps
    return out


def rasterize(points_xy, origin, theta, cfg):
    S, res = cfg.raster_size, cfg.raster_res
    img = np.zeros((1, S, S), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return img
    c, s = np.cos(theta), np.sin(theta)
    x, y = points_xy[:, 0] - origin[0], points_xy[:, 1] - origin[1]
    u = ((c * x + s * y) / res + S / 2).astype(np.int64)
    v = ((-s * x + c * y) / res + S / 2).astype(np.int64)
    m = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    img[0, v[m], u[m]] = 1.0
    return img


def build_inputs(histories, labels, yaws, focal_ids, t_now, cfg, ego_hist=None, obstacles_xy=None):
    """Batch the inputs for ``focal_ids``.

    histories: {id: (K, 6) [t, x, y, vx, vy, observed]} world frame, causal
    labels / yaws: {id: int / float}
    ego_hist: optional (K, 6) robot history (added as a neighbour with is_robot=1)
    Returns (inputs dict of stacked arrays, meta dict with origins / thetas).
    """
    C = len(CLASSES)
    res = {i: resample_history(histories[i], t_now, cfg) for i in histories}
    ego = resample_history(ego_hist, t_now, cfg) if ego_hist is not None else None
    ids = list(res.keys())
    cur = np.array([res[i][-1, :2] for i in ids]) if ids else np.zeros((0, 2))

    B, Nn, Tp = len(focal_ids), cfg.max_neighbors, cfg.hist_len
    out = dict(
        agent_hist=np.zeros((B, Tp, HIST_FEATS), np.float32),
        agent_class=np.zeros((B, C), np.float32),
        nbr_hist=np.zeros((B, Nn, Tp, HIST_FEATS), np.float32),
        nbr_static=np.zeros((B, Nn, C + 1), np.float32),
        nbr_mask=np.zeros((B, Nn), np.float32),
    )
    if cfg.use_raster:
        out["raster"] = np.zeros((B, 1, cfg.raster_size, cfg.raster_size), np.float32)
    origins, thetas = np.zeros((B, 2)), np.zeros(B)

    for b, fid in enumerate(focal_ids):
        h = res[fid]
        origin = h[-1, :2].copy()
        theta = heading_of(h[-1], yaws.get(fid, 0.0), cfg.heading_min_speed)
        origins[b], thetas[b] = origin, theta
        out["agent_hist"][b] = to_agent_frame(h, origin, theta)
        out["agent_class"][b, labels[fid]] = 1.0

        cand = []
        for j, oid in enumerate(ids):
            if oid == fid:
                continue
            d = np.linalg.norm(cur[j] - origin)
            if d <= cfg.neighbor_radius:
                cand.append((d, oid))
        cand.sort(key=lambda x: x[0])
        slots = [(res[oid], labels[oid], 0.0) for _, oid in cand]
        if ego is not None and np.linalg.norm(ego[-1, :2] - origin) <= cfg.neighbor_radius:
            slots.insert(0, (ego, -1, 1.0))    # the robot always gets a slot: people react to it
        for n, (nh, lab, is_robot) in enumerate(slots[:Nn]):
            out["nbr_hist"][b, n] = to_agent_frame(nh, origin, theta)
            if lab >= 0:
                out["nbr_static"][b, n, lab] = 1.0
            out["nbr_static"][b, n, C] = is_robot
            out["nbr_mask"][b, n] = 1.0
        if cfg.use_raster:
            out["raster"][b] = rasterize(obstacles_xy, origin, theta, cfg)
    return out, dict(origins=origins, thetas=thetas, ids=list(focal_ids))


def future_to_agent_frame(fut_xy, origin, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y = fut_xy[..., 0] - origin[0], fut_xy[..., 1] - origin[1]
    return np.stack([c * x + s * y, -s * x + c * y], axis=-1)


def agent_to_world(xy, origin, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * xy[..., 0] - s * xy[..., 1] + origin[0],
                     s * xy[..., 0] + c * xy[..., 1] + origin[1]], axis=-1)
