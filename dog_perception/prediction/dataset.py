"""Self-supervised prediction data from the tracker's own logs.

The tracker output *is* the label source: cut every confirmed track into
(2 s past, 3 s future) windows. Two details matter:

* inputs = the causal tracker estimates (what the network sees online);
* targets = an RTS-smoothed (non-causal) version of the observed positions,
  which removes most tracker lag/noise from the supervision.

Windows whose future is mostly coasting (occluded) are dropped, and the split
into train/val should be by recording, not by window (windows overlap).
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..geometry import yaw_of
from .features import FeatureConfig, build_inputs, future_to_agent_frame

ROW = ("t", "id", "label", "x", "y", "vx", "vy", "yaw", "observed")


class TrackLogger:
    """Collects tracker outputs (+ robot pose, + online tier-1 predictions) frame by frame.

    Logging the online IMM predictions lets tier 1 and tier 2 be compared on
    exactly the same windows with exactly what tier 1 produced at runtime.
    """

    def __init__(self, max_modes=2):
        self.rows = []
        self.ego = []            # (t, x, y, yaw)
        self.max_modes = max_modes
        self.pred_index, self.pred_modes, self.pred_probs = [], [], []

    def add(self, stamp, track_states, T_world_body=None, predictions=None):
        for s in track_states:
            self.rows.append((stamp, s.track_id, s.label, s.position[0], s.position[1],
                              s.velocity[0], s.velocity[1], s.yaw, 0.0 if s.coasting else 1.0))
        if T_world_body is not None:
            self.ego.append((stamp, T_world_body[0, 3], T_world_body[1, 3], yaw_of(T_world_body[:3, :3])))
        for p in predictions or ():
            K = self.max_modes
            order = np.argsort(-p.probs)[:K]
            modes = np.repeat(p.modes[order[:1]], K, axis=0)
            probs = np.zeros(K)
            modes[:len(order)], probs[:len(order)] = p.modes[order], p.probs[order]
            self.pred_index.append((stamp, p.track_id))
            self.pred_modes.append(modes)
            self.pred_probs.append(probs)

    def to_dict(self):
        d = dict(rows=np.asarray(self.rows, dtype=np.float64).reshape(-1, len(ROW)),
                 ego=np.asarray(self.ego, dtype=np.float64).reshape(-1, 4))
        if self.pred_index:
            d.update(pred_index=np.asarray(self.pred_index), pred_modes=np.asarray(self.pred_modes),
                     pred_probs=np.asarray(self.pred_probs))
        return d

    def save(self, path):
        np.savez_compressed(path, **self.to_dict())


def rts_smooth(t, xy, q_acc=1.0, r_pos=0.15):
    """Rauch-Tung-Striebel smoother, CV model, irregular timestamps. Returns (N, 2)."""
    n = len(t)
    if n < 3:
        return xy.copy()
    xs_f, Ps_f, xs_p, Ps_p, Fs = [], [], [], [], []
    x = np.array([xy[0, 0], xy[0, 1], 0.0, 0.0])
    P = np.diag([r_pos ** 2, r_pos ** 2, 4.0, 4.0])
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    R = np.eye(2) * r_pos ** 2
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        x_p, P_p = F @ x, F @ P @ F.T + (q_acc ** 2) * g @ g.T
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)
        x = x_p + K @ (xy[k] - H @ x_p)
        P = (np.eye(4) - K @ H) @ P_p
        xs_f.append(x)
        Ps_f.append(P)
        xs_p.append(x_p)
        Ps_p.append(P_p)
        Fs.append(F)
    xs = [None] * n
    xs[-1] = xs_f[-1]
    P_s = Ps_f[-1]
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
        P_s = Ps_f[k] + C @ (P_s - Ps_p[k + 1]) @ C.T
    return np.asarray(xs)[:, :2]


def ego_history(ego, t_now, horizon):
    """(K, 6) [t, x, y, vx, vy, observed=1] from logged robot poses up to t_now."""
    m = (ego[:, 0] <= t_now + 1e-6) & (ego[:, 0] >= t_now - horizon - 1e-6)
    e = ego[m]
    if len(e) < 2:
        return None
    v = np.gradient(e[:, 1:3], e[:, 0], axis=0)
    return np.column_stack([e[:, 0], e[:, 1:3], v, np.ones(len(e))])


@dataclass
class SampleConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    min_history: float = 0.5       # [s] of track before t_now
    min_future_observed: float = 0.8
    frame_stride: int = 2
    smooth_targets: bool = True


def build_samples(log, cfg=None):
    """log: dict from ``TrackLogger.to_dict()``. Returns dict of stacked arrays."""
    cfg = cfg or SampleConfig()
    fc = cfg.features
    rows, ego = log["rows"], log.get("ego")
    if len(rows) == 0:
        return {}
    tracks = defaultdict(list)
    for r in rows:
        tracks[int(r[1])].append(r)
    tracks = {k: np.asarray(sorted(v, key=lambda r: r[0])) for k, v in tracks.items()}

    targets = {}
    for tid, tr in tracks.items():
        obs = tr[:, 8] > 0.5
        if obs.sum() < 3:
            continue
        xy = tr[obs][:, 3:5]
        sm = rts_smooth(tr[obs][:, 0], xy) if cfg.smooth_targets else xy
        targets[tid] = (tr[obs][:, 0], sm)

    online = {}
    if "pred_index" in log:
        for (t, tid), m, p in zip(log["pred_index"], log["pred_modes"], log["pred_probs"]):
            online[(round(float(t), 6), int(tid))] = (m, p)

    stamps = np.unique(rows[:, 0])
    fut_t = fc.fut_dt * np.arange(1, fc.fut_len + 1)
    horizon = fc.hist_dt * (fc.hist_len - 1)
    out = defaultdict(list)
    for t_now in stamps[::cfg.frame_stride]:
        hist, labels, yaws = {}, {}, {}
        for tid, tr in tracks.items():
            m = (tr[:, 0] <= t_now + 1e-6) & (tr[:, 0] >= t_now - horizon - 1e-6)
            if m.sum() == 0 or abs(tr[m][-1, 0] - t_now) > 1e-6:
                continue
            h = tr[m]
            hist[tid] = np.column_stack([h[:, 0], h[:, 3:7], h[:, 8]])
            labels[tid], yaws[tid] = int(h[-1, 2]), float(h[-1, 7])
        n_future_frames = int(((stamps > t_now) & (stamps <= t_now + fut_t[-1] + 1e-6)).sum())
        focal, futs, fvalid = [], [], []
        for tid, h in hist.items():
            if tid not in targets or h[-1, 0] - h[0, 0] < cfg.min_history - 1e-6:
                continue
            tt, sm = targets[tid]
            q = t_now + fut_t
            ok = (q <= tt[-1] + 1e-6)
            # fraction of the future frames in which the track was actually observed
            span = (tt > t_now) & (tt <= t_now + fut_t[-1] + 1e-6)
            if not ok.all() or span.sum() < cfg.min_future_observed * n_future_frames:
                continue
            fut = np.column_stack([np.interp(q, tt, sm[:, 0]), np.interp(q, tt, sm[:, 1])])
            focal.append(tid)
            futs.append(fut)
            fvalid.append(ok.astype(np.float32))
        if not focal:
            continue
        eh = ego_history(ego, t_now, horizon) if ego is not None and len(ego) else None
        inp, meta = build_inputs(hist, labels, yaws, focal, t_now, fc, ego_hist=eh)
        for b, tid in enumerate(focal):
            for k, v in inp.items():
                out[k].append(v[b])
            out["future"].append(future_to_agent_frame(futs[b], meta["origins"][b], meta["thetas"][b]).astype(np.float32))
            out["future_valid"].append(fvalid[b])
            out["origin"].append(meta["origins"][b])
            out["theta"].append(meta["thetas"][b])
            out["label"].append(labels[tid])
            out["track_id"].append(tid)
            out["t"].append(t_now)
            out["future_world"].append(futs[b])
            if online:
                m, p = online.get((round(float(t_now), 6), tid), (None, None))
                if m is None:     # keep arrays aligned; flagged by zero probabilities
                    m, p = np.repeat(futs[b][None], log["pred_modes"].shape[1], 0) * np.nan, np.zeros(log["pred_probs"].shape[1])
                out["imm_modes"].append(future_to_agent_frame(m, meta["origins"][b], meta["thetas"][b]))
                out["imm_probs"].append(p)
    return {k: np.asarray(v) for k, v in out.items()}


def concat_samples(list_of_samples):
    list_of_samples = [s for s in list_of_samples if s]
    if not list_of_samples:
        return {}
    return {k: np.concatenate([s[k] for s in list_of_samples]) for k in list_of_samples[0]}
