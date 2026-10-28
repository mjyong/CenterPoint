"""Self-supervised prediction data from the tracker's own logs.

The tracker output *is* the label source: cut every confirmed track into
(2 s past, 3 s future) windows. Two details matter:

* inputs = the causal tracker estimates (what the network sees online);
* targets = an RTS-smoothed (non-causal) version of the observed positions,
  which removes most tracker lag/noise from the supervision. "Observed": the
  matched detections' centers themselves (``targets="measured"``, the default;
  logs written since 2026-10-28), or the tracker's updated positions in the
  frames it matched a detection (``"filtered"``, older logs). Either is a
  pseudo ground truth; pipeline_eval measures both against the reviewed labels.
  On 0921 (T23) the measured targets of moving vehicles lie 0.2 m from the
  labels at 3 s, the filtered ones 0.5-0.7 m (filtered twice, they overshoot
  when a vehicle slows down).

The state now (velocity, moving) is a target too, for the state head (model.py): the targets'
velocity over +-0.5 s, and moving by the auto-label's rule (autolabel.py: trajectory faster than
0.4 m/s and velocity head faster than 0.15 m/s, short bursts and stops cleaned), applied to the
matched detections. So the network learns what the labels call moving, from the past only.

Windows whose future is mostly coasting (occluded) are dropped, and the split
into train/val should be by recording, not by window (windows overlap).
"""
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

import numpy as np

from ..geometry import yaw_of
from .features import FeatureConfig, build_inputs, future_to_agent_frame

ROW = ("t", "id", "label", "x", "y", "vx", "vy", "yaw", "observed",
       # the matched detection (NaN when coasting): center, yaw, velocity head, score
       "det_x", "det_y", "det_yaw", "det_vx", "det_vy", "det_score",
       "motion")     # the tracker's motion state, MOTION_CODE
OLD_ROW = 9          # logs written before 2026-10-28 stop after "observed"
MEASURED_ROW = 15    # T23's logs stop after "det_score"
MOTION_CODE = {"static": 0, "moving": 1, "stopped": 2}


def log_row(stamp, s):
    """One ``ROW`` of a TrackState (also the runtime buffer of ``learned.LearnedPredictor``)."""
    m = s.measurement if s.measurement is not None else np.full(6, np.nan)
    return (stamp, s.track_id, s.label, s.position[0], s.position[1], s.velocity[0], s.velocity[1], s.yaw,
            0.0 if s.coasting else 1.0, *m, MOTION_CODE.get(s.motion, 0))


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
        self.rows.extend(log_row(stamp, s) for s in track_states)
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
    targets: str = "measured"      # what is smoothed into the targets: "measured" / "filtered" (module docstring)


def split_tracks(rows):
    """{track id: its rows sorted by time}."""
    tracks = defaultdict(list)
    for r in rows:
        tracks[int(r[1])].append(r)
    return {k: np.asarray(sorted(v, key=lambda r: r[0])) for k, v in tracks.items()}


Target = namedtuple("Target", "t xy vel moving")      # over the observed frames: (N,), (N, 2), (N, 2), (N,)


def track_targets(tracks, source="measured", smooth=True):
    """{track id: Target} over the frames each track was observed: smoothed positions, and the
    velocity and moving flags as the auto-label decides them (module docstring).
    ``source``: "filtered" (the tracker's positions) or "measured" (the matched detections' centers)."""
    from ..autolabel import AutoLabelConfig, _clean_moving, _running_median, _window_velocity

    if source not in ("filtered", "measured"):
        raise ValueError("targets: 'filtered' or 'measured', not %r" % source)
    al = AutoLabelConfig()
    out = {}
    for tid, tr in tracks.items():
        if source == "measured" and tr.shape[1] <= OLD_ROW:
            raise ValueError("targets='measured' needs a log with the matched detections (written since "
                             "2026-10-28); for an older log use targets='filtered' (train_predictor.py --targets)")
        obs = tr[:, 8] > 0.5
        if source == "measured":
            obs &= np.isfinite(tr[:, 9])
        if obs.sum() < 3:
            continue
        t = tr[obs][:, 0]
        xy = tr[obs][:, 3:5] if source == "filtered" else tr[obs][:, 9:11]
        xy = rts_smooth(t, xy) if smooth else xy
        v = _window_velocity(t, xy[:, 0], xy[:, 1], al.velocity_window)
        head = tr[obs][:, 12:14] if tr.shape[1] > OLD_ROW else tr[obs][:, 5:7]
        head = _running_median(np.hypot(head[:, 0], head[:, 1]), 2)
        moving = (np.hypot(v[:, 0], v[:, 1]) > al.static_speed) & ~(head <= al.teacher_speed)   # NaN: no head
        moving = _clean_moving(t, moving, al)
        out[tid] = Target(t, xy, np.where(moving[:, None], v, 0.0), moving)
    return out


def state_at(target, t_now, tol=0.15):
    """(velocity (2,), moving) of ``target`` at its observation closest to ``t_now``; None when that
    is more than ``tol`` s away (the track coasts)."""
    if target is None:
        return None
    k = int(np.argmin(np.abs(target.t - t_now)))
    if abs(target.t[k] - t_now) > tol:
        return None
    return target.vel[k], bool(target.moving[k])


def window_target(target, t_now, fut_t, stamps, min_future_observed=0.8):
    """(T, 2) target positions at ``t_now + fut_t`` from ``track_targets``' (times, positions), or None
    when the window is not used for training: the future runs past the track's last observation, or
    the track is observed in fewer than ``min_future_observed`` of the logged frames (``stamps``) of it."""
    if target is None:
        return None
    tt, sm = target.t, target.xy
    q = t_now + fut_t
    if q[-1] > tt[-1] + 1e-6:
        return None
    n_future = int(((stamps > t_now) & (stamps <= q[-1] + 1e-6)).sum())
    span = (tt > t_now) & (tt <= q[-1] + 1e-6)
    if span.sum() < min_future_observed * n_future:
        return None
    return np.column_stack([np.interp(q, tt, sm[:, 0]), np.interp(q, tt, sm[:, 1])])


def build_samples(log, cfg=None):
    """log: dict from ``TrackLogger.to_dict()``. Returns dict of stacked arrays."""
    cfg = cfg or SampleConfig()
    fc = cfg.features
    rows, ego = log["rows"], log.get("ego")
    if len(rows) == 0:
        return {}
    tracks = split_tracks(rows)
    targets = track_targets(tracks, cfg.targets, cfg.smooth_targets)
    if fc.use_detections and rows.shape[1] < MEASURED_ROW:
        raise ValueError("use_detections needs a log with the matched detections (written since 2026-10-28)")
    has_motion = rows.shape[1] > MEASURED_ROW

    online = {}
    if "pred_index" in log:
        for (t, tid), m, p in zip(log["pred_index"], log["pred_modes"], log["pred_probs"]):
            online[(round(float(t), 6), int(tid))] = (m, p)

    stamps = np.unique(rows[:, 0])
    fut_t = fc.fut_dt * np.arange(1, fc.fut_len + 1)
    horizon = fc.hist_dt * (fc.hist_len - 1)
    out = defaultdict(list)
    for t_now in stamps[::cfg.frame_stride]:
        hist, dets, labels, yaws, motion = {}, {}, {}, {}, {}
        for tid, tr in tracks.items():
            m = (tr[:, 0] <= t_now + 1e-6) & (tr[:, 0] >= t_now - horizon - 1e-6)
            if m.sum() == 0 or abs(tr[m][-1, 0] - t_now) > 1e-6:
                continue
            h = tr[m]
            hist[tid] = np.column_stack([h[:, 0], h[:, 3:7], h[:, 8]])
            if fc.use_detections:
                dets[tid] = detection_history(h)
            labels[tid], yaws[tid] = int(h[-1, 2]), float(h[-1, 7])
            motion[tid] = int(h[-1, 15]) if has_motion else -1
        focal, futs, fvalid = [], [], []
        for tid, h in hist.items():
            if h[-1, 0] - h[0, 0] < cfg.min_history - 1e-6:
                continue
            fut = window_target(targets.get(tid), t_now, fut_t, stamps, cfg.min_future_observed)
            if fut is None:
                continue
            focal.append(tid)
            futs.append(fut)
            fvalid.append(np.ones(len(fut_t), np.float32))
        if not focal:
            continue
        eh = ego_history(ego, t_now, horizon) if ego is not None and len(ego) else None
        inp, meta = build_inputs(hist, labels, yaws, focal, t_now, fc, ego_hist=eh, detections=dets)
        for b, tid in enumerate(focal):
            for k, v in inp.items():
                out[k].append(v[b])
            out["future"].append(future_to_agent_frame(futs[b], meta["origins"][b], meta["thetas"][b]).astype(np.float32))
            st = state_at(targets.get(tid), t_now)
            c, s = np.cos(meta["thetas"][b]), np.sin(meta["thetas"][b])
            v = np.zeros(2) if st is None else st[0]
            out["vel_now"].append(np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]], np.float32))
            out["moving_now"].append(np.float32(st is not None and st[1]))
            out["state_valid"].append(np.float32(st is not None))
            out["tracker_motion"].append(motion[tid])            # -1: not logged
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


def detection_history(rows):
    """(K, 6) [t, x, y, vx, vy, score] of the matched detections in a track's log rows (NaN when coasting)."""
    return np.column_stack([rows[:, 0], rows[:, 9:11], rows[:, 12:15]])


def concat_samples(list_of_samples):
    list_of_samples = [s for s in list_of_samples if s]
    if not list_of_samples:
        return {}
    return {k: np.concatenate([s[k] for s in list_of_samples]) for k in list_of_samples[0]}


def select_samples(samples, idx):
    """The windows ``idx`` (index array or bool mask) of a sample dict."""
    return {k: v[idx] for k, v in samples.items()} if samples else {}


def moving_mask(samples, disp=1.5):
    """Windows whose agent ends ``disp`` m or more from where it is now (agent frame: from the origin)."""
    if not samples:
        return np.zeros(0, bool)
    return np.hypot(samples["future"][:, -1, 0], samples["future"][:, -1, 1]) >= disp


def mirror_samples(samples):
    """The windows mirrored left / right in the agent frame (y -> -y, for the agent, its neighbours, the
    future and the raster): a vehicle turning left becomes one turning right, twice the motions to learn
    from. World-frame fields are copied unchanged; they only serve evaluation."""
    out = {k: v.copy() for k, v in samples.items()}
    for k in ("agent_hist", "nbr_hist", "agent_det"):
        if k in out:
            out[k][..., 1] *= -1        # y
            out[k][..., 3] *= -1        # vy
    for k in ("future", "imm_modes", "vel_now"):
        if k in out:
            out[k][..., 1] *= -1
    if "raster" in out:
        out["raster"] = out["raster"][..., ::-1, :].copy()   # rows are agent-frame y
    return out
