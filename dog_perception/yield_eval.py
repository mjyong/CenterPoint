"""Offline evaluation of the yield gate (yield_gate.py) on recorded scenes with reviewed labels.

The robot stands still in these recordings (1104: at a junction), so a crossing can be imagined from where
it stands: a straight path from the robot, in the det frame of the first frame. Every ``arrive_every``
seconds the robot "arrives" at the start of the path; the gate (fed by the tracker and IMM predictor re-run
on the saved detections, as on the robot) decides frame by frame; when it says go, the robot is assumed to
cross at ``dog_speed`` without stopping. The labels tell what really happened:

- **unsafe**: a labelled vehicle (moving) reached a point of the path less than ``tau_true`` after the robot
  left it, or while the robot was on it (box + ``true_margin``). Must be 0;
- **oracle**: the first frame from the arrival where a gate with perfect knowledge of the future (labels,
  the gate's own margins and ``pet``, no uncertainty) would have gone. ``extra wait`` = gate - oracle;
- **timeout**: no go within ``max_wait``.

Labels are keyframes (``labels_sustech2.pkl``, about 2 Hz): positions are interpolated between keyframes of
a track no more than ``max_gap`` apart. An episode whose crossing ends after the last label is unknown.
"""
from dataclasses import dataclass, field

import numpy as np

from .detection.boxes import Detections
from .prediction.kinematic import IMMPredictor
from .tracking import MultiObjectTracker, TrackerConfig
from .yield_gate import YieldConfig, YieldGate, assess, inside_boxes, resample_path


@dataclass
class EvalConfig:
    arrive_every: float = 1.0       # [s] between arrivals
    warmup: float = 3.0             # [s] after the first frame: the tracker needs a few frames
    max_wait: float = 30.0          # [s] no go by then: timeout
    latency: float = 0.15           # [s] the gate runs this long after the scan stamp (node + DDS + planning)
    tau_true: float = 1.0           # [s] post-encroachment time below this is a serious conflict
    true_margin: float = 0.3        # [m] each side of a labelled vehicle for "unsafe"
    min_speed: float = 0.3          # [m/s] labelled vehicles slower than this do not count as traffic
    max_gap: float = 1.1            # [s] longest keyframe gap of a track that is interpolated
    grid_dt: float = 0.1            # [s] time grid of the labels
    classes: tuple = ("vehicle", "cyclist")


# ------------------------------------------------------------------ perception, as on the robot
def track_objects(results, pred_horizon=3.0, tracker_cfg=None):
    """Re-run the tracker and the IMM predictor on saved detections (run_sequence.py results.pkl rows).
    Returns [(stamp, objects)] with gate objects in the world frame."""
    tracker = MultiObjectTracker(tracker_cfg or TrackerConfig())
    predictor = IMMPredictor(horizon=pred_horizon)
    out = []
    for r in results:
        stamp = float(r["stamp"])
        if r.get("pose_bad"):
            dets = Detections(frame="world", stamp=stamp)
        else:
            dets = r["detections"].transform(np.asarray(r["T_world_det"]), "world")
        states = tracker.step(dets, stamp)
        preds = {p.track_id: p for p in predictor(tracker)}
        objs = []
        for st in states:
            p = preds.get(st.track_id)
            objs.append(dict(
                id=int(st.track_id), label=st.name, position=st.position[:2].copy(), velocity=st.velocity.copy(),
                yaw=float(st.yaw), size=st.size.copy(), motion=st.motion, age=float(st.age),
                covariance=st.covariance, standing_for=float(min(st.age, st.time_since_moving)),
                pred_times=None if p is None else p.times, pred_modes=None if p is None else p.modes,
                pred_probs=None if p is None else p.probs, pred_covs=None if p is None else p.covs))
        out.append((stamp, objs))
    return out


# ------------------------------------------------------------------ labels
def gt_tracks(labels, classes=("vehicle", "cyclist")):
    """Labelled tracks in the world frame: id -> dict(t, xy, yaw (unwrapped), l, w). Ignore boxes left out."""
    rows = {}
    for fr in labels["frames"]:
        if fr.get("pose_bad"):
            continue
        T = np.asarray(fr["T_world_det"])
        dyaw = np.arctan2(T[1, 0], T[0, 0])
        for o in fr["objects"]:
            if o["label"] not in classes or o.get("ignore"):
                continue
            b = np.asarray(o["box"], dtype=float)
            xy = T[:2, :3] @ b[:3] + T[:2, 3]
            rows.setdefault(o["track_id"], []).append((float(fr["stamp"]), xy[0], xy[1], b[6] + dyaw, b[3], b[4]))
    tracks = {}
    for tid, r in rows.items():
        r = np.array(sorted(r))
        _, keep = np.unique(r[:, 0], return_index=True)
        r = r[keep]
        tracks[tid] = dict(t=r[:, 0], xy=r[:, 1:3], yaw=np.unwrap(r[:, 3]), l=float(np.median(r[:, 4])),
                           w=float(np.median(r[:, 5])))
    return tracks


def gt_sample(track, times, max_gap):
    """Interpolated (xy (K, 2), yaw (K,), speed (K,), valid (K,)) of a labelled track at ``times``."""
    t = track["t"]
    times = np.asarray(times, dtype=float)
    if len(t) == 1:
        valid = np.abs(times - t[0]) < 1e-6
        return (np.repeat(track["xy"], len(times), 0), np.full(len(times), track["yaw"][0]),
                np.zeros(len(times)), valid)
    k = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 2)
    dt = t[k + 1] - t[k]
    valid = (times >= t[0]) & (times <= t[-1]) & (dt <= max_gap)
    xy = np.column_stack([np.interp(times, t, track["xy"][:, 0]), np.interp(times, t, track["xy"][:, 1])])
    yaw = np.interp(times, t, track["yaw"])
    v = np.diff(track["xy"], axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]
    return xy, yaw, np.linalg.norm(v[k], axis=1), valid


@dataclass
class Occupancy:
    """Which labelled vehicle covers which path point at which grid time."""
    grid: np.ndarray                # (K,) times
    occ: np.ndarray                 # (J, K) bool
    who: np.ndarray                 # (J, K) track id (-1 none)
    nxt: np.ndarray = None          # (J, K + 1) first occupied k' >= k (K: none)
    prv: np.ndarray = None          # (J, K + 1) last occupied k' < k (-1: none); prv[:, k] looks before k

    def __post_init__(self):
        J, K = self.occ.shape
        self.nxt = np.full((J, K + 1), K, dtype=np.int64)
        self.prv = np.full((J, K + 1), -1, dtype=np.int64)
        for k in range(K - 1, -1, -1):
            self.nxt[:, k] = np.where(self.occ[:, k], k, self.nxt[:, k + 1])
        for k in range(K):
            self.prv[:, k + 1] = np.where(self.occ[:, k], k, self.prv[:, k])

    def index(self, t):
        return np.clip(np.searchsorted(self.grid, t - 1e-9), 0, len(self.grid))


def gt_occupancy(points, tracks, grid, extra, min_speed, max_gap, moving=True):
    """Labelled vehicles over the path points: boxes grown by ``extra`` each side; ``moving``: only while
    at least ``min_speed``, else only slower (parked)."""
    occ = np.zeros((len(points), len(grid)), dtype=bool)
    who = np.full(occ.shape, -1, dtype=np.int64)
    for tid, tr in tracks.items():
        xy, yaw, speed, valid = gt_sample(tr, grid, max_gap)
        m = valid & ((speed >= min_speed) if moving else (speed < min_speed))
        if not m.any():
            continue
        ks = np.nonzero(m)[0]
        hit = inside_boxes(points, xy[ks], yaw[ks], np.full(len(ks), 0.5 * tr["l"] + extra),
                           np.full(len(ks), 0.5 * tr["w"] + extra))
        sub = occ[:, ks]
        who[:, ks] = np.where(hit & ~sub, tid, who[:, ks])
        occ[:, ks] = sub | hit
    return Occupancy(grid, occ, who)


def gap_after(o, lo, hi):
    """Unsafe check: per path point, first labelled hit at or after the robot arrives (``lo``) minus when
    it leaves (``hi``); negative while the robot is there. Returns (min gap, track id)."""
    K = len(o.grid)
    J = o.occ.shape[0]
    k = o.nxt[np.arange(J), o.index(lo)]
    has = k < K
    if not has.any():
        return np.inf, -1
    g = np.where(has, o.grid[np.minimum(k, K - 1)] - hi, np.inf)
    j = int(np.argmin(g))
    return float(g[j]), int(o.who[j, k[j]])


def gap_both(o, now, lo, hi):
    """Oracle check, as the gate sees it (vehicles from ``now`` on, gap on either side of the robot)."""
    K = len(o.grid)
    J = o.occ.shape[0]
    ar = np.arange(J)
    kl = o.index(lo)
    k1 = o.nxt[ar, kl]                                         # first hit at or after the robot arrives
    g1 = np.where(k1 < K, np.maximum(o.grid[np.minimum(k1, K - 1)] - hi, lo - o.grid[np.minimum(k1, K - 1)]),
                  np.inf)
    k0 = o.prv[ar, kl]                                         # last hit before the robot arrives
    t0 = o.grid[np.maximum(k0, 0)]
    g0 = np.where((k0 >= 0) & (t0 >= now - 1e-9), lo - t0, np.inf)
    return float(np.min(np.minimum(g0, g1)))


# ------------------------------------------------------------------ paths
def fan_paths(n, length, start=0.0):
    """``n`` straight paths from the robot in the det frame (x right, y forward): name -> (2, 2) points.
    The name is the angle in degrees counter-clockwise from forward (090 = to the left)."""
    out = {}
    for i in range(n):
        a = np.radians(360.0 * i / n)
        u = np.array([-np.sin(a), np.cos(a)])
        out["fan%03d" % round(360.0 * i / n)] = np.vstack([start * u, (start + length) * u])
    return out


def det_to_world(path_det, T_world_det):
    p = np.column_stack([path_det, np.zeros(len(path_det))])
    return (p @ np.asarray(T_world_det)[:3, :3].T + np.asarray(T_world_det)[:3, 3])[:, :2]


# ------------------------------------------------------------------ one scene, one path
@dataclass
class Episode:
    scene: str
    path: str
    arrive: float
    go: float = None                # None: timeout
    reason: str = ""                # what the gate last waited for
    oracle: float = None            # None: the oracle would not go within max_wait
    known: bool = True
    gap_true: float = np.inf        # labelled post-encroachment time of the crossing [s]
    gap_track: int = -1
    unsafe: bool = False            # went, known, and gap_true < tau_true
    oracle_ok_at_go: bool = True    # False: went where the oracle would have waited ("risky")
    waited_for: list = field(default_factory=list)
    why: dict = None                # unsafe only: the labelled vehicle when the robot went (explain())
    t0: float = 0.0                 # first frame of the scene


def explain(track, objs, stamp, T_world_det, ev=EvalConfig(), match=2.5):
    """Where the labelled vehicle was at ``stamp`` (robot's det frame): range [m], bearing [deg, counter-
    clockwise from forward, 0-360], speed [km/h], and whether the tracker reported a vehicle within ``match``."""
    xy, _, speed, valid = gt_sample(track, [stamp], ev.max_gap)
    if not valid[0]:
        return dict(labelled=False)
    T = np.asarray(T_world_det)
    d = T[:2, :2].T @ (xy[0] - T[:2, 3])
    tracked = any(o["label"] == "vehicle" and np.linalg.norm(np.asarray(o["position"][:2]) - xy[0]) <= match
                  for o in objs)
    return dict(labelled=True, range=float(np.hypot(*d)), bearing=float(np.degrees(np.arctan2(-d[0], d[1])) % 360),
                speed_kmh=float(speed[0] * 3.6), tracked=bool(tracked))


def evaluate_path(scene, name, path_world, frames, tracks, gt_end, cfg=YieldConfig(), ev=EvalConfig(),
                  T_world_det=None):
    """Episodes of one path in one scene. ``frames``: track_objects() output; ``tracks``: gt_tracks();
    ``T_world_det``: the robot's det frame, to explain unsafe episodes (bearing)."""
    pts, s = resample_path(path_world, cfg.path_step, cfg.path_length)
    t0 = frames[0][0]
    grid = np.arange(t0 - 2.0, gt_end + 1e-9, ev.grid_dt)
    corridor = 0.5 * cfg.dog_width + cfg.lateral_margin
    occ_true = gt_occupancy(pts, tracks, grid, 0.5 * cfg.dog_width + ev.true_margin, ev.min_speed, ev.max_gap)
    occ_gate = gt_occupancy(pts, tracks, grid, cfg.vehicle_margin + corridor, ev.min_speed, ev.max_gap)
    occ_parked = gt_occupancy(pts, tracks, grid, 0.5 * cfg.dog_width, ev.min_speed, ev.max_gap, moving=False)
    rel_lo = np.maximum(0.0, s - 0.5 * cfg.dog_length) / cfg.dog_speed
    rel_hi = (s + 0.5 * cfg.dog_length) / cfg.dog_speed
    crossing = rel_hi[-1]

    nows = np.array([st + ev.latency for st, _ in frames])
    A = [assess(path_world, objs, st + ev.latency, st, cfg) for st, objs in frames]
    known = nows + crossing + max(ev.tau_true, cfg.pet) <= gt_end
    oracle_ok = np.array([gap_both(occ_gate, now, now + rel_lo, now + rel_hi) >= cfg.pet for now in nows])

    eps = []
    for ta in np.arange(t0 + ev.warmup, nows[-1], ev.arrive_every):
        k0 = int(np.searchsorted(nows, ta))
        if k0 >= len(nows):
            break
        e = Episode(scene, name, float(nows[k0]), t0=float(t0))
        gate = YieldGate(cfg)
        waited = set()
        for k in range(k0, len(nows)):
            if nows[k] - e.arrive > ev.max_wait:
                break
            d = gate.update(A[k], nows[k], dog_moving=False)
            if d.go:
                e.go = float(nows[k])
                e.known = bool(known[k])
                e.gap_true, e.gap_track = gap_after(occ_true, e.go + rel_lo, e.go + rel_hi)
                e.unsafe = e.known and e.gap_true < ev.tau_true
                if e.unsafe and e.gap_track in tracks:
                    e.why = explain(tracks[e.gap_track], frames[k][1], frames[k][0],
                                    np.eye(4) if T_world_det is None else T_world_det, ev)
                e.oracle_ok_at_go = bool(oracle_ok[k])
                break
            e.reason = d.reason
            waited.update(d.ids)
        else:
            e.known = False                                    # the scene ended before a go or a timeout
        e.waited_for = sorted(waited)
        ko = [k for k in range(k0, len(nows)) if nows[k] - e.arrive <= ev.max_wait and oracle_ok[k] and known[k]]
        e.oracle = float(nows[ko[0]]) if ko else None
        eps.append(e)
    parked = float(np.mean(occ_parked.occ.any(0))) if occ_parked.occ.size else 0.0
    return eps, dict(parked_frac=parked, crossing_s=float(crossing))


# ------------------------------------------------------------------ summaries
def summarize(eps):
    known = [e for e in eps if e.known]
    went = [e for e in known if e.go is not None]
    waits = np.array([e.go - e.arrive for e in went])
    extra = np.array([e.go - e.oracle for e in went if e.oracle is not None])
    owait = np.array([e.oracle - e.arrive for e in known if e.oracle is not None])

    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")
    return dict(episodes=len(known), unknown=len(eps) - len(known), went=len(went),
                unsafe=sum(e.unsafe for e in went), risky=sum(not e.oracle_ok_at_go for e in went),
                timeouts=sum(e.go is None for e in known),
                oracle_timeouts=sum(e.oracle is None for e in known),
                wait_med=q(waits, 50), wait_p90=q(waits, 90), wait_max=q(waits, 100),
                oracle_wait_med=q(owait, 50), oracle_wait_p90=q(owait, 90),
                extra_med=q(extra, 50), extra_p90=q(extra, 90), extra_max=q(extra, 100),
                immediate=int(np.sum(waits < 1e-6)) if len(waits) else 0)


def traffic_stats(tracks, dog_xy, grid, ev=EvalConfig(), radius=30.0):
    """Per labelled track that moves within ``radius`` of the robot: its top speed there [m/s]."""
    top = []
    for tr in tracks.values():
        xy, _, speed, valid = gt_sample(tr, grid, ev.max_gap)
        m = valid & (np.linalg.norm(xy - dog_xy, axis=1) <= radius) & (speed >= 1.0)
        if m.any():
            top.append(float(speed[m].max()))
    return np.array(top)


def first_seen(tracks, frames, dog_xy, ev=EvalConfig(), far=25.0, match=2.5):
    """Labelled vehicles that come from at least ``far`` [m] and get closer: range [m] when the tracker
    first reports a vehicle within ``match`` of them (NaN: never)."""
    out = []
    stamps = np.array([st for st, _ in frames])
    for tid, tr in tracks.items():
        xy, _, speed, valid = gt_sample(tr, stamps, ev.max_gap)
        r = np.linalg.norm(xy - dog_xy, axis=1)
        if not valid.any() or speed[valid].max() < 1.0:
            continue
        kv = np.nonzero(valid)[0]
        if r[kv[0]] < far or r[kv].min() > r[kv[0]] - 5.0:
            continue
        seen = np.nan
        for k in kv:
            if any(o["label"] == "vehicle" and np.linalg.norm(np.asarray(o["position"][:2]) - xy[k]) <= match
                   for o in frames[k][1]):
                seen = float(r[k])
                break
        out.append((tid, float(r[kv[0]]), seen))
    return out
