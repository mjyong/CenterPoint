"""What the robot gets after the detector, scored against the reviewed labels (docs/PIPELINE_TUNING.md).

The back half of the pipeline (geometric filters, tracker, motion state, IMM prediction) runs on CPU,
so it is re-run offline from saved detections (``run_sequence.py`` results.pkl) for any set of
parameters, scene by scene, and compared with the reviewed keyframe labels (2 Hz):

* detections at the reporting threshold (``TrackerConfig.high_score``): precision / recall within
  ``near_range`` and beyond, false detections per frame, AP@2 / AP@4 m;
* tracks: CLEAR MOT and IDF1 on the keyframes (2 m), ID switches and false tracks per minute, how
  long a vehicle that comes into ``near_range`` waits for a confirmed track, and why tracks are false
  and vehicles missed (cause, range, points, blind-sector zone);
* motion state against the labels' ``moving`` flags (``tracking/motion_eval.py``);
* prediction against the labelled future (keyframes every 0.5 s are the 0.5 s prediction steps):
  ADE / FDE of the most likely mode and over all modes, miss rate, at 1 / 2 / 3 s, IMM against
  constant velocity, for moving vehicles split into straight / turning and by speed;
* a hazard proxy: a moving vehicle that comes within ``hazard_radius`` of the robot within
  ``hazard_horizon`` should raise an alarm (a moving track whose most likely predicted path, or its
  position now, comes that close), how early it does, and alarms without such a vehicle.

Ground truth: vehicles that are not ignore boxes, lie within ``max_range`` and hold at least one
current-sweep point. The others (ignore boxes, hidden ones, those just beyond the range) are not
scored, and a detection or track on them is not a false one. The labels' track ids are stitched
first: the auto-label splits a vehicle the teacher lost for a while into two tracks, which would
count against the tracker as ID switches and lower IDF1 (``stitch_gt``).

Detections come from ``results`` (as saved: already filtered), ``raw`` (saved before the filters
with ``--save-raw``, filtered here with ``EvalConfig.filters``) or ``labels`` (the 10 Hz labels as
perfect detections: what the tracker and the predictor do with a perfect detector).
"""
import copy
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, fields, replace
from itertools import product

import numpy as np

from .blind_sector import BLIND_SECTOR_DEG, EDGE_DEG, zone_of
from .detection.boxes import CLASSES, Detections, score_threshold_array
from .detection.evaluate import _on_ignore, evaluate_detections, match_frame
from .detection.filters import DetectionFilterConfig, filter_reasons
from .geometry import inv_T, wrap_angle
from .prediction.kinematic import IMMPredictor
from .prediction.metrics import displacement_metrics
from .tracking import MultiObjectTracker, TrackerConfig
from .tracking.evaluate import MotAccumulator
from .tracking.motion_eval import MotionStateAccumulator
from .tracking.tracker import ClassParams

VEHICLE = CLASSES.index("vehicle")
HORIZONS = (1.0, 2.0, 3.0)
RANGE_BINS = ((10, "0-10 m"), (20, "10-20 m"), (30, "20-30 m"), (None, "30-40 m"))
POINT_BINS = ((10, "1-9 pts"), (50, "10-49 pts"), (None, ">=50 pts"))
FP_CAUSES = ("second track on a tracked vehicle", "beside a vehicle (2-4 m off)", "coasting after the vehicle was lost",
             "nothing labelled nearby")
FN_CAUSES = ("detected, not tracked", "not detected")
# filter settings that change what is measured, not only the thresholds: sweeping them needs the clouds
_MEASURED = ("box_margin", "local_ground", "ground_cell", "ground_ring", "ground_percentile", "ground_min_cells")


@dataclass
class EvalConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    filters: DetectionFilterConfig = field(default_factory=DetectionFilterConfig)
    predictor: dict = field(default_factory=lambda: dict(horizon=3.0, step=0.5, merge_dist=0.25))
    predict: bool = True
    # tier-2 model (.pt, tools/dog/train_predictor.py) instead of the IMM rollout; "" = IMM. A sweep
    # --set eval.learned_model=,model.pt compares the two on the same tracks (paired)
    learned_model: str = ""
    max_range: float = 40.0          # [m] scored range, from the robot
    near_range: float = 30.0         # [m] where a vehicle matters to the robot (DATA_COLLECTION_PLAN.md 4)
    match_dist: float = 2.0          # [m] center distance of a match
    moving_disp: float = 1.5         # [m] in the 3 s horizon: a moving prediction sample
    turn_deg: float = 15.0           # heading change over the horizon: turning
    hazard_radius: float = 5.0       # [m]
    hazard_horizon: float = 3.0      # [s]
    bad_fde: float = 3.0             # [m] predictions listed as issues
    # [s] after its last / before its first label a vehicle is still there, not scored (``_unlabelled``);
    # None: the vehicle max_age + 0.5 s, so a track coasting into the blind sector is never a false one
    unlabelled_grace: float = None
    sector: tuple = BLIND_SECTOR_DEG
    edge: float = EDGE_DEG


# ------------------------------------------------------------------ parameters
def parse_value(text):
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for kind in (int, float):
        try:
            return kind(t)
        except ValueError:
            pass
    return t


def _models(text):
    """IMM models "cv:1.5+ct:1.5:0.25" -> (("cv", 1.5), ("ct", 1.5, 0.25))."""
    return tuple((p.split(":")[0],) + tuple(float(v) for v in p.split(":")[1:]) for p in str(text).split("+"))


def _set_field(obj, name, value, cls=None):
    """``obj`` with field ``name`` set; a {class: value} field takes ``value`` for ``cls`` (vehicle by default)."""
    if name not in {f.name for f in fields(obj)}:
        raise KeyError("%s has no field %r" % (type(obj).__name__, name))
    cur = getattr(obj, name)
    if isinstance(cur, dict):
        value = dict(cur, **{cls or "vehicle": value})
    elif cls is not None:
        raise KeyError("%s.%s is not per class" % (type(obj).__name__, name))
    return replace(obj, **{name: value})


def apply_overrides(cfg, overrides):
    """A copy of ``cfg`` with ``overrides`` ({key: value}) applied. Keys:

    tracker.<field>[.<class>]   TrackerConfig (a per-class field without a class: vehicle)
    tracker.<class>.<field>     ClassParams of that class; ``models`` as "cv:1.5+ct:1.5:0.25"
    filter.<field>[.<class>]    DetectionFilterConfig (``min_points`` is per class)
    predict.<field>             IMMPredictor (horizon, step, merge_dist)
    eval.<field>                EvalConfig (max_range, hazard_radius, ...)"""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        part = key.split(".")
        if part[0] == "tracker" and len(part) == 3 and part[1] in CLASSES:
            cp = cfg.tracker.class_params[part[1]]
            if part[2] not in {f.name for f in fields(ClassParams)}:
                raise KeyError("ClassParams has no field %r" % part[2])
            value = _models(value) if part[2] == "models" else value
            cfg.tracker.class_params = dict(cfg.tracker.class_params, **{part[1]: replace(cp, **{part[2]: value})})
        elif part[0] == "tracker" and len(part) in (2, 3):
            cfg.tracker = _set_field(cfg.tracker, part[1], value, part[2] if len(part) == 3 else None)
        elif part[0] == "filter" and len(part) in (2, 3):
            cfg.filters = _set_field(cfg.filters, part[1], value, part[2] if len(part) == 3 else None)
        elif part[0] == "predict" and len(part) == 2:
            if part[1] not in cfg.predictor:
                raise KeyError("predict.%s: one of %s" % (part[1], ", ".join(cfg.predictor)))
            cfg.predictor = dict(cfg.predictor, **{part[1]: value})
        elif part[0] == "eval" and len(part) == 2:
            cfg = _set_field(cfg, part[1], value)
        else:
            raise KeyError("unknown parameter %r (tracker. / filter. / predict. / eval.)" % key)
    return cfg


def parse_grid(sets):
    """["tracker.vehicle.gate=2,3", "tracker.high_score=0.3,0.35"] -> list of override dicts, every
    combination, the first values first. A key given again replaces the earlier values (fixed settings
    first, the ones to sweep after them)."""
    grid = {}
    for s in sets or ():
        k, v = s.split("=", 1)
        k = k.strip()
        grid.pop(k, None)
        grid[k] = [parse_value(x) for x in v.split(",")] if not k.endswith(".models") else v.split(",")
    return [dict(zip(grid, combo)) for combo in product(*grid.values())] if grid else [{}]


def needs_clouds(cfg):
    base = DetectionFilterConfig()
    return any(getattr(cfg.filters, k) != getattr(base, k) for k in _MEASURED)


# ------------------------------------------------------------------ scene data
def _to_world(box, T):
    return Detections(np.asarray(box, float)[None], np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64)).transform(
        np.asarray(T, float), "world").boxes[0]


def stitch_gt(tracks, static_spread=1.0, same_place=1.5, max_gap=3.0, link_dist=2.5):
    """Label track ids that are one vehicle. ``tracks``: {track id: (stamps (K,), world xy (K, 2))}.

    Track b continues track a when b starts after a ends and either both stand still (90 % of the
    centers within ``static_spread`` of their median) within ``same_place`` m of each other, at any gap
    (a parked vehicle the teacher lost for a while), or b starts within ``max_gap`` s where a, going
    on at its last speed, would be (``link_dist`` m). Tracks seen in the same keyframe are never one.
    Returns ({track id: stitched id}, merges); the stitched id is the group's earliest track id."""
    info = {}
    for tid, (t, xy) in tracks.items():
        t, xy = np.asarray(t, float), np.asarray(xy, float).reshape(-1, 2)
        o = np.argsort(t)
        t, xy = t[o], xy[o]
        med = np.median(xy, 0)
        static = float(np.percentile(np.hypot(*(xy - med).T), 90)) < static_spread
        last = t >= t[-1] - 1.0
        dt = t[last][-1] - t[last][0]
        v = (xy[last][-1] - xy[last][0]) / dt if dt > 0 else np.zeros(2)
        info[tid] = (t, xy, med, static, v)
    cands = []
    for a, b in product(info, info):
        ta, xa, ma, sa, va = info[a]
        tb, xb, mb, sb, _ = info[b]
        if a == b or ta[-1] >= tb[0]:
            continue
        gap = tb[0] - ta[-1]
        if (sa and sb and np.hypot(*(ma - mb)) < same_place) or \
                (gap <= max_gap and np.hypot(*(xb[0] - xa[-1] - va * gap)) <= link_dist):
            cands.append((gap, a, b))
    parent = {t: t for t in info}
    stamps = {t: set(np.round(info[t][0], 3)) for t in info}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    merges = 0
    for _, a, b in sorted(cands, key=lambda c: (c[0], str(c[1]), str(c[2]))):
        ra, rb = find(a), find(b)
        if ra == rb or stamps[ra] & stamps[rb]:
            continue
        if info[rb][0][0] < info[ra][0][0]:
            ra, rb = rb, ra
        parent[rb] = ra
        stamps[ra] |= stamps[rb]
        merges += 1
    return {t: find(t) for t in info}, merges


@dataclass
class SceneData:
    name: str
    rows: list                    # results.pkl rows, by stamp
    keyframes: dict               # round(stamp, 6) -> reviewed labels frame
    dense: dict = None            # round(stamp, 6) -> 10 Hz labels frame (drawing, ``labels`` detections)
    gt_id: dict = None            # label track id -> stitched id
    gt_track: dict = None         # stitched id -> dict(t, xy, zb (box bottom), moving, box (world), by stamp)
    merges: int = 0
    dense_id: dict = field(default_factory=dict)   # 10 Hz label track id -> keyframe label track id


def _key(stamp):
    return round(float(stamp), 6)


def load_scene(name, rows, labels, dense=None, **stitch):
    """``rows``: results.pkl rows; ``labels``: reviewed keyframe labels (labels_sustech2.pkl);
    ``dense``: optional 10 Hz labels (labels_train2.pkl / labels_full2.pkl)."""
    rows = sorted(rows, key=lambda r: r["stamp"])
    key = {_key(f["stamp"]): f for f in labels["frames"] if not f.get("pose_bad")}
    obs = defaultdict(list)
    for f in key.values():
        for o in f["objects"]:
            if o["label"] == "vehicle" and not o.get("ignore"):
                w = _to_world(o["box"], f["T_world_det"])
                obs[o["track_id"]].append((f["stamp"], w[:2], w[2] - w[5] / 2, bool(o.get("moving")), w))
    gt_id, merges = stitch_gt({t: ([s for s, *_ in v], np.array([p for _, p, *_ in v])) for t, v in obs.items()},
                              **stitch)
    groups = defaultdict(list)
    for tid, v in obs.items():
        groups[gt_id[tid]] += v
    gt_track = {}
    for g, v in groups.items():
        v.sort(key=lambda x: x[0])
        t, xy = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        gt_track[g] = dict(t=t, xy=xy, zb=np.array([x[2] for x in v]), moving=np.array([x[3] for x in v]),
                           box=np.array([x[4] for x in v]), v_start=_edge_velocity(t, xy, False),
                           v_end=_edge_velocity(t, xy, True))
    dense_key = None if dense is None else {_key(f["stamp"]): f for f in dense["frames"] if not f.get("pose_bad")}
    return SceneData(name, rows, key, dense_key, gt_id, gt_track, merges, match_dense_ids(key, dense_key or {}))


def match_dense_ids(keyframes, dense, label="vehicle", dist=0.5):
    """{10 Hz label track id: keyframe label track id}. ``propagate_keyframes`` renumbers the boxes the
    reviewer added, so the ids are matched where both have a box: at the keyframes, by position.
    ``keyframes`` / ``dense``: {round(stamp, 6): labels frame}."""
    out = {}
    for k, f in keyframes.items():
        d = dense.get(k)
        kv = [o for o in f["objects"] if o["label"] == label]
        for o in (d["objects"] if d else []):
            if o["label"] != label or o["track_id"] in out or not kv:
                continue
            gap = [np.hypot(*(np.asarray(o["box"][:2]) - q["box"][:2])) for q in kv]
            if min(gap) < dist:
                out[o["track_id"]] = kv[int(np.argmin(gap))]["track_id"]
    return out


def _at(track, times, tol=0.06):
    """(len(times), 2) positions of a stitched GT track at ``times``; NaN where it has no keyframe."""
    out = np.full((len(times), 2), np.nan)
    if track is None or len(track["t"]) == 0:
        return out
    i = np.clip(np.searchsorted(track["t"], times), 0, len(track["t"]) - 1)
    for k, (t, j) in enumerate(zip(times, i)):
        for jj in (j - 1, j):
            if 0 <= jj < len(track["t"]) and abs(track["t"][jj] - t) <= tol:
                out[k] = track["xy"][jj]
    return out


def _edge_velocity(t, xy, end, window=1.0):
    """Mean velocity over the first / last ``window`` s of a track's labelled centers."""
    m = t >= t[-1] - window if end else t <= t[0] + window
    tt, pp = t[m], xy[m]
    return (pp[-1] - pp[0]) / (tt[-1] - tt[0]) if len(tt) > 1 and tt[-1] > tt[0] else np.zeros(2)


def _unlabelled(track, stamp, grace):
    """World box of a stitched GT track at ``stamp`` when the labels leave it out although the vehicle
    is there: between two of its labelled stretches (the nearest box moved to the interpolated
    center), or up to ``grace`` s after its last / before its first label (moved on at its speed
    there). It went into the blind sector or behind something, or the labels pick it up a little late;
    a track coasting there, or started early, is not a false one."""
    t = track["t"]
    if np.abs(t - stamp).min() <= 0.06:
        return None
    if t[0] < stamp < t[-1]:
        box = track["box"][int(np.argmin(np.abs(t - stamp)))].copy()
        box[0], box[1] = np.interp(stamp, t, track["xy"][:, 0]), np.interp(stamp, t, track["xy"][:, 1])
        return box
    if t[-1] < stamp <= t[-1] + grace:
        box = track["box"][-1].copy()
        box[:2] = track["xy"][-1] + track["v_end"] * (stamp - t[-1])
        return box
    if t[0] - grace <= stamp < t[0]:
        box = track["box"][0].copy()
        box[:2] = track["xy"][0] - track["v_start"] * (t[0] - stamp)
        return box
    return None


def _path(track, t0, t1, zb=0.0):
    if track is None:
        return None
    m = (track["t"] >= t0 - 1e-6) & (track["t"] <= t1 + 1e-6)
    if m.sum() < 2:
        return None
    return np.column_stack([track["xy"][m], track["zb"][m] + 0.1])


def _runs(idx):
    """Runs of consecutive integers in a sorted list: [(first, last)]."""
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(r) for r in out]


def _bin(value, bins):
    """Name of the first bin whose upper end (exclusive, None = open) lies above ``value``."""
    return next(name for hi, name in bins if hi is None or value < hi)


# ------------------------------------------------------------------ one scene
@dataclass
class SceneResult:
    name: str
    t0: float
    predictor: str = "IMM"        # what made the predictions: "IMM" (tier 1) or "learned" (tier 2)
    frames: int = 0
    keyframes: int = 0
    keyframe_s: float = 0.0
    counts: Counter = field(default_factory=Counter)
    mot: dict = field(default_factory=dict)
    fp_episodes: int = 0
    fp_episodes_long: int = 0
    latencies: list = field(default_factory=list)
    never_tracked: int = 0
    motion_rows: dict = field(default_factory=dict)
    motion_s: float = 0.0
    pred: list = field(default_factory=list)
    hazard: list = field(default_factory=list)
    false_alarms: int = 0
    det_frames: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    timings: dict = field(default_factory=lambda: defaultdict(list))
    snapshot: dict = None
    merges: int = 0
    mcap: tuple = None            # (path, messages, bytes) when drawn


def _detections(sd, row, cfg, source, cloud_of):
    """(detections fed to the tracker, dropped by the filters or None, their reasons) in the det frame."""
    if source == "labels":
        f = (sd.dense or {}).get(_key(row["stamp"]))
        objs = [o for o in (f["objects"] if f else []) if o["label"] == "vehicle" and not o.get("ignore")]
        if not objs:
            return Detections(stamp=row["stamp"]), None, None
        return Detections(np.array([o["box"] for o in objs], float), np.array([o["velocity"] for o in objs], float),
                          np.full(len(objs), 0.99), np.full(len(objs), VEHICLE, np.int64), "det", row["stamp"]), None, None
    if source == "raw":
        raw = row.get("raw_detections")
        if raw is None:
            raise ValueError("%s: results.pkl has no raw_detections (run_sequence.py --save-raw)" % sd.name)
        if needs_clouds(cfg):
            why = filter_reasons(raw, cloud_of(row["stamp"]) if cloud_of else None, cfg.filters)
        else:
            why = filter_reasons(raw, None, cfg.filters, measured=row["raw_measured"])
        keep = why == ""
        return raw.select(np.nonzero(keep)[0]), raw.select(np.nonzero(~keep)[0]), list(why[~keep])
    return row["detections"], None, None


_LEARNED = {}


def _learned_predictor(path):
    """One LearnedPredictor per model file and process (loading costs more than a scene)."""
    if path not in _LEARNED:
        from .prediction.learned import LearnedPredictor
        _LEARNED[path] = LearnedPredictor(path)
    return _LEARNED[path]


def run_scene(sd, cfg=None, source="results", viz=None, cloud_of=None, lidar_stride=2, logger=None):
    """Re-run filters, tracker and predictor on one scene and score it. ``viz``: a ClipVisualizer to
    draw every frame into; ``cloud_of(stamp)``: the saved detector input (N, 5, det frame) or None,
    for the lidar in ``viz`` and the snapshot, and for ``raw`` with changed measuring settings.
    ``logger``: a ``prediction.TrackLogger`` that records the tracks, robot poses and predictions
    (the tier-2 training data, tools/dog/train_predictor.py)."""
    cfg = cfg or EvalConfig()
    res = SceneResult(sd.name, sd.rows[0]["stamp"] if sd.rows else 0.0, merges=sd.merges)
    tracker = MultiObjectTracker(cfg.tracker)
    learned = _learned_predictor(cfg.learned_model) if cfg.predict and cfg.learned_model else None
    res.predictor = "learned" if learned is not None else "IMM"
    predictor = learned or (IMMPredictor(**cfg.predictor) if cfg.predict else None)
    if learned is not None:
        times = learned.fc.fut_dt * np.arange(1, learned.fc.fut_len + 1)
    elif predictor is not None:
        times = predictor.step * np.arange(1, int(round(predictor.horizon / predictor.step)) + 1)
    else:
        times = np.arange(1, 7) * 0.5
    ego = deque(maxlen=64)              # the robot's own history, a neighbour of every agent for tier 2
    mot = MotAccumulator(["vehicle"], cfg.match_dist)
    motion = MotionStateAccumulator(("vehicle",), cfg.match_dist, relevant_speed=1.0, relevant_range=cfg.near_range)
    good = [r for r in sd.rows if not r.get("pose_bad")]
    rob_t = np.array([r["stamp"] for r in good])
    rob_xy = np.array([np.asarray(r["T_world_det"])[:2, 3] for r in good]).reshape(-1, 2)

    def robot_at(ts):
        ts = np.atleast_1d(ts)
        if len(rob_t) == 0:
            return np.zeros((len(ts), 2))
        return np.c_[np.interp(ts, rob_t, rob_xy[:, 0]), np.interp(ts, rob_t, rob_xy[:, 1])]

    keys = [_key(r["stamp"]) for r in sd.rows if _key(r["stamp"]) in sd.keyframes and not r.get("pose_bad")]
    grace = cfg.unlabelled_grace if cfg.unlabelled_grace is not None else \
        cfg.tracker.class_params["vehicle"].max_age + 0.5
    snap_key = keys[len(keys) // 2] if keys else None
    kidx = -1
    seen_near, tracked_at, first_scored = {}, {}, {}
    fp_kf, fn_kf, alarm_kf, hazard_kf, match_kf = defaultdict(list), defaultdict(list), defaultdict(list), \
        defaultdict(list), {}
    cum = Counter()
    last_pose = None
    for row in sd.rows:
        stamp, T = float(row["stamp"]), np.asarray(row["T_world_det"], float)
        bad = bool(row.get("pose_bad"))
        res.frames += 1
        t1 = time.perf_counter()
        dets, dropped, why = _detections(sd, row, cfg, source, cloud_of)
        if bad:
            dets, dropped, why = Detections(stamp=stamp), None, None
        t2 = time.perf_counter()
        tracks = tracker.step(dets.transform(T, "world"), stamp)
        t3 = time.perf_counter()
        T_body = np.asarray(row.get("T_world_body", T), float)
        v = (T_body[:2, 3] - ego[-1][1:3]) / (stamp - ego[-1][0]) if ego and stamp > ego[-1][0] else np.zeros(2)
        ego.append(np.array([stamp, T_body[0, 3], T_body[1, 3], v[0], v[1], 1.0]))
        if learned is not None:
            preds = learned(tracker, np.asarray(ego))
        else:
            preds = predictor(tracker) if predictor else []
        t4 = time.perf_counter()
        if logger is not None:
            logger.add(stamp, tracks, T_body, preds)
        res.timings["filter_ms"].append(1e3 * (t2 - t1))
        res.timings["track_ms"].append(1e3 * (t3 - t2))
        res.timings["predict_ms"].append(1e3 * (t4 - t3))

        kf = None if bad else sd.keyframes.get(_key(stamp))
        T_dw = inv_T(T)
        reported = dets.scores >= score_threshold_array(dets.labels, cfg.tracker.high_score)
        det_tp, det_gt, trk_info, fde_info, items, ev, metrics = None, {}, {}, {}, [], None, None
        if kf is not None:
            kidx += 1
            res.keyframes += 1
            # ---- ground truth of this keyframe
            gts = []
            for o in kf["objects"]:
                if o["label"] != "vehicle":
                    continue
                b = np.asarray(o["box"], float)
                rng = float(np.hypot(*b[:2]))
                g = None if o.get("ignore") else sd.gt_id.get(o["track_id"])
                scored = g is not None and rng <= cfg.max_range and int(o.get("points", 1)) > 0
                gts.append(dict(o=o, box=b, rng=rng, g=g, scored=scored, w=_to_world(b, T), zone=zone_of(b[:2], cfg.sector, cfg.edge)))
            # a labelled vehicle the labels leave out here (between its stretches, or just after / before
            # them) is not scored either
            for g, trk in sd.gt_track.items():
                w = _unlabelled(trk, stamp, grace)
                if w is not None:
                    b = Detections(w[None], np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int64)).transform(
                        T_dw, "det").boxes[0]
                    gts.append(dict(o=dict(track_id=-g, velocity=np.zeros(2), points="-", unlabelled=True), box=b,
                                    rng=float(np.hypot(*b[:2])), g=None, scored=False, w=w,
                                    zone=zone_of(b[:2], cfg.sector, cfg.edge)))
            sc = [x for x in gts if x["scored"]]
            unscored = np.array([x["box"] for x in gts if not x["scored"]]).reshape(-1, 7)
            gt_boxes = np.array([x["box"] for x in sc]).reshape(-1, 7)
            # ---- detections at the reporting threshold
            veh = np.nonzero((dets.labels == VEHICLE) & (np.hypot(*dets.boxes[:, :2].T) <= cfg.max_range))[0]
            rep = veh[reported[veh]]
            tp, gt_m = match_frame(dict(boxes=dets.boxes[rep], scores=dets.scores[rep], labels=dets.labels[rep]),
                                   dict(boxes=gt_boxes, labels=np.full(len(sc), VEHICLE)), cfg.match_dist)
            det_tp = np.zeros(len(dets), bool)
            det_tp[rep[tp]] = True
            for j, i in enumerate(gt_m):
                if i >= 0:
                    det_gt[int(rep[i])] = sc[j]["g"]
            n_fp = 0
            for i, ok in zip(rep, tp):
                near = "near" if np.hypot(*dets.boxes[i, :2]) <= cfg.near_range else "far"
                if ok:
                    res.counts["det_tp_" + near] += 1
                elif not _on_ignore(dets.boxes[i, :2], unscored, cfg.match_dist):
                    res.counts["det_fp_" + near] += 1
                    n_fp += 1
                else:                            # on a box that is not scored: drawn without the FP outline
                    det_tp[i], det_gt[int(i)] = True, "not scored"
            for j, x in enumerate(sc):
                if gt_m[j] < 0:
                    res.counts["det_fn_" + ("near" if x["rng"] <= cfg.near_range else "far")] += 1
            res.det_frames.append((dict(boxes=dets.boxes[veh], scores=dets.scores[veh], labels=np.zeros(len(veh), int)),
                                   dict(boxes=np.r_[gt_boxes, unscored].reshape(-1, 7),
                                        labels=np.zeros(len(sc) + len(unscored), int),
                                        ignore=np.r_[np.zeros(len(sc), bool), np.ones(len(unscored), bool)])))
            # ---- tracks
            t_det = [T_dw[:3, :3] @ s.position + T_dw[:3, 3] for s in tracks]
            keep = []
            for s, p in zip(tracks, t_det):
                if s.label != VEHICLE or np.hypot(*p[:2]) > cfg.max_range:
                    continue
                near_scored = len(gt_boxes) and np.hypot(*(gt_boxes[:, :2] - p[:2]).T).min() <= cfg.match_dist
                if not near_scored and _on_ignore(p[:2], unscored, cfg.match_dist):
                    continue                     # on an ignore box / a hidden or out-of-range vehicle
                keep.append((s, p))
            ev = mot.update([x["g"] for x in sc], np.array([x["w"][:2] for x in sc]).reshape(-1, 2), np.zeros(len(sc)),
                            [s.track_id for s, _ in keep], np.array([s.position[:2] for s, _ in keep]).reshape(-1, 2),
                            np.zeros(len(keep)))
            matched = {g: (tid, d) for g, tid, d, _ in ev["matches"]}
            match_kf[kidx] = {g: tid for g, (tid, _) in matched.items()}
            by_g = {x["g"]: x for x in sc}
            by_id = {s.track_id: (s, p) for s, p in keep}
            for g, (tid, d) in matched.items():
                trk_info[tid] = {"eval": "TP", "gt_id": g, "center_err_m": "%.2f" % d}
            for x in sc:
                g = x["g"]
                first_scored.setdefault(g, kidx)
                if x["rng"] <= cfg.near_range:
                    seen_near.setdefault(g, (kidx, stamp))
                if g in matched and g in seen_near:
                    tracked_at.setdefault(g, stamp)
                res.counts["gt_zone|" + x["zone"]] += 1
            for s, p in keep:
                res.counts["trk_zone|" + zone_of(p[:2], cfg.sector, cfg.edge)] += 1
            # why tracks are false
            twin = {matched[x["g"]][0]: x for x in sc if x["g"] in matched}
            for tid in ev["fp"]:
                s, p = by_id[tid]
                trk_info[tid] = {"eval": "FP"}
                fp_kf[tid].append((kidx, stamp, float(np.hypot(*p[:2])), zone_of(p[:2], cfg.sector, cfg.edge)))
                d = np.hypot(*(gt_boxes[:, :2] - p[:2]).T) if len(gt_boxes) else np.zeros(0)
                if any(_on_ignore(p[:2], x["box"][None], 0.0) for x in twin.values()):
                    cause = FP_CAUSES[0]
                elif (d <= 2 * cfg.match_dist).any():
                    cause = FP_CAUSES[1]
                elif s.coasting:
                    cause = FP_CAUSES[2]
                else:
                    cause = FP_CAUSES[3]
                res.counts["fp_cause|" + cause] += 1
                res.counts["fp_zone|" + zone_of(p[:2], cfg.sector, cfg.edge)] += 1
            # why vehicles are missed
            rep_xy = dets.boxes[rep, :2]
            for g in ev["fn"]:
                x = by_g[g]
                det = bool(len(rep_xy)) and np.hypot(*(rep_xy - x["box"][:2]).T).min() <= cfg.match_dist
                res.counts["fn_cause|" + FN_CAUSES[0 if det else 1]] += 1
                res.counts["fn_range|" + _bin(x["rng"], RANGE_BINS)] += 1
                res.counts["fn_points|" + _bin(int(x["o"].get("points", 0)), POINT_BINS)] += 1
                res.counts["fn_zone|" + x["zone"]] += 1
                if x["rng"] <= cfg.near_range:
                    fn_kf[g].append((kidx, stamp, x["rng"], x["zone"]))
            for g, old, new in ev["idsw"]:
                x = by_g[g]
                res.issues.append(dict(kind="ID switch", stamp=stamp, duration=0.0, track=new, gt=g, range=x["rng"],
                                       zone=x["zone"], value=1.0, note="T%d -> T%d" % (old, new)))
            # ---- motion state (all labelled vehicles, like motion_eval.py)
            motion.update(stamp, [dict(key=(sd.name, x["g"], res.t0), label=VEHICLE, xy=x["w"][:2],
                                       moving=bool(x["o"].get("moving")), speed=float(np.hypot(*x["o"]["velocity"])),
                                       range=x["rng"]) for x in gts if x["g"] is not None], tracks)
            # ---- prediction
            pred_by = {p.track_id: p for p in preds}
            fdes = []
            for g, (tid, _) in matched.items():
                p, s = pred_by.get(tid), by_id[tid][0]
                fut = _at(sd.gt_track.get(g), stamp + times)
                if p is None or np.isnan(fut).any():
                    continue
                cur = by_g[g]["w"][:2]
                path = np.vstack([cur, fut])
                seg = np.diff(path, axis=0)
                seg = seg[np.hypot(*seg.T) > 0.3]
                turn = float(np.degrees(abs(wrap_angle(np.arctan2(seg[-1, 1], seg[-1, 0]) - np.arctan2(seg[0, 1], seg[0, 0]))))) \
                    if len(seg) >= 2 else 0.0
                sample = dict(scene=sd.name, t=stamp, gt=fut, modes=p.modes, probs=p.probs,
                              cv=s.position[:2] + np.outer(times, s.velocity[:2]),
                              moving=bool(np.hypot(*(fut[-1] - cur)) >= cfg.moving_disp), turning=turn > cfg.turn_deg,
                              speed=float(np.hypot(*seg.T).sum() / times[-1]) if len(seg) else 0.0,
                              range=by_g[g]["rng"], gt_id=g, track=tid)
                res.pred.append(sample)
                fde_k = np.hypot(*(p.modes[:, -1] - fut[-1]).T)
                f1 = float(fde_k[int(np.argmax(p.probs))])
                fde_info[tid] = (fut[-1], f1, float(fde_k.min()))
                fdes.append(f1)
                if sample["moving"] and f1 > cfg.bad_fde:
                    res.issues.append(dict(kind="bad prediction", stamp=stamp, duration=times[-1], track=tid, gt=g,
                                           range=by_g[g]["rng"], zone=by_g[g]["zone"], value=f1,
                                           note="FDE %.1f m at %.0f s, %s, %.1f m/s" % (
                                               f1, times[-1], "turning" if sample["turning"] else "straight",
                                               sample["speed"])))
            # ---- hazard: a moving vehicle that comes within hazard_radius, and the alarms
            h_times = stamp + np.r_[0.0, times[times <= cfg.hazard_horizon + 1e-6]]
            rob = robot_at(h_times)
            for x in gts:
                if x["g"] is None:
                    continue
                pos = _at(sd.gt_track.get(x["g"]), h_times)
                pos[0] = x["w"][:2]
                ok = ~np.isnan(pos[:, 0])
                dist = np.hypot(*(pos[ok] - rob[ok]).T)
                span = h_times[ok][-1] - stamp
                moving = bool(x["o"].get("moving")) or (span >= 1.0 and np.hypot(*(pos[ok][-1] - pos[0])) / span >= 0.5)
                if moving and dist.min() <= cfg.hazard_radius:
                    hazard_kf[x["g"]].append((kidx, stamp, float(h_times[ok][int(np.argmin(dist))]), float(dist.min())))
            for s in tracks:
                if s.label != VEHICLE or s.motion != "moving":
                    continue
                p = pred_by.get(s.track_id)
                path = np.vstack([s.position[:2], p.best if p is not None else
                                  s.position[:2] + np.outer(times, s.velocity[:2])])
                n = 1 + int((times <= cfg.hazard_horizon + 1e-6).sum())
                if np.hypot(*(path[:n] - rob[:n]).T).min() <= cfg.hazard_radius:
                    alarm_kf[s.track_id].append(kidx)
            # ---- per-keyframe numbers, drawing
            for k in ("tp", "fp", "fn", "idsw"):
                cum["trk_" + k] += len(ev["matches"]) if k == "tp" else len(ev[k])
            cum["gt"] += len(sc)
            metrics = dict(t=stamp, gt=len(sc), det_tp=int(tp.sum()), det_fn=int((gt_m < 0).sum()),
                           det_fp=n_fp, trk_tp=len(ev["matches"]), trk_fp=len(ev["fp"]),
                           trk_fn=len(ev["fn"]), idsw=len(ev["idsw"]),
                           mota_cum=1 - (cum["trk_fn"] + cum["trk_fp"] + cum["trk_idsw"]) / max(cum["gt"], 1),
                           fde_top_mean=float(np.mean(fdes)) if fdes else float("nan"), tracks=len(tracks),
                           detections=int(reported.sum()), alarms=sum(1 for v in alarm_kf.values() if v and v[-1] == kidx))
            status = {g: "tracked" for g in matched}
            status.update({g: "missed" for g in ev["fn"]})
            for x in gts:
                g = x["g"]
                trk = sd.gt_track.get(g)
                items.append(dict(box=x["w"], id=g if g is not None else -int(x["o"]["track_id"]), label=VEHICLE,
                                  status=status.get(g, "ignored") if x["scored"] else "ignored",
                                  vel=np.asarray(x["o"]["velocity"], float) @ T[:2, :2].T,
                                  past=_path(trk, stamp - 2.0, stamp), future=_path(trk, stamp, stamp + 3.0),
                                  meta={"gt_id": g if g is not None else "-", "label_track": x["o"]["track_id"],
                                        "ignore": bool(x["o"].get("ignore")), "points": x["o"].get("points", "-"),
                                        "unlabelled": bool(x["o"].get("unlabelled")),
                                        "moving": bool(x["o"].get("moving")), "range_m": "%.1f" % x["rng"],
                                        "zone": x["zone"], "scored": x["scored"],
                                        "status": status.get(g, "not scored") if x["scored"] else "not scored"}))
            if _key(stamp) == snap_key:
                cloud = cloud_of(stamp) if cloud_of else None
                cur = cloud[cloud[:, -1] <= 1e-6, :3][::3] if cloud is not None else np.zeros((0, 3))
                res.snapshot = dict(scene=sd.name, t=stamp, t0=res.t0, T_global_ego=T, gt=items, tracks=tracks,
                                    preds=preds, metrics=metrics, lidar=cur @ T[:3, :3].T + T[:3, 3])
        if viz is not None:
            speed = 0.0 if last_pose is None or stamp <= last_pose[0] else \
                float(np.hypot(*(T[:2, 3] - last_pose[1]))) / (stamp - last_pose[0])
            _draw(viz, sd, row, T, dets, reported, det_tp, det_gt, dropped, why, tracks, trk_info, preds, fde_info,
                  items, kf is not None, ev, metrics, speed, cfg, cloud_of, lidar_stride)
        last_pose = (stamp, T[:2, 3].copy())

    _finish(res, sd, cfg, mot, motion, keys, seen_near, tracked_at, fp_kf, fn_kf, alarm_kf, hazard_kf, match_kf)
    return res


def _finish(res, sd, cfg, mot, motion, keys, seen_near, tracked_at, fp_kf, fn_kf, alarm_kf, hazard_kf, match_kf):
    period = float(np.median(np.diff(sorted(keys)))) if len(keys) > 1 else 0.5
    res.keyframe_s = res.keyframes * period
    res.mot = mot.summary()["vehicle"]
    # a vehicle that comes into near_range after the scene started: time to its first match
    for g, (k, t) in seen_near.items():
        if k == 0:
            continue
        if g in tracked_at:
            res.latencies.append(tracked_at[g] - t)
        else:
            res.never_tracked += 1
    for tid, rows in fp_kf.items():
        for a, b in _runs([r[0] for r in rows]):
            res.fp_episodes += 1
            if b > a:
                res.fp_episodes_long += 1
                seg = [r for r in rows if a <= r[0] <= b]
                res.issues.append(dict(kind="false track", stamp=seg[0][1], duration=(b - a + 1) * period, track=tid,
                                       gt=None, range=float(np.median([r[2] for r in seg])), zone=seg[0][3],
                                       value=(b - a + 1) * period, note=""))
    for g, rows in fn_kf.items():
        for a, b in _runs([r[0] for r in rows]):
            if b > a:
                seg = [r for r in rows if a <= r[0] <= b]
                res.issues.append(dict(kind="lost vehicle", stamp=seg[0][1], duration=(b - a + 1) * period, track=None,
                                       gt=g, range=float(np.median([r[2] for r in seg])), zone=seg[0][3],
                                       value=(b - a + 1) * period, note="within %.0f m, no track" % cfg.near_range))
    # motion state
    res.motion_rows = {k: list(v) for k, v in motion.rows.items()}
    res.motion_s = res.keyframe_s
    motion.recording_s = res.motion_s
    for d in motion.summary()["details"]:
        if d["kind"] == "missed" and not d.get("relevant"):
            continue
        _, g, _ = d["key"]
        res.issues.append(dict(kind="missed moving vehicle" if d["kind"] == "missed" else "false moving",
                               stamp=d["t"], duration=d["duration"], track=None, gt=g, range=d["range"],
                               zone="", value=d["duration"],
                               note="label %.1f m/s" % d["label_speed"] if d["kind"] == "missed" else
                               "parked vehicle reported moving"))
    # hazard events and alarms
    alarm_set = {tid: set(v) for tid, v in alarm_kf.items()}
    hazard_at = defaultdict(set)
    for g, rows in hazard_kf.items():
        for r in rows:
            hazard_at[r[0]].add(g)
        for a, b in _runs([r[0] for r in rows]):
            seg = [r for r in rows if a <= r[0] <= b]
            closest = min(seg, key=lambda r: r[3])
            hits = [r for r in seg if match_kf.get(r[0], {}).get(g) in alarm_set
                    and r[0] in alarm_set[match_kf[r[0]][g]]]
            lead = closest[2] - hits[0][1] if hits else float("nan")
            res.hazard.append(dict(gt=g, stamp=seg[0][1], reported=bool(hits), lead=lead, closest=closest[3]))
            if not hits:
                res.issues.append(dict(kind="hazard not reported", stamp=seg[0][1], duration=(b - a + 1) * period,
                                       track=None, gt=g, range=closest[3], zone="", value=1.0,
                                       note="moving vehicle within %.0f m, no alarm" % cfg.hazard_radius))
    for tid, ks in alarm_kf.items():
        for a, b in _runs(sorted(set(ks))):
            real = any(g in hazard_at.get(k, ()) and match_kf.get(k, {}).get(g) == tid for k in range(a, b + 1)
                       for g in hazard_at.get(k, ()))
            res.false_alarms += not real
    for it in res.issues:
        it["scene"], it["t"] = sd.name, it["stamp"] - res.t0


def _draw(viz, sd, row, T, dets, reported, det_tp, det_gt, dropped, why, tracks, trk_info, preds, fde_info, items,
          keyframe, ev, metrics, speed, cfg, cloud_of, lidar_stride):
    stamp = float(row["stamp"])
    viz.transforms(stamp, [("map", "ego", T)])
    cloud = cloud_of(stamp) if cloud_of else None
    if cloud is not None:
        cur = cloud[cloud[:, -1] <= 1e-6][::lidar_stride]
        viz.lidar(stamp, cur[:, :3] @ T[:3, :3].T + T[:3, 3], cur[:, 3])
    viz.ego(stamp, T, speed, size=(0.5, 0.9, 0.6), offset=0.0)
    viz.sector(stamp, cfg.sector, cfg.edge, cfg.max_range)
    if not keyframe and sd.dense is not None:
        f = sd.dense.get(_key(stamp))
        for o in (f["objects"] if f else []):
            if o["label"] == "vehicle" and not o.get("ignore"):
                kid = sd.dense_id.get(o["track_id"], o["track_id"])
                items.append(dict(box=_to_world(o["box"], T), id=sd.gt_id.get(kid, -int(kid)),
                                  label=VEHICLE, status="interp"))
    if keyframe or sd.dense is not None:
        viz.ground_truth(stamp, items, keyframe)
    viz.detections(stamp, dets.transform(T, "world"), reported, det_tp, det_gt)
    if dropped is not None:
        viz.filtered(stamp, dropped.transform(T, "world"), why)
    viz.tracks(stamp, tracks, trk_info)
    z_of = {}
    for s in tracks:
        z_of[s.track_id] = s.position[2] - s.size[2] / 2 + 0.15
        z_of[("xy", s.track_id)] = s.position[:2]
    viz.predictions(stamp, preds, z_of, fde_info if keyframe else None)
    if keyframe:
        gt_pos = {it["id"]: np.asarray(it["box"][:3]) for it in items}
        viz.events(stamp, ev, gt_pos, {s.track_id: s.position for s in tracks})
        viz.metrics(stamp, metrics)


# ------------------------------------------------------------------ all scenes
def _pad(samples, kind):
    if kind == "cv":
        return np.stack([s["cv"][None] for s in samples]), np.ones((len(samples), 1))
    K = max(len(s["probs"]) for s in samples)
    M = np.zeros((len(samples), K) + samples[0]["gt"].shape)
    P = np.zeros((len(samples), K))
    for b, s in enumerate(samples):
        k = len(s["probs"])
        M[b, :k], P[b, :k] = s["modes"], s["probs"]
        M[b, k:] = s["modes"][int(np.argmax(s["probs"]))]
    return M, P


def prediction_metrics(samples, step=0.5, name="IMM"):
    """{predictor: {horizon: displacement metrics}} for the online predictor (all its modes; ``name``:
    "IMM" or "learned") and constant velocity."""
    if not samples:
        return {}
    gt = np.stack([s["gt"] for s in samples])
    out = {}
    for kind, name in (("imm", name), ("cv", "CV")):
        M, P = _pad(samples, kind)
        out[name] = {}
        for h in HORIZONS:
            n = int(round(h / step))
            if n <= gt.shape[1]:
                out[name][h] = displacement_metrics(M[:, :, :n], P, gt[:, :n])
    return out


def prediction_subsets(samples):
    mov = [s for s in samples if s["moving"]]
    subsets = {"all matched": samples, "moving": mov,
               "moving, straight": [s for s in mov if not s["turning"]],
               "moving, turning": [s for s in mov if s["turning"]],
               "moving, < 3 m/s": [s for s in mov if s["speed"] < 3],
               "moving, 3-6 m/s": [s for s in mov if 3 <= s["speed"] < 6],
               "moving, >= 6 m/s": [s for s in mov if s["speed"] >= 6]}
    return {k: v for k, v in subsets.items() if v}


def _q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def summarize(results, step=0.5, tables=True):
    """Metrics over scenes (a list of SceneResult; repeats allowed, as in a bootstrap draw). Flat keys
    for the sweep table, and with ``tables`` the breakdowns for the report."""
    m = {}
    minutes = sum(r.keyframe_s for r in results) / 60.0
    c = Counter()
    for r in results:
        c.update(r.counts)
    for near in ("near", "far"):
        tp, fp, fn = c["det_tp_" + near], c["det_fp_" + near], c["det_fn_" + near]
        m["det_recall_" + near] = tp / (tp + fn) if tp + fn else float("nan")
        m["det_precision_" + near] = tp / (tp + fp) if tp + fp else float("nan")
        m["det_f1_" + near] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    kf = sum(r.keyframes for r in results)
    m["det_fp_per_frame"] = (c["det_fp_near"] + c["det_fp_far"]) / kf if kf else float("nan")
    k = Counter()
    for r in results:
        for key in ("num_gt", "num_trk", "TP", "FP", "FN", "IDSW", "FRAG", "MT", "ML", "num_gt_tracks", "IDTP"):
            k[key] += r.mot.get(key, 0) or 0
        k["dist"] += (r.mot.get("MOTP") or 0.0) * r.mot.get("TP", 0) if r.mot.get("TP") else 0.0
    m["MOTA"] = 1 - (k["FN"] + k["FP"] + k["IDSW"]) / k["num_gt"] if k["num_gt"] else float("nan")
    m["MOTP"] = k["dist"] / k["TP"] if k["TP"] else float("nan")
    m["IDF1"] = 2 * k["IDTP"] / (k["num_gt"] + k["num_trk"]) if k["num_gt"] + k["num_trk"] else float("nan")
    m["trk_recall"] = k["TP"] / k["num_gt"] if k["num_gt"] else float("nan")
    m["trk_precision"] = k["TP"] / k["num_trk"] if k["num_trk"] else float("nan")
    m["IDSW_per_min"] = k["IDSW"] / minutes if minutes else float("nan")
    m["false_tracks_per_min"] = sum(r.fp_episodes for r in results) / minutes if minutes else float("nan")
    lat = np.concatenate([np.asarray(r.latencies, float) for r in results]) if results else np.zeros(0)
    m["confirm_p50"], m["confirm_p90"] = _q(lat, 50), _q(lat, 90)
    m["never_tracked"] = sum(r.never_tracked for r in results)
    # motion state: the scenes' label stretches together (keys made unique per draw)
    acc = MotionStateAccumulator(("vehicle",), relevant_speed=1.0)
    for i, r in enumerate(results):
        for key, rows in r.motion_rows.items():
            acc.rows[(i,) + tuple(key)] = list(rows)
        acc.recording_s += r.motion_s
    ms = acc.summary()
    for key in ("moving_recall", "false_moving", "false_episodes_per_h", "relevant_missed", "relevant_segments",
                "relevant_latency_median", "relevant_latency_p90", "static_flagged", "static_tracks"):
        m["motion_" + key] = ms[key]
    samples = [s for r in results for s in r.pred]
    mov = [s for s in samples if s["moving"]]
    online = results[0].predictor if results else "IMM"
    pm = prediction_metrics(mov, step, online)
    for name, key in ((online, "pred"), ("CV", "CV")):     # pred_*: whatever predicts online
        for h in HORIZONS:
            r3 = pm.get(name, {}).get(h)
            m["%s_FDE%d_moving" % (key, h)] = r3["FDE1"] if r3 else float("nan")
        r3 = pm.get(name, {}).get(HORIZONS[-1])
        m["%s_minFDE3_moving" % key] = r3["minFDE"] if r3 else float("nan")
        m["%s_MR3_moving" % key] = r3["MR"] if r3 else float("nan")
    m["pred_samples_moving"] = len(mov)
    hz = [h for r in results for h in r.hazard]
    m["hazard_events"] = len(hz)
    m["hazard_reported"] = sum(h["reported"] for h in hz)
    m["hazard_lead_median"] = _q([h["lead"] for h in hz if h["reported"]], 50)
    hours = minutes / 60.0
    m["false_alarms_per_h"] = sum(r.false_alarms for r in results) / hours if hours else float("nan")
    timing = {key: np.concatenate([np.asarray(r.timings.get(key, []), float) for r in results] or [np.zeros(0)])
              for key in ("filter_ms", "track_ms", "predict_ms")}
    for key, v in timing.items():
        m[key] = _q(v, 50)
    if not tables:
        return m
    det = evaluate_detections([p for r in results for p, _ in r.det_frames], [g for r in results for _, g in r.det_frames],
                              ["vehicle"], dist_thresholds=(2.0, 4.0), range_bins=((0, 20), (20, 40)))["vehicle"]
    m["AP@2.0"], m["AP@4.0"] = det.get("AP@2.0", float("nan")), det.get("AP@4.0", float("nan"))
    t = dict(counts=c, mot=dict(k), minutes=minutes, keyframes=kf, frames=sum(r.frames for r in results),
             merges=sum(r.merges for r in results), detection=det, motion=ms,
             prediction={name: prediction_metrics(v, step, online) for name, v in prediction_subsets(samples).items()},
             latency_n=len(lat), hazard=hz, timing={key: (_q(v, 50), _q(v, 90)) for key, v in timing.items()})
    m["tables"] = t
    return m


def bootstrap(results_by_config, n=500, seed=0, step=0.5):
    """Scene bootstrap, the same draws for every configuration (paired): {config: {metric: (n,) array}}.
    ``results_by_config``: {config name: [SceneResult] in the same scene order}."""
    names = list(results_by_config)
    S = len(results_by_config[names[0]])
    draws = np.random.default_rng(seed).integers(0, S, (n, S))
    out = {name: defaultdict(list) for name in names}
    for d in draws:
        for name in names:
            rs = results_by_config[name]
            for key, v in summarize([rs[i] for i in d], step, tables=False).items():
                out[name][key].append(v)
    return {name: {k: np.asarray(v, float) for k, v in vals.items()} for name, vals in out.items()}


# Main metrics of docs/PIPELINE_TUNING.md, and which way is better
MAIN_METRICS = (("det_recall_near", +1), ("det_precision_near", +1), ("det_f1_near", +1), ("det_fp_per_frame", -1),
                ("MOTA", +1),
                ("IDF1", +1), ("IDSW_per_min", -1), ("false_tracks_per_min", -1), ("confirm_p90", -1),
                ("motion_relevant_missed", -1), ("motion_false_episodes_per_h", -1), ("pred_FDE3_moving", -1),
                ("pred_MR3_moving", -1))
