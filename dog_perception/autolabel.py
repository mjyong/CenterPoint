"""Offboard auto-labelling: teacher detections -> tracks -> refined labels.

The teacher (CenterPoint-Voxel with flip TTA, nuScenes weights at first,
later your own fine-tuned voxel model) runs frame by frame; then, with the
whole recording available, every track is refined *non-causally*:

* class        : score-weighted vote over the track (kills class flicker);
                 a vehicle's subclass (car / truck / bus) is voted the same way
                 among the observations that carry one
* size, z      : score-weighted median (rigid objects do not change size)
* center       : RTS smoothing + interpolation across short gaps
                 (recovers frames the teacher missed)
* static/moving: a frame is moving only when both the smoothed trajectory
                 (difference over +-0.5 s, > 0.4 m/s) and the teacher's
                 velocity head (> 0.15 m/s) say so. A static object seen
                 from a moving robot drifts (its visible side changes), so
                 the trajectory alone gives parked cars a few tenths of
                 m/s; the velocity head alone under-reads slow movers.
                 Then stops shorter than 0.5 s inside a motion count as
                 moving, and moving stretches shorter than 1 s count as
                 static unless they are most of the track: on 0921, 45 of
                 the 104 vehicle tracks with any moving frame had only such
                 bursts (a parked truck whose box drifts as the robot walks
                 past, the last frames of a track), which kept them from
                 being pinned to one position and gave them velocity labels
* static track : zero velocity; fixed position (score-weighted median) when
                 the boxes stay close (per-class spread)
* heading      : direction of motion when moving fast, else circular mean
                 of the box yaws, 180 deg flips resolved against the motion
                 direction (or the best box when the track never moves)
* velocity     : trajectory difference when moving, zero when static
* drop         : tracks that are too short or too low-scoring

Humans then only fix what is left. ``review`` flags tracks the teacher is not
sure of: mean score below 0.5 and fewer than 5 frames scored 0.6 or more (a
far car scores 0.3-0.4 most of the time, but a real one is seen clearly at
some point; a false one rarely is, several times).
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .detection.boxes import CLASSES, SUBCLASS_TO_CLASS, SUBCLASSES
from .geometry import wrap_angle
from .prediction.dataset import rts_smooth
from .tracking import MultiObjectTracker, TrackerConfig


@dataclass
class AutoLabelConfig:
    min_track_len: int = 5          # observations
    min_mean_score: float = 0.3
    max_gap: float = 0.5            # [s] gaps filled by interpolation
    moving_speed: float = 1.0       # [m/s] heading from motion above this
    static_speed: float = 0.4       # [m/s] trajectory speed below this: static frame
    teacher_speed: float = 0.15     # [m/s] teacher velocity-head speed below this: static frame
    # [m] a static track is pinned to one position only when 90% of its boxes stay this close to the median;
    # wider spread means the track drifted or switched objects, keep the smoothed path then
    max_static_spread: dict = field(default_factory=lambda: {"vehicle": 1.0, "pedestrian": 0.5, "cyclist": 1.5})
    velocity_window: float = 0.5    # [s] half-width of the trajectory difference
    min_moving_time: float = 1.0    # [s] shorter moving stretches are static, unless most of the track
    min_stop_time: float = 0.5      # [s] shorter stops inside a motion are moving
    # a track is flagged for review when its mean score is below review_score and fewer than
    # confident_frames of its detections score confident_score or more
    review_score: float = 0.5
    confident_score: float = 0.6
    confident_frames: int = 5


def _weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def associate_offline(frames, tracker_cfg=None):
    """frames: list of (stamp, Detections in world frame). Returns {track_id: list of obs}."""
    trk = MultiObjectTracker(tracker_cfg or TrackerConfig(classes=CLASSES))
    obs = defaultdict(list)
    for stamp, dets in frames:
        trk.step(dets, stamp)
        for tid, d in trk.last_matches.items():
            obs[tid].append(dict(t=stamp, box=dets.boxes[d].copy(), vel=dets.velocities[d].copy(),
                                 score=float(dets.scores[d]), label=int(dets.labels[d]),
                                 subclass=int(dets.subclasses[d])))
    return obs


def refine_track(obs, stamps, cfg):
    """obs: list of observations of one track. Returns list of per-frame labels or None."""
    if len(obs) < cfg.min_track_len:
        return None
    t = np.array([o["t"] for o in obs])
    boxes = np.stack([o["box"] for o in obs])
    scores = np.array([o["score"] for o in obs])
    if scores.mean() < cfg.min_mean_score:
        return None
    votes = np.zeros(len(CLASSES))
    for o in obs:
        votes[o["label"]] += o["score"]
    label = int(np.argmax(votes))
    sub_votes = np.zeros(len(SUBCLASSES))
    for o in obs:
        s = o.get("subclass", -1)
        if s >= 0 and SUBCLASS_TO_CLASS[SUBCLASSES[s]] == CLASSES[label]:
            sub_votes[s] += o["score"]
    subclass = int(np.argmax(sub_votes)) if sub_votes.any() else -1
    size = np.array([_weighted_median(boxes[:, k], scores) for k in (3, 4, 5)])
    z = _weighted_median(boxes[:, 2], scores)

    # frames covered by the track, with small gaps filled
    span = stamps[(stamps >= t[0] - 1e-6) & (stamps <= t[-1] + 1e-6)]
    gap_ok = np.array([np.min(np.abs(t - s)) <= cfg.max_gap for s in span])
    span = span[gap_ok]
    xy = rts_smooth(t, boxes[:, :2])
    sx = np.interp(span, t, xy[:, 0])
    sy = np.interp(span, t, xy[:, 1])
    v_traj = _window_velocity(span, sx, sy, cfg.velocity_window)
    v_teacher = _running_median(np.array([np.hypot(*o["vel"]) for o in obs]), 2)
    moving = (np.hypot(v_traj[:, 0], v_traj[:, 1]) > cfg.static_speed) & \
        (np.interp(span, t, v_teacher) > cfg.teacher_speed)
    moving = _clean_moving(span, moving, cfg)
    if not moving.any():                      # static object: one position for the whole track
        c = np.array([_weighted_median(boxes[:, 0], scores), _weighted_median(boxes[:, 1], scores)])
        if np.percentile(np.hypot(*(boxes[:, :2] - c).T), 90) <= cfg.max_static_spread.get(CLASSES[label], 1.0):
            sx, sy = np.full(len(span), c[0]), np.full(len(span), c[1])
    vel = np.where(moving[:, None], v_traj, 0.0)
    speed = np.hypot(vel[:, 0], vel[:, 1])

    # box yaw: resolve 180 deg flips against the motion direction, else the highest-scoring box
    fast = speed > cfg.moving_speed
    if fast.any():
        k = int(np.argmax(speed))
        ref = float(np.arctan2(vel[k, 1], vel[k, 0]))
    else:
        ref = boxes[np.argmax(scores), 6]
    yaws = boxes[:, 6].copy()
    flip = np.abs(wrap_angle(yaws - ref)) > np.pi / 2
    yaws[flip] += np.pi
    static_yaw = float(np.arctan2(np.sum(scores * np.sin(yaws)), np.sum(scores * np.cos(yaws))))

    confident = int(np.sum(scores >= cfg.confident_score))
    review = bool(scores.mean() < cfg.review_score and confident < cfg.confident_frames)
    out = []
    for k, s in enumerate(span):
        yaw = float(np.arctan2(vel[k, 1], vel[k, 0])) if fast[k] else static_yaw
        near = int(np.argmin(np.abs(t - s)))
        observed = abs(t[near] - s) <= 1e-6
        out.append(dict(t=float(s), label=label, subclass=subclass,
                        box=np.array([sx[k], sy[k], z, *size, wrap_angle(yaw)]),
                        velocity=vel[k].copy(), score=float(scores.mean()), moving=bool(moving[k]),
                        det_score=float(scores[near]) if observed else 0.0, confident_frames=confident,
                        review=review, interpolated=not observed))
    return out


def _runs(mask):
    """(start, end) index pairs of the True stretches of a bool array (end inclusive)."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1))


def _clean_moving(t, moving, cfg):
    """Per-frame moving flags without the bursts: stops shorter than ``min_stop_time`` between two
    moving stretches become moving, then moving stretches shorter than ``min_moving_time`` become
    static unless they cover at least half of the track."""
    moving = moving.copy()
    if len(t) < 2 or not moving.any():
        return moving
    period = float(np.median(np.diff(t)))
    for a, b in _runs(~moving):
        if a > 0 and b < len(t) - 1 and t[b] - t[a] + period < cfg.min_stop_time:
            moving[a:b + 1] = True
    span = t[-1] - t[0] + period
    for a, b in _runs(moving):
        dur = t[b] - t[a] + period
        if dur < cfg.min_moving_time and dur < 0.5 * span:
            moving[a:b + 1] = False
    return moving


def _window_velocity(t, x, y, half):
    """Velocity from the position difference over [t - half, t + half] (clipped to the track)."""
    if len(t) < 2:
        return np.zeros((len(t), 2))
    lo = np.clip(t - half, t[0], t[-1])
    hi = np.clip(t + half, t[0], t[-1])
    dt = np.maximum(hi - lo, 1e-3)
    return np.stack([(np.interp(hi, t, x) - np.interp(lo, t, x)) / dt,
                     (np.interp(hi, t, y) - np.interp(lo, t, y)) / dt], 1)


def _running_median(x, k):
    return np.array([np.median(x[max(0, i - k):i + k + 1]) for i in range(len(x))])


def auto_label(frames, cfg=None, tracker_cfg=None):
    """frames: list of (stamp, Detections world). Returns {stamp: list of label dicts (world)}."""
    cfg = cfg or AutoLabelConfig()
    stamps = np.array([s for s, _ in frames])
    labels = defaultdict(list)
    for tid, obs in associate_offline(frames, tracker_cfg).items():
        refined = refine_track(obs, stamps, cfg)
        if refined is None:
            continue
        for r in refined:
            r["track_id"] = tid
            labels[round(r["t"], 6)].append(r)
    return {round(float(s), 6): labels.get(round(float(s), 6), []) for s in stamps}
