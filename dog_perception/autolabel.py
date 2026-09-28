"""Offboard auto-labelling: teacher detections -> tracks -> refined labels.

The teacher (CenterPoint-Voxel with flip TTA, nuScenes weights at first,
later your own fine-tuned voxel model) runs frame by frame; then, with the
whole recording available, every track is refined *non-causally*:

* class        : score-weighted vote over the track (kills class flicker)
* size, z      : score-weighted median (rigid objects do not change size)
* center       : RTS smoothing + interpolation across short gaps
                 (recovers frames the teacher missed)
* heading      : smoothed velocity direction when moving, circular mean of
                 the (180 deg disambiguated) box yaws otherwise
* velocity     : derivative of the smoothed trajectory
* drop         : tracks that are too short or too low-scoring

Humans then only fix what is left (``review`` flags low-confidence tracks).
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .detection.boxes import CLASSES
from .geometry import wrap_angle
from .prediction.dataset import rts_smooth
from .tracking import MultiObjectTracker, TrackerConfig


@dataclass
class AutoLabelConfig:
    min_track_len: int = 5          # observations
    min_mean_score: float = 0.3
    max_gap: float = 0.5            # [s] gaps filled by interpolation
    moving_speed: float = 1.0       # [m/s] heading from motion above this
    review_score: float = 0.5       # tracks below this mean score are flagged for review


def _weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def associate_offline(frames, tracker_cfg=None):
    """frames: list of (stamp, Detections in world frame). Returns {track_id: list of obs}."""
    trk = MultiObjectTracker(tracker_cfg or TrackerConfig())
    obs = defaultdict(list)
    for stamp, dets in frames:
        trk.step(dets, stamp)
        for tid, d in trk.last_matches.items():
            obs[tid].append(dict(t=stamp, box=dets.boxes[d].copy(), vel=dets.velocities[d].copy(),
                                 score=float(dets.scores[d]), label=int(dets.labels[d])))
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
    size = np.array([_weighted_median(boxes[:, k], scores) for k in (3, 4, 5)])
    z = _weighted_median(boxes[:, 2], scores)

    # frames covered by the track, with small gaps filled
    span = stamps[(stamps >= t[0] - 1e-6) & (stamps <= t[-1] + 1e-6)]
    gap_ok = np.array([np.min(np.abs(t - s)) <= cfg.max_gap for s in span])
    span = span[gap_ok]
    xy = rts_smooth(t, boxes[:, :2])
    sx = np.interp(span, t, xy[:, 0])
    sy = np.interp(span, t, xy[:, 1])
    if len(span) > 1:
        vx, vy = np.gradient(sx, span), np.gradient(sy, span)
    else:
        vx, vy = np.array([obs[0]["vel"][0]]), np.array([obs[0]["vel"][1]])

    # box yaw: resolve 180 deg flips against the highest-scoring observation
    ref = boxes[np.argmax(scores), 6]
    yaws = boxes[:, 6].copy()
    flip = np.abs(wrap_angle(yaws - ref)) > np.pi / 2
    yaws[flip] += np.pi
    static_yaw = float(np.arctan2(np.sum(scores * np.sin(yaws)), np.sum(scores * np.cos(yaws))))

    out = []
    for k, s in enumerate(span):
        speed = np.hypot(vx[k], vy[k])
        if speed > cfg.moving_speed:
            yaw = float(np.arctan2(vy[k], vx[k]))
        else:
            yaw = static_yaw
        out.append(dict(t=float(s), label=label, box=np.array([sx[k], sy[k], z, *size, wrap_angle(yaw)]),
                        velocity=np.array([vx[k], vy[k]]), score=float(scores.mean()),
                        review=bool(scores.mean() < cfg.review_score), interpolated=bool(np.min(np.abs(t - s)) > 1e-6)))
    return out


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
