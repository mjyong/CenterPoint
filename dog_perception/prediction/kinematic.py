"""Tier 1: IMM rollout (CV / CT), 3 s horizon at 0.5 s steps.

Each IMM mode becomes one predicted mode (``straight`` vs ``keeps turning``)
with the current mode probability; modes whose trajectories coincide (the
pedestrian CV-low / CV-high pair) are merged by moment matching, which turns
the high-noise CV into the "uncertainty inflation" around a single path.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    track_id: int
    label: int
    stamp: float
    times: np.ndarray                # (T,) seconds ahead of stamp
    modes: np.ndarray                # (K, T, 2) xy
    probs: np.ndarray                # (K,)
    covs: np.ndarray = None          # (K, T, 2, 2) or None
    source: str = "imm"
    frame: str = "world"

    @property
    def best(self):
        return self.modes[int(np.argmax(self.probs))]

    def transform(self, T_new_old, frame):
        R = T_new_old[:2, :2]
        modes = self.modes @ R.T + T_new_old[:2, 3]
        covs = None if self.covs is None else R @ self.covs @ R.T
        return Prediction(self.track_id, self.label, self.stamp, self.times, modes, self.probs.copy(),
                          covs, self.source, frame)


def merge_modes(means, covs, probs, merge_dist):
    """Greedy moment-matching merge of modes closer than ``merge_dist`` at every step."""
    order = np.argsort(-probs)
    groups = []
    for j in order:
        for g in groups:
            if np.max(np.linalg.norm(means[j] - means[g[0]], axis=-1)) < merge_dist:
                g.append(j)
                break
        else:
            groups.append([j])
    out_m, out_c, out_p = [], [], []
    for g in groups:
        w = probs[g] / probs[g].sum()
        m = np.einsum("j,jtk->tk", w, means[g])
        d = means[g] - m
        c = np.einsum("j,jtkl->tkl", w, covs[g] + d[..., :, None] * d[..., None, :])
        out_m.append(m)
        out_c.append(c)
        out_p.append(probs[g].sum())
    return np.stack(out_m), np.stack(out_c), np.asarray(out_p)


class IMMPredictor:
    def __init__(self, horizon=3.0, step=0.5, merge_dist=0.25):
        self.horizon, self.step, self.merge_dist = horizon, step, merge_dist

    def predict_imm(self, imm, track_id, label, stamp):
        r = imm.rollout(self.horizon, self.step)
        means = r["mode_means"][:, :, :2]
        covs = r["mode_covs"][:, :, :2, :2]
        means, covs, probs = merge_modes(means, covs, r["mode_probs"], self.merge_dist)
        return Prediction(track_id, label, stamp, r["times"], means, probs, covs, "imm")

    def __call__(self, tracker):
        return [self.predict_imm(t.imm, t.id, t.label, t.stamp) for t in tracker.tracks if t.confirmed]


def constant_velocity(position, velocity, times):
    """(T, 2) baseline used in evaluation."""
    return position[None, :2] + velocity[None, :2] * np.asarray(times)[:, None]
