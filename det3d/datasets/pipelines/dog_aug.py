"""Training augmentation for the robot-dog data: pose errors between the stacked sweeps.

Online the sweeps are stacked with the INS poses, which on the kygi robot run ahead by up to
0.45 m while braking and are off by a few degrees after fast turns (docs/BAG_SPLIT.md 3.5); the
training frames are stacked with refined poses. ``SweepPoseJitter`` moves the older sweeps (dt > 0)
as a velocity / yaw-rate error would: sweep k of age dt_k is turned by w * dt_k about the
network-frame z axis and shifted by v * dt_k (v, w drawn per frame). The current sweep (dt = 0)
and the boxes stay as they are, so the labels remain right while the input shows the ghosts that
bad poses cause. Frames whose own pose is untrusted are left out of the datasets instead
(tools/dog/export_dataset.py): their boxes would not sit on their points.
"""
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class SweepPoseJitter(object):
    def __init__(self, prob=0.3, sigma_v=0.5, max_v=1.5, sigma_w_deg=3.0, max_w_deg=10.0, seed=None, **kwargs):
        self.prob, self.sigma_v, self.max_v = prob, sigma_v, max_v
        self.sigma_w, self.max_w = np.radians(sigma_w_deg), np.radians(max_w_deg)
        # without a seed, numpy's global generator: PyTorch seeds it differently in every DataLoader
        # worker, while a Generator made here would be copied into every worker with the same state
        # (all workers drawing the same jitter, epoch after epoch)
        self.rng = np.random.default_rng(seed) if seed is not None else np.random

    def __call__(self, res, info):
        if res.get("mode", "train") != "train" or self.rng.random() >= self.prob:
            return res, info
        pts = res["lidar"]["points"]
        if pts is None or len(pts) == 0 or pts.shape[1] < 5:
            return res, info
        v = self.rng.normal(0.0, self.sigma_v, 2)
        n = np.linalg.norm(v)
        if n > self.max_v:
            v *= self.max_v / n
        w = float(np.clip(self.rng.normal(0.0, self.sigma_w), -self.max_w, self.max_w))
        dt = pts[:, 4]
        old = dt > 1e-6
        if old.any():
            a = w * dt[old]
            c, s = np.cos(a), np.sin(a)
            x, y = pts[old, 0].copy(), pts[old, 1].copy()
            pts[old, 0] = c * x - s * y + v[0] * dt[old]
            pts[old, 1] = s * x + c * y + v[1] * dt[old]
        res["lidar"]["points"] = pts
        if "combined" in res["lidar"]:
            res["lidar"]["combined"] = pts
        return res, info
