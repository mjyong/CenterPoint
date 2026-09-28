"""Ground-truth driven detector with realistic errors.

Lets the tracker / predictor be developed and regression-tested without
network weights (and in the simulator). Error model: detection probability
grows with the number of lidar points on the object, position / size / yaw /
velocity noise, occasional 180 deg yaw flips and false positives.
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import wrap_angle
from .boxes import CLASSES, Detections


@dataclass
class OracleNoise:
    pos_std: float = 0.10
    size_std: float = 0.05
    yaw_std: float = 0.08
    vel_std: float = 0.25
    yaw_flip_prob: float = 0.05
    points_for_half_recall: float = 8.0   # detection prob = 0.5 at this many points
    false_positive_rate: float = 0.3      # per frame
    max_range: float = 40.0


class OracleDetector:
    def __init__(self, noise=None, seed=0):
        self.noise = noise or OracleNoise()
        self.rng = np.random.default_rng(seed)

    def __call__(self, gt_objects, T_det_world, stamp=None, use_points=True):
        """gt_objects: list of dicts (sim.ground_truth format, world frame)."""
        n, rng = self.noise, self.rng
        boxes, vels, scores, labels = [], [], [], []
        R = T_det_world[:3, :3]
        dyaw = np.arctan2(R[1, 0], R[0, 0])
        for g in gt_objects:
            if g["label"] not in CLASSES:
                continue
            c = R @ g["center"] + T_det_world[:3, 3]
            if np.hypot(c[0], c[1]) > n.max_range:
                continue
            if use_points:
                pts = g.get("num_points", 0)
                p_det = pts / (pts + n.points_for_half_recall)
            else:
                p_det = 0.9
            if rng.random() > p_det:
                continue
            yaw = g["yaw"] + dyaw + rng.normal(0, n.yaw_std)
            if rng.random() < n.yaw_flip_prob:
                yaw += np.pi
            size = g["size"] * np.exp(rng.normal(0, n.size_std, 3))
            boxes.append([*(c + rng.normal(0, n.pos_std, 3)), *size, wrap_angle(yaw)])
            vels.append(R[:2, :2] @ g["velocity"] + rng.normal(0, n.vel_std, 2))
            scores.append(float(np.clip(0.3 + 0.7 * p_det + rng.normal(0, 0.1), 0.11, 0.99)))
            labels.append(CLASSES.index(g["label"]))
        for _ in range(rng.poisson(n.false_positive_rate)):
            r, a = rng.uniform(3, n.max_range), rng.uniform(-np.pi, np.pi)
            lab = int(rng.integers(len(CLASSES)))
            boxes.append([r * np.cos(a), r * np.sin(a), 0.8, 0.6, 0.6, 1.7, rng.uniform(-np.pi, np.pi)])
            vels.append(rng.normal(0, 0.5, 2))
            scores.append(float(rng.uniform(0.1, 0.45)))
            labels.append(lab)
        if not boxes:
            return Detections(frame="det", stamp=stamp)
        return Detections(np.asarray(boxes), np.asarray(vels), np.asarray(scores),
                          np.asarray(labels, dtype=np.int64), "det", stamp)
