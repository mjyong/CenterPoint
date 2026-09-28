"""Detection containers, class mapping and box-convention conversions.

Standard box used everywhere outside the network: ``[x, y, z, l, w, h, yaw]``
(geometric center, l along the heading, yaw CCW from +x).

det3d/CenterPoint nuScenes boxes are ``[x, y, z, w, l, h, vx, vy, r]`` with
``r = -yaw - pi/2`` (see ``nusc_common._fill_trainval_infos``).
"""
from dataclasses import dataclass, field

import numpy as np

from ..geometry import wrap_angle

CLASSES = ("vehicle", "pedestrian", "cyclist")

# nuScenes detection classes -> the three classes the dog cares about.
# barrier / traffic_cone are dropped (static, handled by the occupancy map).
NUSC_TO_DOG = {
    "car": "vehicle",
    "truck": "vehicle",
    "construction_vehicle": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "motorcycle": "cyclist",
    "bicycle": "cyclist",
    "pedestrian": "pedestrian",
}


def class_index(name):
    return CLASSES.index(name) if name in CLASSES else -1


@dataclass
class Detections:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))        # (N, 7) standard
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (N, 2) over ground
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    frame: str = "det"
    stamp: float = None

    def __len__(self):
        return len(self.scores)

    @property
    def names(self):
        return [CLASSES[i] for i in self.labels]

    def select(self, mask):
        return Detections(self.boxes[mask], self.velocities[mask], self.scores[mask],
                          self.labels[mask], self.frame, self.stamp)

    def transform(self, T_new_old, frame):
        """Move into another gravity-aligned frame (yaw-only rotation + translation)."""
        R = T_new_old[:3, :3]
        dyaw = np.arctan2(R[1, 0], R[0, 0])
        boxes = self.boxes.copy()
        boxes[:, :3] = self.boxes[:, :3] @ R.T + T_new_old[:3, 3]
        boxes[:, 6] = wrap_angle(self.boxes[:, 6] + dyaw)
        vel = self.velocities @ R[:2, :2].T
        return Detections(boxes, vel, self.scores.copy(), self.labels.copy(), frame, self.stamp)

    @staticmethod
    def concat(dets):
        dets = [d for d in dets if d is not None]
        if not dets:
            return Detections()
        return Detections(
            np.concatenate([d.boxes for d in dets]), np.concatenate([d.velocities for d in dets]),
            np.concatenate([d.scores for d in dets]), np.concatenate([d.labels for d in dets]),
            dets[0].frame, dets[0].stamp)


def det3d_to_standard(box9):
    """(N, 9) det3d nuScenes-style -> (standard (N, 7), velocity (N, 2))."""
    box9 = np.asarray(box9, dtype=np.float64).reshape(-1, 9)
    std = np.stack([box9[:, 0], box9[:, 1], box9[:, 2], box9[:, 4], box9[:, 3], box9[:, 5],
                    wrap_angle(-box9[:, 8] - np.pi / 2)], axis=1)
    return std, box9[:, 6:8].copy()


def standard_to_det3d(std, vel=None):
    std = np.asarray(std, dtype=np.float64).reshape(-1, 7)
    vel = np.zeros((len(std), 2)) if vel is None else np.asarray(vel).reshape(-1, 2)
    return np.stack([std[:, 0], std[:, 1], std[:, 2], std[:, 4], std[:, 3], std[:, 5],
                     vel[:, 0], vel[:, 1], wrap_angle(-std[:, 6] - np.pi / 2)], axis=1)


def circle_nms(xy, scores, radius, max_keep=None):
    """Greedy center-distance NMS. ``radius`` in meters. Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    xy = xy[order]
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    r2 = radius * radius
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        if max_keep is not None and len(keep) >= max_keep:
            break
        d2 = np.sum((xy[i + 1:] - xy[i]) ** 2, axis=1)
        suppressed[i + 1:] |= d2 <= r2
    return np.asarray(keep, dtype=np.int64)


def classwise_circle_nms(dets, radii):
    """``radii``: {class_name: radius_m}. Classes without an entry are kept as is."""
    keep = []
    for ci, name in enumerate(CLASSES):
        idx = np.nonzero(dets.labels == ci)[0]
        if len(idx) == 0:
            continue
        r = radii.get(name)
        if r is None or r <= 0:
            keep.append(idx)
        else:
            keep.append(idx[circle_nms(dets.boxes[idx, :2], dets.scores[idx], r)])
    if not keep:
        return dets.select(np.zeros(0, dtype=np.int64))
    return dets.select(np.sort(np.concatenate(keep)))
