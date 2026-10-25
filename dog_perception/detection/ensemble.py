"""Several teachers' detections of one frame merged into one list (weighted box fusion).

Boxes of the same class from different teachers that cover the same object (centers within ``dist``,
or one center inside the other box grown by ``margin``: a long truck's two boxes can sit apart) are
one object, at most one box per teacher (its best). The fused box is the score-weighted mean of
them (headings first turned to agree with the best box's within 90 deg, a box reads the same turned
by 180 deg); velocity likewise; subclass from the best box. The score is the mean over all teachers,
a teacher that did not detect the object counting as 0: what both detect ranks above what one
detects, and one teacher's box keeps half its score.

With the auto-label's rules (tools/dog/auto_label.py: a track needs a mean score >= 0.3, and one
below 0.5 without 5 frames >= 0.6 is flagged for review) that is the two-teacher policy: tracks both
teachers see follow the usual rules; a track only one teacher sees stays only when that teacher is
sure of it (its own mean >= 0.6) and is always flagged for review.
"""
import numpy as np

from ..geometry import wrap_angle
from .boxes import Detections


def _covers(a, b, dist, margin):
    """Boxes ``a`` and ``b`` (standard, 7) are the same object."""
    d = b[:2] - a[:2]
    if np.hypot(*d) <= dist:
        return True
    for box, p in ((a, d), (b, -d)):
        c, s = np.cos(box[6]), np.sin(box[6])
        if abs(c * p[0] + s * p[1]) <= box[3] / 2 + margin and abs(-s * p[0] + c * p[1]) <= box[4] / 2 + margin:
            return True
    return False


def fuse(dets_list, dist=1.5, margin=0.5):
    """(Detections, agree): the fused detections of the teachers' ``dets_list`` (same frame), and for each
    how many teachers detected it."""
    dets_list = [d for d in dets_list if d is not None]
    n = len(dets_list)
    if n == 0:
        return Detections(), np.zeros(0, np.int64)
    if n == 1:
        return dets_list[0], np.ones(len(dets_list[0]), np.int64)
    rows = [(float(d.scores[i]), t, i) for t, d in enumerate(dets_list) for i in range(len(d))]
    rows.sort(key=lambda r: -r[0])
    clusters = []                                     # [label, best box, {teacher: index}]
    for score, t, i in rows:
        d = dets_list[t]
        box, label = d.boxes[i], int(d.labels[i])
        hit = next((c for c in clusters if c[0] == label and t not in c[2] and _covers(c[1], box, dist, margin)), None)
        if hit is None:
            clusters.append([label, box, {t: i}])
        else:
            hit[2][t] = i
    boxes, vels, scores, labels, subs, agree = [], [], [], [], [], []
    for label, best, members in clusters:
        w = np.array([dets_list[t].scores[i] for t, i in members.items()], float)
        bs = np.array([dets_list[t].boxes[i] for t, i in members.items()], float)
        vs = np.array([dets_list[t].velocities[i] for t, i in members.items()], float)
        flip = np.abs(wrap_angle(bs[:, 6] - best[6])) > np.pi / 2
        bs[flip, 6] += np.pi
        box = (bs * w[:, None]).sum(0) / w.sum()
        box[6] = np.arctan2((np.sin(bs[:, 6]) * w).sum(), (np.cos(bs[:, 6]) * w).sum())
        t0, i0 = next(iter(members.items()))          # the best box (rows were sorted by score)
        boxes.append(box)
        vels.append((vs * w[:, None]).sum(0) / w.sum())
        scores.append(w.sum() / n)
        labels.append(label)
        subs.append(int(dets_list[t0].subclasses[i0]))
        agree.append(len(members))
    ref = dets_list[0]
    out = Detections(np.array(boxes).reshape(-1, 7), np.array(vels).reshape(-1, 2), np.array(scores),
                     np.array(labels, np.int64), ref.frame, ref.stamp, np.array(subs, np.int64))
    return out, np.array(agree, np.int64)
