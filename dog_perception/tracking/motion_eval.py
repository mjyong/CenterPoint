"""Online motion state (``TrackState.motion``) against the auto-labels' ``moving`` flags.

The robot stops for a moving vehicle and walks past a parked one, so the two errors cost differently:

* false moving: a vehicle the labels call static is reported moving, and the robot stops for nothing;
* late or missed moving: a moving vehicle is reported static, and the robot does not stop in time.

The labels decide with hindsight over the whole track (``autolabel.refine_track``); the tracker
only has the past. Each frame, the reported tracks are matched to the label objects of the same
class within ``dist`` (closest pairs first). Per match the label's ``moving`` flag is compared with
the online state; "stopped" counts as not moving (the robot may treat it either way).

Summary:
  moving_recall       matched label-moving frames reported moving
  false_moving        matched label-static frames reported moving
  static_tracks       label tracks never moving; ``static_flagged`` of them reported moving at
                      least once, ``false_episodes_per_h`` switches to moving per hour of static
                      vehicles in view
  segments            label moving stretches of at least ``min_segment`` seconds seen by a track;
                      ``latency_*``: from the stretch start (or the first match in it) to the first
                      frame reported moving; ``missed`` never reported moving
"""
from collections import defaultdict

import numpy as np

from ..detection.boxes import CLASSES
from .tracker import MultiObjectTracker, TrackerConfig


def _match(gt_xy, trk_xy, dist):
    if len(gt_xy) == 0 or len(trk_xy) == 0:
        return []
    d = np.linalg.norm(gt_xy[:, None] - trk_xy[None], axis=2)
    pairs = []
    while True:
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > dist:
            return pairs
        pairs.append((i, j))
        d[i, :] = np.inf
        d[:, j] = np.inf


class MotionStateAccumulator:
    def __init__(self, classes=("vehicle",), dist=2.0, min_segment=1.0):
        self.classes = [CLASSES.index(c) for c in classes]
        self.dist = dist
        self.min_segment = min_segment
        self.rows = defaultdict(list)      # label track key -> [(stamp, label moving, online state or None)]

    def update(self, stamp, gt, tracks):
        """``gt``: list of dict(key, label (class index), xy (2,) world, moving); ``tracks``: reported
        ``TrackState`` in the same world frame."""
        for c in self.classes:
            g = [o for o in gt if o["label"] == c]
            t = [s for s in tracks if s.label == c]
            pairs = dict(_match(np.array([o["xy"] for o in g]).reshape(-1, 2),
                                np.array([s.position[:2] for s in t]).reshape(-1, 2), self.dist))
            for i, o in enumerate(g):
                self.rows[o["key"]].append((stamp, bool(o["moving"]), t[pairs[i]].motion if i in pairs else None))

    def summary(self):
        mov = mov_hit = stat = stat_hit = 0
        static_tracks = static_flagged = episodes = 0
        static_time = 0.0
        latencies, missed = [], 0
        for rows in self.rows.values():
            rows.sort(key=lambda r: r[0])
            t = np.array([r[0] for r in rows])
            lab = np.array([r[1] for r in rows])
            on = [r[2] for r in rows]
            matched = np.array([s is not None for s in on])
            online = np.array([s == "moving" for s in on])
            period = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
            mov += int(np.sum(matched & lab))
            mov_hit += int(np.sum(matched & lab & online))
            stat += int(np.sum(matched & ~lab))
            stat_hit += int(np.sum(matched & ~lab & online))
            if not lab.any() and matched.any():
                static_tracks += 1
                static_flagged += int(online.any())
                episodes += int(np.sum(np.diff(np.r_[0, online.astype(np.int8)]) == 1))
                static_time += period * int(matched.sum())
            # moving stretches of the label
            edges = np.diff(np.r_[0, lab.astype(np.int8), 0])
            for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
                if t[b] - t[a] + period < self.min_segment or not matched[a:b + 1].any():
                    continue
                start = t[a:b + 1][matched[a:b + 1]][0]
                hit = np.nonzero(online[a:b + 1])[0]
                if len(hit):
                    latencies.append(max(0.0, t[a + hit[0]] - start))
                else:
                    missed += 1
        lat = np.array(latencies)
        return dict(
            moving_frames=mov, moving_recall=mov_hit / mov if mov else float("nan"),
            static_frames=stat, false_moving=stat_hit / stat if stat else float("nan"),
            static_tracks=static_tracks, static_flagged=static_flagged,
            false_episodes_per_h=episodes / (static_time / 3600.0) if static_time else float("nan"),
            segments=len(lat) + missed, missed=missed,
            latency_median=float(np.median(lat)) if len(lat) else float("nan"),
            latency_p90=float(np.percentile(lat, 90)) if len(lat) else float("nan"))


def evaluate_motion(sequences, tracker_cfg=None, classes=("vehicle",), dist=2.0, min_segment=1.0):
    """``sequences``: list of scenes, each a list of (stamp, Detections in the world frame, gt list as
    in ``MotionStateAccumulator.update``). The tracker runs on each scene from scratch."""
    acc = MotionStateAccumulator(classes, dist, min_segment)
    for frames in sequences:
        trk = MultiObjectTracker(tracker_cfg or TrackerConfig())
        for stamp, dets, gt in frames:
            acc.update(stamp, gt, trk.step(dets, stamp))
    return acc.summary()
