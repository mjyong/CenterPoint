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
                      least once
  false_episodes      switches to moving on those tracks; ``false_episodes_per_h``: per hour of
                      recording (robot time, not vehicle time)
  segments            label moving stretches of at least ``min_segment`` seconds seen by a track;
                      ``latency_*``: from the stretch start (or the first match in it) to the first
                      frame reported moving; ``missed`` never reported moving
  relevant_*          the same for the stretches that matter to the robot: median label speed at
                      least ``relevant_speed`` and median range at most ``relevant_range``. A truck
                      creeping at 0.6 m/s 40 m away is a moving stretch for the labels, but no
                      reason to stop
  details             the missed stretches and false episodes, one dict each, for a look in Foxglove
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
    def __init__(self, classes=("vehicle",), dist=2.0, min_segment=1.0, relevant_speed=1.0, relevant_range=30.0):
        self.classes = [CLASSES.index(c) for c in classes]
        self.dist = dist
        self.min_segment = min_segment
        self.relevant_speed = relevant_speed
        self.relevant_range = relevant_range
        # label track key -> [(stamp, label moving, label speed, range, online state or None,
        #                      displacement speed, velocity-head speed)]
        self.rows = defaultdict(list)
        self.recording_s = 0.0

    def update(self, stamp, gt, tracks):
        """``gt``: list of dict(key, label (class index), xy (2,) world, moving, and optionally speed
        (label, m/s) and range (from the robot, m)); ``tracks``: reported ``TrackState`` in the same
        world frame."""
        for c in self.classes:
            g = [o for o in gt if o["label"] == c]
            t = [s for s in tracks if s.label == c]
            pairs = dict(_match(np.array([o["xy"] for o in g]).reshape(-1, 2),
                                np.array([s.position[:2] for s in t]).reshape(-1, 2), self.dist))
            for i, o in enumerate(g):
                s = t[pairs[i]] if i in pairs else None
                self.rows[o["key"]].append((
                    stamp, bool(o["moving"]), float(o.get("speed", np.nan)), float(o.get("range", np.nan)),
                    None if s is None else s.motion,
                    np.nan if s is None else float(getattr(s, "speed_displacement", np.nan)),
                    np.nan if s is None else float(getattr(s, "speed_head", np.nan))))

    def summary(self):
        mov = mov_hit = stat = stat_hit = 0
        static_tracks = static_flagged = episodes = 0
        lat, rel_lat = [], []
        missed = rel_missed = 0
        details = []
        for key, rows in self.rows.items():
            rows.sort(key=lambda r: r[0])
            t = np.array([r[0] for r in rows])
            lab = np.array([r[1] for r in rows])
            speed = np.array([r[2] for r in rows])
            rng = np.array([r[3] for r in rows])
            on = [r[4] for r in rows]
            disp = np.array([r[5] for r in rows])
            head = np.array([r[6] for r in rows])
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
                edges = np.diff(np.r_[0, online.astype(np.int8), 0])
                for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
                    episodes += 1
                    details.append(dict(kind="false moving", key=key, t=float(t[a]), duration=float(t[b] - t[a] + period),
                                        range=float(np.nanmedian(rng[a:b + 1])),
                                        displacement_speed=float(np.nanmax(disp[a:b + 1])),
                                        head_speed=float(np.nanmax(head[a:b + 1]))))
            # moving stretches of the label
            edges = np.diff(np.r_[0, lab.astype(np.int8), 0])
            for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
                if t[b] - t[a] + period < self.min_segment or not matched[a:b + 1].any():
                    continue
                v, r = float(np.nanmedian(speed[a:b + 1])), float(np.nanmedian(rng[a:b + 1]))
                relevant = not (v < self.relevant_speed) and not (r > self.relevant_range)
                start = t[a:b + 1][matched[a:b + 1]][0]
                hit = np.nonzero(online[a:b + 1])[0]
                if len(hit):
                    lat.append(max(0.0, t[a + hit[0]] - start))
                    if relevant:
                        rel_lat.append(lat[-1])
                else:
                    missed += 1
                    rel_missed += int(relevant)
                    with np.errstate(all="ignore"):
                        details.append(dict(kind="missed", key=key, t=float(t[a]), duration=float(t[b] - t[a] + period),
                                            label_speed=v, range=r, relevant=relevant,
                                            displacement_speed=float(np.nanmax(disp[a:b + 1])),
                                            head_speed=float(np.nanmax(head[a:b + 1]))))
        lat, rel_lat = np.array(lat), np.array(rel_lat)
        q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")   # noqa: E731
        hours = self.recording_s / 3600.0
        return dict(
            moving_frames=mov, moving_recall=mov_hit / mov if mov else float("nan"),
            static_frames=stat, false_moving=stat_hit / stat if stat else float("nan"),
            static_tracks=static_tracks, static_flagged=static_flagged, false_episodes=episodes,
            recording_h=hours, false_episodes_per_h=episodes / hours if hours else float("nan"),
            segments=len(lat) + missed, missed=missed, latency_median=q(lat, 50), latency_p90=q(lat, 90),
            relevant_segments=len(rel_lat) + rel_missed, relevant_missed=rel_missed,
            relevant_latency_median=q(rel_lat, 50), relevant_latency_p90=q(rel_lat, 90),
            details=sorted(details, key=lambda d: (d["kind"], str(d["key"]), d["t"])))


def evaluate_motion(sequences, tracker_cfg=None, classes=("vehicle",), dist=2.0, min_segment=1.0, **relevance):
    """``sequences``: list of scenes, each a list of (stamp, Detections in the world frame, gt list as
    in ``MotionStateAccumulator.update``). The tracker runs on each scene from scratch."""
    acc = MotionStateAccumulator(classes, dist, min_segment, **relevance)
    for frames in sequences:
        trk = MultiObjectTracker(tracker_cfg or TrackerConfig())
        for stamp, dets, gt in frames:
            acc.update(stamp, gt, trk.step(dets, stamp))
        if len(frames) > 1:
            acc.recording_s += frames[-1][0] - frames[0][0] + float(np.median(np.diff([f[0] for f in frames])))
    return acc.summary()
