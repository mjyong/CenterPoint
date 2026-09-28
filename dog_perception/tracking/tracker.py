"""Multi-object tracker in the (gravity aligned) LIO world frame.

Why world frame: the dog turns in place and bounces; in the body frame a
static pedestrian would appear to move. The detector's velocity is already
over-ground (sweeps are motion compensated), so it is a direct measurement
of the track velocity once rotated into the world frame.

Association (SimpleTrack / ByteTrack style):
  1. high-score detections vs. all tracks (IMM-predicted centers, class gated)
  2. low-score detections vs. still unmatched confirmed tracks only
     (recovers occluded / far pedestrians without spawning false tracks)
  3. unmatched high-score detections start tentative tracks

Life cycle: tentative -> confirmed after ``min_hits`` hits; tentative tracks
tolerate a single missed frame; confirmed tracks coast on the IMM prediction for
up to ``max_age`` seconds (occlusion) before being deleted.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..detection.boxes import CLASSES
from ..geometry import wrap_angle
from .association import match
from .imm import IMM, build_model


@dataclass
class ClassParams:
    gate: float                 # association radius [m]
    max_age: float              # coasting time before deletion [s]
    min_hits: int
    pos_std: float              # detector center noise [m]
    vel_std: float              # detector velocity noise [m/s]
    models: tuple               # IMM model specs
    stay_prob: float = 0.95
    init_omega_std: float = 0.3


DEFAULT_CLASS_PARAMS = {
    "vehicle": ClassParams(3.0, 1.0, 2, 0.30, 0.8, (("cv", 1.5), ("ct", 1.5, 0.25))),
    "pedestrian": ClassParams(1.5, 1.5, 2, 0.15, 0.5, (("cv", 0.4), ("cv", 2.0))),
    "cyclist": ClassParams(2.5, 1.0, 2, 0.20, 0.6, (("cv", 1.5), ("ct", 1.5, 0.4))),
}


@dataclass
class TrackerConfig:
    class_params: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_PARAMS))
    high_score: float = 0.35
    low_score: float = 0.1
    matching: str = "greedy"             # or "hungarian"
    use_velocity: bool = True            # CenterPoint velocity head as a measurement
    tentative_max_age: float = 0.15      # [s]; at 10 Hz a tentative track survives one missed frame
    output_coasting: bool = True
    history: float = 3.0                 # [s] kept for the predictors
    size_alpha: float = 0.2
    z_alpha: float = 0.3
    yaw_alpha: float = 0.4
    heading_from_velocity: float = 1.0   # [m/s] above this, box heading follows the velocity


@dataclass
class TrackState:
    track_id: int
    label: int
    stamp: float
    position: np.ndarray       # (3,) world
    velocity: np.ndarray       # (2,) world
    yaw: float
    size: np.ndarray           # (3,) l, w, h
    score: float
    covariance: np.ndarray     # (4, 4) over [px, py, vx, vy]
    mode_probs: np.ndarray
    age: float
    hits: int
    coasting: bool
    history: np.ndarray        # (K, 6) [t, x, y, vx, vy, observed]

    @property
    def name(self):
        return CLASSES[self.label]


class Track:
    def __init__(self, track_id, label, box, vel, score, stamp, params, cfg):
        self.id = track_id
        self.label = label
        self.params = params
        self.cfg = cfg
        self.imm = IMM([build_model(s) for s in params.models], params.stay_prob)
        v = vel if (cfg.use_velocity and vel is not None) else np.zeros(2)
        v_var = params.vel_std ** 2 if cfg.use_velocity else 4.0
        P0 = np.diag([params.pos_std ** 2] * 2 + [v_var] * 2 + [params.init_omega_std ** 2])
        self.imm.initialize([box[0], box[1], v[0], v[1], 0.0], P0)
        self.z, self.size, self.yaw = float(box[2]), np.array(box[3:6], dtype=np.float64), float(box[6])
        self.score = float(score)
        self.birth = self.last_update = self.stamp = stamp
        self.hits = 1
        self.confirmed = params.min_hits <= 1
        self.history = deque()
        self._record(True)

    @property
    def time_since_update(self):
        return self.stamp - self.last_update

    def predict(self, stamp):
        dt = stamp - self.stamp
        if dt > 0:
            self.imm.predict(dt)
            self.stamp = stamp

    def gate(self):
        _, P = self.imm.state
        sigma = np.sqrt(max(np.trace(P[:2, :2]) / 2, 0.0))
        return float(np.clip(3 * sigma, self.params.gate, 3 * self.params.gate))

    def update(self, box, vel, score):
        p, cfg = self.params, self.cfg
        if cfg.use_velocity and vel is not None:
            H = np.zeros((4, 5))
            H[:4, :4] = np.eye(4)
            z = np.array([box[0], box[1], vel[0], vel[1]])
            R = np.diag([p.pos_std ** 2] * 2 + [p.vel_std ** 2] * 2)
        else:
            H = np.zeros((2, 5))
            H[:2, :2] = np.eye(2)
            z, R = np.array(box[:2]), np.eye(2) * p.pos_std ** 2
        self.imm.update(z, H, R)

        self.z += cfg.z_alpha * (box[2] - self.z)
        self.size += cfg.size_alpha * (np.asarray(box[3:6]) - self.size)
        self._update_yaw(float(box[6]))
        self.score = 0.7 * self.score + 0.3 * float(score)
        self.hits += 1
        self.last_update = self.stamp
        if self.hits >= p.min_hits:
            self.confirmed = True
        self._record(True)

    def _update_yaw(self, yaw_det):
        cfg = self.cfg
        x, _ = self.imm.state
        speed = np.hypot(x[2], x[3])
        if CLASSES[self.label] == "pedestrian":
            # pedestrian boxes are ~square: heading only means something when walking
            if speed > 0.4:
                self.yaw = float(np.arctan2(x[3], x[2]))
            return
        if abs(wrap_angle(yaw_det - self.yaw)) > np.pi / 2:   # resolve the 180 deg ambiguity
            yaw_det += np.pi
        if speed > cfg.heading_from_velocity:
            heading = np.arctan2(x[3], x[2])
            if abs(wrap_angle(yaw_det - heading)) > np.pi / 2:
                yaw_det += np.pi
        self.yaw = float(wrap_angle(self.yaw + cfg.yaw_alpha * wrap_angle(yaw_det - self.yaw)))

    def mark_missed(self):
        self.score *= 0.9
        self._record(False)

    def _record(self, observed):
        x, _ = self.imm.state
        self.history.append((self.stamp, x[0], x[1], x[2], x[3], float(observed)))
        while self.history and self.history[0][0] < self.stamp - self.cfg.history:
            self.history.popleft()

    def to_state(self):
        x, P = self.imm.state
        return TrackState(
            track_id=self.id, label=self.label, stamp=self.stamp,
            position=np.array([x[0], x[1], self.z]), velocity=x[2:4].copy(), yaw=self.yaw,
            size=self.size.copy(), score=self.score, covariance=P[:4, :4].copy(),
            mode_probs=self.imm.mu.copy(), age=self.stamp - self.birth, hits=self.hits,
            coasting=self.time_since_update > 1e-6, history=np.asarray(self.history))


class MultiObjectTracker:
    def __init__(self, cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.tracks = []
        self._next_id = 1
        self.stamp = None
        self.last_matches = {}

    def reset(self):
        self.tracks, self._next_id, self.stamp, self.last_matches = [], 1, None, {}

    def _params(self, label):
        return self.cfg.class_params[CLASSES[label]]

    def _associate(self, dets, det_idx, trk_idx):
        if len(det_idx) == 0 or len(trk_idx) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        d_xy = dets.boxes[det_idx, :2]
        t_xy = np.array([self.tracks[t].imm.state[0][:2] for t in trk_idx])
        cost = np.linalg.norm(d_xy[:, None, :] - t_xy[None, :, :], axis=2)
        gates = np.array([self.tracks[t].gate() for t in trk_idx])
        same = dets.labels[det_idx][:, None] == np.array([self.tracks[t].label for t in trk_idx])[None, :]
        cost = np.where(same & (cost <= gates[None, :]), cost, np.inf)
        m = match(cost, 1e9, self.cfg.matching)
        return np.stack([det_idx[m[:, 0]], trk_idx[m[:, 1]]], axis=1) if len(m) else m

    def step(self, dets, stamp):
        """dets: ``Detections`` in the world frame. Returns list of ``TrackState``."""
        cfg = self.cfg
        self.stamp = stamp
        for t in self.tracks:
            t.predict(stamp)

        keep = dets.labels >= 0
        high = np.nonzero(keep & (dets.scores >= cfg.high_score))[0]
        low = np.nonzero(keep & (dets.scores >= cfg.low_score) & (dets.scores < cfg.high_score))[0]

        all_trk = np.arange(len(self.tracks))
        m1 = self._associate(dets, high, all_trk)
        matched_t = set(m1[:, 1].tolist())
        rest = np.array([i for i in all_trk if i not in matched_t and self.tracks[i].confirmed], dtype=np.int64)
        m2 = self._associate(dets, low, rest)
        matches = np.concatenate([m1, m2]) if len(m2) else m1

        vel = dets.velocities if cfg.use_velocity else None
        self.last_matches = {}          # track id -> detection index (used by auto-labelling)
        for d, t in matches:
            self.tracks[t].update(dets.boxes[d], None if vel is None else vel[d], dets.scores[d])
            self.last_matches[self.tracks[t].id] = int(d)
        matched_t = set(matches[:, 1].tolist())
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.mark_missed()

        matched_d = set(matches[:, 0].tolist())
        for d in high:
            if d not in matched_d:
                self.tracks.append(Track(self._next_id, int(dets.labels[d]), dets.boxes[d],
                                         None if vel is None else vel[d], dets.scores[d], stamp,
                                         self._params(int(dets.labels[d])), cfg))
                self.last_matches[self._next_id] = int(d)
                self._next_id += 1

        self.tracks = [t for t in self.tracks if
                       (t.confirmed and t.time_since_update <= self._params(t.label).max_age)
                       or (not t.confirmed and t.time_since_update <= cfg.tentative_max_age)]
        return self.outputs()

    def outputs(self):
        out = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            if t.time_since_update > 1e-6 and not self.cfg.output_coasting:
                continue
            out.append(t.to_state())
        return out
