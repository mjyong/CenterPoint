"""The four stages wired together.

    pipe = PerceptionPipeline(PreprocessConfig(...), detector, predictor="imm")
    pipe.on_imu(t, gyro); pipe.on_odometry(t, R, p, v)     # high-rate callbacks
    out = pipe.on_scan(scan)                                 # 10 Hz
    out.tracks / out.predictions                             # world frame
    pipe.to_local(out)                                       # for the local planner
"""
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detection.boxes import Detections
from .geometry import inv_T
from .pose_buffer import ImuPropagator, PoseBuffer
from .prediction.kinematic import IMMPredictor
from .preprocess import PreprocessConfig, Preprocessor
from .tracking import MultiObjectTracker, TrackerConfig


@dataclass
class PipelineOutput:
    stamp: float
    frame: object                 # PreprocessedFrame
    detections: Detections        # det frame
    detections_world: Detections
    tracks: list                  # TrackState, world frame
    predictions: list             # Prediction, world frame
    timings: dict = field(default_factory=dict)


class PerceptionPipeline:
    def __init__(self, preprocess_cfg=None, detector=None, tracker_cfg=None, predictor="imm",
                 pose_buffer=None, gyro_bias=(0.0, 0.0, 0.0)):
        """detector: callable(points_det (N,5), stamp) -> Detections (e.g. CenterPointDetector).
        predictor: "imm", None, or an object with ``__call__(tracker, ego_hist)`` (LearnedPredictor)."""
        self.poses = pose_buffer or PoseBuffer()
        self.imu = ImuPropagator(self.poses, gyro_bias)
        self.pre = Preprocessor(preprocess_cfg or PreprocessConfig(), self.poses)
        self.detector = detector
        self.tracker = MultiObjectTracker(tracker_cfg or TrackerConfig())
        self.predictor = IMMPredictor() if predictor == "imm" else predictor
        self._ego = deque(maxlen=64)

    # ---------------------------------------------------------- pose inputs
    def on_imu(self, t, gyro):
        self.imu.on_imu(t, gyro)

    def on_odometry(self, t, R_wb, p_wb, v_w=None):
        self.imu.on_odometry(t, R_wb, p_wb, v_w)

    # ---------------------------------------------------------- lidar input
    def on_scan(self, scan, dynamic_boxes=None, detections=None):
        """``detections`` may be given to bypass the detector (oracle / offline results)."""
        t0 = time.perf_counter()
        frame = self.pre.process(scan, dynamic_boxes)
        if frame is None:
            return None
        t1 = time.perf_counter()
        dets = detections if detections is not None else self.detector(frame.points, frame.stamp)
        dets.stamp = frame.stamp
        t2 = time.perf_counter()
        dets_w = dets.transform(frame.T_world_det, "world")
        tracks = self.tracker.step(dets_w, frame.stamp)
        t3 = time.perf_counter()

        T = frame.T_world_body
        if self._ego:
            dt = frame.stamp - self._ego[-1][0]
            v = (T[:2, 3] - self._ego[-1][1:3]) / dt if dt > 0 else np.zeros(2)
        else:
            v = np.zeros(2)
        self._ego.append(np.array([frame.stamp, T[0, 3], T[1, 3], v[0], v[1], 1.0]))

        preds = []
        if self.predictor is not None:
            if isinstance(self.predictor, IMMPredictor):
                preds = self.predictor(self.tracker)
            else:
                preds = self.predictor(self.tracker, np.asarray(self._ego))
        t4 = time.perf_counter()
        timings = dict(frame.timings)
        timings.update(preprocess_ms=1e3 * (t1 - t0), detect_ms=1e3 * (t2 - t1),
                       track_ms=1e3 * (t3 - t2), predict_ms=1e3 * (t4 - t3), total_ms=1e3 * (t4 - t0))
        if hasattr(self.detector, "last_timing"):
            timings.update({"det_" + k: v for k, v in self.detector.last_timing.items()})
        return PipelineOutput(frame.stamp, frame, dets, dets_w, tracks, preds, timings)

    def to_local(self, out):
        """Tracks / predictions in the det frame of the scan (gravity aligned, yaw
        following the body, origin on the ground: a ``base_footprint``-like frame
        that 2D local planners expect)."""
        T_dw = inv_T(out.frame.T_world_det)
        R = T_dw[:3, :3]
        dyaw = np.arctan2(R[1, 0], R[0, 0])
        tracks = []
        for s in out.tracks:
            tracks.append(dict(id=s.track_id, label=s.name, position=R @ s.position + T_dw[:3, 3],
                               velocity=R[:2, :2] @ s.velocity, yaw=float(s.yaw + dyaw),
                               size=s.size, score=s.score, coasting=s.coasting))
        return tracks, [p.transform(T_dw, "det") for p in out.predictions]
