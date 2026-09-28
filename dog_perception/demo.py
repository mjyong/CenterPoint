"""Glue used by the simulation demo and the tests."""
import numpy as np

from .detection import OracleDetector, OracleNoise
from .geometry import inv_T
from .pose_buffer import ImuPropagator, PoseBuffer
from .preprocess import det_frame_from_body
from .prediction.dataset import TrackLogger
from .prediction.kinematic import IMMPredictor
from .sim import GaitModel, RobotTrajectory, SimObject
from .tracking import MultiObjectTracker, TrackerConfig


def oracle_tracking_log(seed=0, duration=60.0, num_pedestrians=12, num_cyclists=4, num_vehicles=3,
                        rate=10.0, noise=None, tracker_cfg=None, area=(25.0, 12.0)):
    """Oracle detections (no ray casting) -> tracker -> TrackLogger dict.

    Cheap way to produce lots of tracker logs for the tier-2 predictor.
    """
    rng = np.random.default_rng(seed)
    robot = RobotTrajectory(GaitModel(speed=0.4, yaw_rate=0.05))
    labels = ["pedestrian"] * num_pedestrians + ["cyclist"] * num_cyclists + ["vehicle"] * num_vehicles
    objs = [SimObject.random(i, lab, rng, duration + 1.0, (0.0, 0.0), area) for i, lab in enumerate(labels)]
    det = OracleDetector(noise or OracleNoise(), seed=seed + 1)
    trk = MultiObjectTracker(tracker_cfg or TrackerConfig())
    log = TrackLogger()
    predictor = IMMPredictor()
    for k in range(int(duration * rate)):
        t = (k + 1) / rate
        T_wb = robot.T(t)
        T_wd = det_frame_from_body(T_wb, robot.g.base_height)
        gt = []
        for o in objs:
            c, y, v = o.state([t])
            gt.append(dict(id=o.id, label=o.label, center=c[0], size=o.size, yaw=float(y[0]),
                           velocity=v[0], num_points=40))
        dets = det(gt, inv_T(T_wd), t, use_points=False).transform(T_wd, "world")
        states = trk.step(dets, t)
        log.add(t, states, T_wb, predictor(trk))
    return log.to_dict()


def feed_sim_poses(sim, t0, t1, prop=None, odom_rate=10.0, odom_latency=0.03):
    """Replays 400 Hz gyro + delayed 10 Hz odometry of the simulator into an ImuPropagator."""
    if prop is None:
        prop = ImuPropagator(PoseBuffer())
    imu_t, imu_w = sim.imu(t0, t1)
    events = [(t, 0, w) for t, w in zip(imu_t, imu_w)]
    for to in np.arange(np.ceil(t0 * odom_rate), np.floor(t1 * odom_rate) + 1) / odom_rate:
        events.append((to + odom_latency, 1, to))
    events.sort(key=lambda e: (e[0], e[1]))
    for te, kind, val in events:
        if te > t1:
            continue
        if kind == 0:
            prop.on_imu(te, val)
        else:
            R, p, v = sim.odometry(val)
            prop.on_odometry(val, R, p, v)
    return prop
