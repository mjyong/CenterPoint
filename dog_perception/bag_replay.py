"""Replay a kygi665 recording (XT32 + RTK IMU + INSPVAX) through the pipeline.

Shared by the regression tests and the trace tools. Message order is the
bag's log order, as on the robot; the pipeline is built on the first scan
(base height estimated from it), with the measured kygi conventions
(``dog_perception/inspvax.py``) unless ``kygi=False``.
"""
from pathlib import Path

import numpy as np

from .geometry import make_T, rot_zyx, transform_points
from .inspvax import KYGI_INS_LEVER_ARM, KYGI_LIDAR_RPY, EnuOrigin, gyro_from_kygi_imu, pose_from_inspvax, register_kygi
from .pipeline import PerceptionPipeline
from .preprocess import DetFrameConfig, PreprocessConfig, estimate_base_height
from .ros_utils import scan_from_msg, stamp_of


def replay_kygi(bag, detector=None, kygi=True, max_scans=None, predictor=None):
    """Returns one entry per scan with a pose: ``PreprocessedFrame`` when
    ``detector`` is None, else ``PipelineOutput``. ``kygi=False`` reproduces
    the old conventions (gyro read as rad/s, lidar axes = body axes)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    rpy = KYGI_LIDAR_RPY if kygi else (0.0, 0.0, 0.0)
    T_bl = make_T(rot_zyx(rpy[2], rpy[1], rpy[0]))
    ts = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    pipe, origin, pending, outs = None, None, [], []
    with AnyReader([Path(bag)], default_typestore=ts) as r:
        for c, _, raw in r.messages():
            m = r.deserialize(raw, c.msgtype)
            if c.msgtype.endswith("Imu"):
                w = m.angular_velocity
                ev = ("imu", stamp_of(m.header), gyro_from_kygi_imu(m) if kygi else (w.x, w.y, w.z))
            elif c.msgtype.endswith("INSPVAX"):
                if origin is None and int(m.ins_status.data) == 3:
                    origin = EnuOrigin(m.latitude, m.longitude, m.height)
                g = pose_from_inspvax(m, origin, KYGI_INS_LEVER_ARM if kygi else None) if origin is not None else None
                if g is None:
                    continue
                ev = ("odom",) + tuple(g)
            else:
                if max_scans is not None and len(outs) >= max_scans:
                    break
                scan = scan_from_msg(m, "timestamp", "absolute")
                if pipe is None:
                    h = estimate_base_height(transform_points(T_bl, scan.xyz[np.isfinite(scan.xyz).all(1)]))
                    pipe = PerceptionPipeline(PreprocessConfig(T_body_lidar=T_bl, det_frame=DetFrameConfig(
                        base_height=h, auto_ground=True, auto_level=kygi)), detector, predictor=predictor)
                    for e in pending:
                        (pipe.on_imu if e[0] == "imu" else pipe.on_odometry)(*e[1:])
                out = pipe.pre.process(scan) if detector is None else pipe.on_scan(scan)
                if out is not None:
                    outs.append(out)
                continue
            if pipe is None:
                pending.append(ev)
            else:
                (pipe.on_imu if ev[0] == "imu" else pipe.on_odometry)(*ev[1:])
    return outs
