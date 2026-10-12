"""Filters / thresholds added after running the sample bags, plus regression
runs on them (skipped when the bags, weights or rosbags are missing)."""
import os

import numpy as np
import pytest

from dog_perception.detection import (DEFAULT_SCORE_THRESHOLDS, Detections, DetectionFilterConfig,
                                      filter_detections)
from dog_perception.preprocess import LidarScan, dedup_returns, estimate_base_height
from dog_perception.tracking import MultiObjectTracker, TrackerConfig

from .conftest import ROOT, requires_det3d

BAG = os.path.join(ROOT, "samples", "rosbag2_2026_09_21-16_41_27")
MOVING_BAG = os.path.join(ROOT, "samples", "rosbag2_2026_09_21-16_41_27_moving")
PP_CKPT = os.path.join(ROOT, "work_dirs", "pretrained", "nusc_pp.pth")


def _box_points(center, size, n, rng):
    return center + (rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size))


def test_filter_min_points_ground_and_cross_class():
    rng = np.random.default_rng(0)
    car = [10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0]
    pts = [_box_points(car[:3], car[3:6], 200, rng)]
    boxes = [car,
             [10.3, 0.2, 0.7, 1.8, 0.6, 1.4, 0.0],     # cyclist head firing on the car -> duplicate
             [20.0, 5.0, 0.9, 0.6, 0.6, 1.7, 0.0],     # pedestrian in empty space -> no points
             [15.0, -5.0, 2.2, 0.6, 0.6, 1.7, 0.0],    # pedestrian floating 1.35 m above ground
             [5.0, 5.0, 0.9, 0.6, 0.6, 1.7, 0.0]]      # real pedestrian
    pts.append(_box_points([15.0, -5.0, 2.2], [0.6, 0.6, 1.7], 50, rng))
    pts.append(_box_points([5.0, 5.0, 0.9], [0.6, 0.6, 1.7], 40, rng))
    cloud = np.vstack(pts)
    cloud = np.column_stack([cloud, np.zeros(len(cloud)), np.zeros(len(cloud))])   # intensity, dt=0
    dets = Detections(np.array(boxes), np.zeros((5, 2)), np.array([0.8, 0.5, 0.6, 0.6, 0.4]),
                      np.array([0, 2, 1, 1, 1]))
    out = filter_detections(dets, cloud, DetectionFilterConfig())
    assert sorted(map(tuple, out.boxes[:, :2].round(1))) == [(5.0, 5.0), (10.0, 0.0)]
    # points of older sweeps do not count: a box seen only 0.4 s ago is not supported now
    old = cloud.copy()
    old[:, -1] = 0.4
    assert len(filter_detections(dets, old)) == 0


def test_ground_support_uses_local_ground():
    """30 m away the ground is 1 m above the det-frame z = 0 (slope): a car
    standing there is kept, a box floating 1.3 m above that ground is not."""
    rng = np.random.default_rng(0)
    xy = rng.uniform([20, -10], [40, 10], (6000, 2))
    ground = np.column_stack([xy, np.full(len(xy), 1.0) + rng.normal(0, 0.02, len(xy))])
    car = [30.0, 0.0, 1.8, 4.5, 1.9, 1.6, 0.0]
    floating = [30.0, 6.0, 3.15, 0.6, 0.6, 1.7, 0.0]
    pts = np.vstack([ground, _box_points(car[:3], car[3:6], 200, rng), _box_points(floating[:3], floating[3:6], 40, rng)])
    cloud = np.column_stack([pts, np.zeros(len(pts)), np.zeros(len(pts))])
    dets = Detections(np.array([car, floating]), np.zeros((2, 2)), np.array([0.8, 0.6]), np.array([0, 1]))
    out = filter_detections(dets, cloud)
    assert list(out.labels) == [0]
    flat = filter_detections(dets, cloud, DetectionFilterConfig(local_ground=False))
    assert len(flat) == 0                       # the old z = 0 check dropped the car as floating


def test_report_thresholds_are_per_class():
    d = Detections(np.zeros((3, 7)), np.zeros((3, 2)), np.array([0.36, 0.36, 0.36]), np.array([0, 1, 2]))
    assert list(d.above(DEFAULT_SCORE_THRESHOLDS).labels) == [0, 1]   # cyclist needs 0.4
    assert len(d.above(0.5)) == 0


def test_dedup_dual_returns():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-30, 30, (1000, 3))
    dup = np.vstack([xyz, xyz[:400], [[np.nan, 1.0, 2.0]]])
    scan = LidarScan(xyz=dup, intensity=np.arange(len(dup), dtype=np.float32), point_times=np.arange(len(dup)) * 1e-5)
    out = dedup_returns(scan)
    assert len(out.xyz) == 1000 and np.array_equal(out.intensity, np.arange(1000))   # first return kept, order kept


def test_estimate_base_height():
    rng = np.random.default_rng(0)
    r, a = rng.uniform(1, 20, 20000), rng.uniform(-np.pi, np.pi, 20000)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), rng.normal(-0.31, 0.01, 20000)])
    clutter = np.column_stack([rng.uniform(-20, 20, (5000, 2)), rng.uniform(-0.3, 4, 5000)])
    assert abs(estimate_base_height(np.vstack([ground, clutter])) - 0.31) < 0.02


def test_tracker_start_and_report_thresholds():
    cfg = TrackerConfig()
    trk = MultiObjectTracker(cfg)
    cyc = lambda s, x: Detections(np.array([[x, 0, 0.8, 1.8, 0.6, 1.5, 0.0]]), np.zeros((1, 2)), np.array([s]),
                                  np.array([2]), "world")
    for k in range(4):                       # 0.38 >= default 0.35 but < cyclist 0.4: never starts a track
        trk.step(cyc(0.38, 5.0), 0.1 * (k + 1))
    assert trk.tracks == []
    for k in range(3):
        out = trk.step(cyc(0.6, 5.0), 0.5 + 0.1 * k)
    assert len(out) == 1
    for k in range(15):                      # kept alive by low-score boxes, score decays below 0.3
        out = trk.step(cyc(0.12, 5.0), 0.8 + 0.1 * k)
    assert len(trk.tracks) == 1 and out == [] and trk.reported_tracks() == []


@requires_det3d
def test_checkpoint_must_match_config():
    if not os.path.exists(PP_CKPT):
        pytest.skip("pillar weights not in the checkout")
    from dog_perception.detection import build_detector
    from dog_perception.detection.centerpoint import checkpoint_arch

    assert checkpoint_arch(PP_CKPT) == "pillar"
    build_detector("pillar", checkpoint=PP_CKPT, device="cpu")
    try:
        import spconv  # noqa: F401
    except ImportError:
        pytest.skip("voxel config needs spconv to build")
    with pytest.raises(ValueError, match="looks like a pillar model"):
        build_detector("voxel", checkpoint=PP_CKPT, device="cpu")


@requires_det3d
def test_sample_bag_regression(tmp_path):
    """Pretrained nuScenes pillar on the sample bag: raw output ~100 boxes/frame,
    what is reported must stay a small, plausible set."""
    if not (os.path.isdir(BAG) and os.path.exists(PP_CKPT)):
        pytest.skip("sample bag / weights not in the checkout")
    pytest.importorskip("rosbags")
    from pathlib import Path

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.detection import build_detector
    from dog_perception.geometry import make_T, transform_points
    from dog_perception.geometry import rot_zyx
    from dog_perception.inspvax import KYGI_LIDAR_RPY, EnuOrigin, gyro_from_kygi_imu, pose_from_inspvax, register_kygi
    from dog_perception.pipeline import PerceptionPipeline
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig
    from dog_perception.ros_utils import scan_from_msg, stamp_of

    det = build_detector("pillar", checkpoint=PP_CKPT, device="cpu")
    pipe = None
    viz = None
    try:
        from dog_perception.mcap_viz import McapViz
        viz = McapViz(str(tmp_path / "out.mcap"))
    except ImportError:
        pass
    ts = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    origin, pending, stats = None, [], []
    with AnyReader([Path(BAG)], default_typestore=ts) as r:
        for c, _, raw in r.messages():
            m = r.deserialize(raw, c.msgtype)
            if c.msgtype.endswith("Imu"):
                ev = ("imu", stamp_of(m.header), gyro_from_kygi_imu(m))
            elif c.msgtype.endswith("INSPVAX"):
                if origin is None and int(m.ins_status.data) == 3:
                    origin = EnuOrigin(m.latitude, m.longitude, m.height)
                g = pose_from_inspvax(m, origin) if origin is not None else None
                if g is None:
                    continue
                ev = ("odom",) + tuple(g)
            else:
                if len(stats) >= 10:
                    break
                scan = scan_from_msg(m, "timestamp", "absolute")
                if pipe is None:
                    h = estimate_base_height(scan.xyz[np.isfinite(scan.xyz).all(1)])
                    assert 0.25 < h < 0.4          # lidar ~0.3 m above ground on this robot
                    T_bl = make_T(rot_zyx(KYGI_LIDAR_RPY[2], KYGI_LIDAR_RPY[1], KYGI_LIDAR_RPY[0]))
                    pipe = PerceptionPipeline(PreprocessConfig(T_body_lidar=T_bl, det_frame=DetFrameConfig(
                        base_height=h, auto_ground=True, auto_level=True)), det)
                    for e in pending:
                        (pipe.on_imu if e[0] == "imu" else pipe.on_odometry)(*e[1:])
                out = pipe.on_scan(scan)
                if out is not None:
                    shown = out.detections.above(DEFAULT_SCORE_THRESHOLDS)
                    stats.append((det.last_raw_count, len(out.detections), len(shown), len(out.tracks)))
                    if viz is not None:
                        cur = out.frame.points[out.frame.points[:, -1] == 0]
                        viz.add(out.stamp, transform_points(out.frame.T_world_det, cur[:, :3]), cur[:, 3],
                                shown.transform(out.frame.T_world_det, "world"), out.frame.T_world_body, out.tracks)
                continue
            if pipe is None:
                pending.append(ev)
            else:
                (pipe.on_imu if ev[0] == "imu" else pipe.on_odometry)(*ev[1:])
    raw, filt, shown, tracks = np.array(stats).T
    assert len(stats) >= 8
    assert raw.mean() > 60                               # the network itself is chatty at score >= 0.1
    assert filt.mean() < 0.7 * raw.mean()                 # geometric filters remove a large share
    assert 5 <= shown.mean() <= 30 and tracks[-1] <= 30   # reported set stays small (parked cars, buses, people)
    if viz is not None:
        viz.close()
        from mcap.reader import make_reader
        with open(tmp_path / "out.mcap", "rb") as f:
            topics = {ch.topic for _, ch, _ in make_reader(f).iter_messages()}
        assert {"/lidar", "/detections", "/tracks", "/pose"} <= topics


def _replay_kygi(bag, detector=None, kygi=True, max_scans=None):
    """Stream a kygi665 bag through the pipeline. ``kygi=False`` reproduces the
    old conventions (gyro read as rad/s, lidar axes = body axes)."""
    from pathlib import Path

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    from dog_perception.geometry import make_T, rot_zyx, transform_points
    from dog_perception.inspvax import KYGI_LIDAR_RPY, EnuOrigin, gyro_from_kygi_imu, pose_from_inspvax, register_kygi
    from dog_perception.pipeline import PerceptionPipeline
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig
    from dog_perception.ros_utils import scan_from_msg, stamp_of

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
                g = pose_from_inspvax(m, origin) if origin is not None else None
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
                        base_height=h, auto_ground=True, auto_level=kygi)), detector, predictor=None)
                    for e in pending:
                        (pipe.on_imu if e[0] == "imu" else pipe.on_odometry)(*e[1:])
                if detector is None:
                    out = pipe.pre.process(scan)
                else:
                    out = pipe.on_scan(scan)
                if out is not None:
                    outs.append(out)
                continue
            if pipe is None:
                pending.append(ev)
            else:
                (pipe.on_imu if ev[0] == "imu" else pipe.on_odometry)(*ev[1:])
    return outs


def _sweep_alignment(points):
    """Median distance from the oldest sweep's structure points to the newest sweep."""
    from scipy.spatial import cKDTree
    r = np.hypot(points[:, 0], points[:, 1])
    m = (points[:, 2] > 0.3) & (r > 2.0) & (r < 30.0)
    dt = points[:, 4]
    new, old = points[m & (dt == 0)], points[m & (dt > dt.max() - 0.05)]
    return float(np.median(cKDTree(new[:, :3]).query(old[:, :3])[0]))


def test_moving_bag_sweeps_align():
    """Robot driving at ~1.5 m/s: with the measured kygi conventions (deg/s gyro,
    180 deg lidar yaw) the 5 accumulated sweeps line up and the ground is level;
    with the old ones the oldest sweep sits ~0.4 m off."""
    if not os.path.isdir(MOVING_BAG):
        pytest.skip("moving sample bag not in the checkout")
    pytest.importorskip("rosbags")
    from dog_perception.preprocess.frames import fit_ground_slope

    def score(kygi):
        frames = _replay_kygi(MOVING_BAG, kygi=kygi)
        full = [f for f in frames if f.points[:, 4].max() > 0.35]
        align = np.median([_sweep_alignment(f.points) for f in full])
        tilts = []
        for f in full[5:]:           # after the levelling EMA settled
            cur = f.points[f.points[:, 4] == 0]
            r = np.hypot(cur[:, 0], cur[:, 1])
            a, b = fit_ground_slope(cur[(r > 2) & (r < 20) & (np.abs(cur[:, 2]) < 0.3), :3])
            tilts.append(np.degrees(np.hypot(np.arctan(a), np.arctan(b))))
        return align, float(np.median(tilts))

    align, tilt = score(True)
    old_align, old_tilt = score(False)
    assert align < 0.15 and old_align > 2 * align
    assert tilt < 0.6 and old_tilt > 1.0


@requires_det3d
def test_moving_bag_static_scene_stays_static():
    """Most of what the detector sees is parked; in the world frame its speed
    must be far below the robot's 1.5 m/s."""
    if not (os.path.isdir(MOVING_BAG) and os.path.exists(PP_CKPT)):
        pytest.skip("moving sample bag / weights not in the checkout")
    pytest.importorskip("rosbags")
    from dog_perception.detection import build_detector

    det = build_detector("pillar", checkpoint=PP_CKPT, device="cpu")
    outs = _replay_kygi(MOVING_BAG, detector=det)
    assert len(outs) >= 25
    det_speed = np.concatenate([np.hypot(*o.detections.above(DEFAULT_SCORE_THRESHOLDS).velocities.T) for o in outs])
    trk_speed = [np.hypot(*t.velocity) for o in outs for t in o.tracks if not t.coasting and t.hits >= 5]
    assert np.median(det_speed) < 0.3 and np.median(trk_speed) < 0.5
    long_lived = {t.track_id for t in outs[-1].tracks if t.hits >= 20}
    assert len(long_lived) >= 3            # parked cars keep their ids through the drive
