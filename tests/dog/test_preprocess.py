import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from dog_perception.geometry import inv_T, make_T, rot_z, so3_exp, so3_log, transform_points, yaw_of
from dog_perception.pose_buffer import ImuPropagator, PoseBuffer, PoseOutOfRange
from dog_perception.preprocess import (DetFrameConfig, GroundHeightEstimator, SelfFilter, SelfFilterConfig,
                                       SweepAccumulator, deskew_points, det_frame_from_body, neighbor_counts)
from dog_perception.sim import GaitModel, make_scenario

WALL_Y = 13.85


def test_so3_roundtrip():
    rng = np.random.default_rng(0)
    rv = rng.normal(0, 1, (50, 3))
    rv[0] = 0.0
    rv[1] = 1e-12
    R = so3_exp(rv)
    assert np.allclose(R, Rotation.from_rotvec(rv).as_matrix(), atol=1e-9)
    small = np.linalg.norm(rv, axis=1) < 3.0
    assert np.allclose(so3_log(R)[small], rv[small], atol=1e-6)
    assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_pose_buffer_matches_scipy_slerp():
    rng = np.random.default_rng(1)
    ts = np.sort(rng.uniform(0, 1, 20))
    rots = Rotation.from_rotvec(rng.normal(0, 0.5, (20, 3)))
    ps = rng.normal(0, 1, (20, 3))
    buf = PoseBuffer(max_age=10)
    for t, R, p in zip(ts, rots.as_matrix(), ps):
        buf.add(t, R, p)
    q = rng.uniform(ts[0], ts[-1], 100)
    R_i, p_i = buf.interpolate(q)
    ref = Slerp(ts, rots)(q).as_matrix()
    assert np.allclose(R_i, ref, atol=1e-9)
    assert np.allclose(p_i[:, 0], np.interp(q, ts, ps[:, 0]))
    with pytest.raises(PoseOutOfRange):
        buf.interpolate([ts[-1] + 1.0])
    assert not buf.add(ts[3], np.eye(3), np.zeros(3))   # out of order ignored


def test_imu_propagation_tracks_gait():
    sim = make_scenario(seed=0, duration=2.0, gait=GaitModel(speed=1.5, yaw_rate=0.5, pitch_amp_deg=8))
    buf = PoseBuffer()
    prop = ImuPropagator(buf)
    imu_t, imu_w = sim.imu(0.0, 1.0, gyro_noise=0.0)
    events = [(t, 0, w) for t, w in zip(imu_t, imu_w)] + [(t + 0.03, 1, t) for t in np.arange(0, 1.0, 0.1)]
    events.sort(key=lambda e: (e[0], e[1]))
    for te, kind, v in events:
        if kind == 0:
            prop.on_imu(te, v)
        else:
            R, p, vel = sim.odometry(v)
            prop.on_odometry(v, R, p, vel)
    q = np.linspace(0.2, 0.95, 50)
    R_est, _ = buf.interpolate(q)
    R_true, _ = sim.robot.pose(q)
    err = np.linalg.norm(so3_log(np.einsum("nji,njk->nik", R_true, R_est)), axis=1)
    assert np.rad2deg(err.max()) < 0.2


def test_deskew_flattens_wall(sim_frames):
    sim, buf = sim_frames["sim"], sim_frames["buffer"]
    scan = sim_frames["scans"][4]
    xyz_b = deskew_points(scan.xyz, scan.point_times, buf, scan.stamp, sim.T_bl)
    T_wb = buf.pose_at(scan.stamp)
    w_d = transform_points(T_wb, xyz_b)
    w_r = transform_points(T_wb @ sim.T_bl, scan.xyz)
    m = (np.abs(np.abs(w_d[:, 1]) - WALL_Y) < 0.6) & (w_d[:, 2] > 0.3) & (w_d[:, 2] < 2.5)
    rms_d = np.sqrt(np.mean((np.abs(w_d[m, 1]) - WALL_Y) ** 2))
    rms_r = np.sqrt(np.mean((np.abs(w_r[m, 1]) - WALL_Y) ** 2))
    assert m.sum() > 500
    assert rms_d < 0.02 < 0.05 < rms_r
    # exact per-point path agrees with the binned default
    exact = deskew_points(scan.xyz[:2000], scan.point_times[:2000], buf, scan.stamp, sim.T_bl, time_resolution=0)
    assert np.abs(exact - xyz_b[:2000]).max() < 2e-3


def test_neighbor_counts_brute_force():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 2, (400, 3))
    r = 0.25
    cnt = neighbor_counts(pts, r)
    keys = np.floor(pts / r).astype(int)
    brute = np.array([np.sum(np.all(np.abs(keys - k) <= 1, axis=1)) - 1 for k in keys])
    assert np.array_equal(cnt, brute)


def test_self_filter_removes_body_and_spray():
    T_bl = make_T(t=[0.2, 0.0, 0.15])
    f = SelfFilter(SelfFilterConfig(body_boxes=((-0.6, 0.6, -0.4, 0.4, -0.8, 0.3),), outlier_min_neighbors=2), T_bl)
    rng = np.random.default_rng(0)
    body_pts = rng.uniform([-0.7, -0.3, -0.5], [0.3, 0.3, 0.1], (200, 3))       # lidar frame, inside the box
    wall = np.stack([np.full(500, 2.0), rng.uniform(-1, 1, 500), rng.uniform(-0.3, 1, 500)], 1)
    spray = np.array([[1.2, 1.2, 0.5], [-1.5, 1.0, -0.2]])
    far = np.array([[30.0, 0.0, 0.0]])
    keep = f(np.vstack([body_pts, wall, spray, far]))
    assert not keep[:200].any()
    assert keep[200:700].mean() > 0.99
    assert not keep[700:702].any()
    assert keep[-1]          # isolated far points are never removed by the near-field filter


def test_det_frame_is_gravity_aligned_and_follows_yaw():
    R = Rotation.from_euler("ZYX", [0.7, 0.15, -0.1]).as_matrix()
    T = make_T(R, [3.0, -2.0, 0.5])
    T_wd = det_frame_from_body(T, 0.45)
    assert np.allclose(T_wd[:3, 2], [0, 0, 1])
    assert np.isclose(yaw_of(T_wd[:3, :3]), yaw_of(R))
    assert np.allclose(T_wd[:3, 3], [3.0, -2.0, 0.05])


def test_accumulator_aligns_static_points_and_dt():
    acc = SweepAccumulator(num_sweeps=3, max_time_span=0.25)
    pt_w = np.array([[5.0, 1.0, 0.5]])
    for k, t in enumerate([0.0, 0.1, 0.2, 0.3]):
        acc.push(t, pt_w, [[float(k)]])
    T_wd = make_T(rot_z(0.4), [1.0, 2.0, 0.0])
    out = acc.build(T_wd, 0.3)
    assert out.shape == (3, 5)                               # oldest sweep dropped (deque) and span limit
    assert np.allclose(out[:, :3], transform_points(inv_T(T_wd), pt_w), atol=1e-5)
    assert np.allclose(out[:, 4], [0.0, 0.1, 0.2], atol=1e-6)   # newest first
    assert np.allclose(out[:, 3], [3.0, 2.0, 1.0])


def test_preprocessed_frame_is_level(sim_frames):
    fr = sim_frames["frames"][-1]
    assert fr.points.shape[1] == 5
    assert np.allclose(np.unique(fr.points[:, 4]), [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-4)
    cur = fr.points[fr.points[:, 4] == 0]
    r = np.hypot(cur[:, 0], cur[:, 1])
    g = (r > 3) & (r < 15) & (np.abs(cur[:, 2]) < 0.3)
    z = cur[g, 2]
    mad = np.median(np.abs(z - np.median(z)))       # robust: object feet also fall in the band
    assert mad < 0.02 and abs(np.median(z)) < 0.05
    # payload self-hits removed
    assert sim_frames["infos"][-1]["self_hits"] > 1000
    assert fr.num_kept < fr.num_raw


def test_ground_estimator_converges():
    est = GroundHeightEstimator(DetFrameConfig(auto_ground=True, ground_alpha=0.5))
    rng = np.random.default_rng(0)
    for _ in range(20):
        xy = rng.uniform(-8, 8, (3000, 2))
        z = np.full(3000, 0.12 - est.offset) + rng.normal(0, 0.01, 3000)
        est.update(np.column_stack([xy, z]))
    assert abs(est.offset - 0.12) < 0.01


def test_imu_propagator_ignores_stale_odometry():
    buf = PoseBuffer()
    prop = ImuPropagator(buf)
    prop.on_odometry(1.0, np.eye(3), [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    for t in np.arange(1.0025, 1.1, 0.0025):
        prop.on_imu(t, [0.0, 0.0, 0.1])
    before = buf.pose_at(1.09)
    prop.on_odometry(0.9, np.eye(3), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])   # late, older message
    assert np.allclose(buf.pose_at(1.09), before)
