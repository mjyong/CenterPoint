import numpy as np

from dog_perception.detection import OracleDetector, OracleNoise
from dog_perception.detection.boxes import Detections
from dog_perception.geometry import inv_T, wrap_angle
from dog_perception.preprocess import det_frame_from_body
from dog_perception.sim import GaitModel, RobotTrajectory, SimObject
from dog_perception.tracking import IMM, CTModel, CVModel, MultiObjectTracker, TrackerConfig, greedy_match, hungarian_match


def test_ct_jacobian_matches_numerical():
    for w in (0.0, 1e-6, 0.4, -1.2):
        x = np.array([1.0, -2.0, 3.0, 1.5, w])
        F = CTModel.F(x, 0.3)
        num = np.zeros((5, 5))
        for i in range(5):
            e = np.zeros(5)
            e[i] = 1e-6
            num[:, i] = (CTModel.f(x + e, 0.3) - CTModel.f(x - e, 0.3)) / 2e-6
        assert np.allclose(F, num, atol=1e-5)


def test_imm_prefers_ct_on_turning_target():
    rng = np.random.default_rng(0)
    imm = IMM([CVModel(1.0), CTModel(1.0, 0.2)], 0.95)
    imm.initialize([0, 0, 5, 0, 0], np.diag([0.1, 0.1, 1, 1, 0.3]))
    H = np.zeros((2, 5))
    H[0, 0] = H[1, 1] = 1
    w, v, yaw, p = 0.5, 5.0, 0.0, np.zeros(2)
    for _ in range(40):
        yaw += w * 0.1
        p = p + 0.1 * v * np.array([np.cos(yaw), np.sin(yaw)])
        imm.predict(0.1)
        imm.update(p + rng.normal(0, 0.05, 2), H, np.eye(2) * 0.05 ** 2)
    assert imm.mu[1] > 0.8
    x, _ = imm.state
    assert abs(x[4] - w) < 0.1
    r = imm.rollout(3.0, 0.5)
    assert r["mode_means"].shape == (2, 6, 5) and r["mean"].shape == (6, 5)
    # CT mode keeps turning, CV mode goes straight
    heading = lambda m: np.arctan2(m[-1, 3], m[-1, 2])
    assert wrap_angle(heading(r["mode_means"][1]) - heading(r["mode_means"][0])) > 1.0


def test_matchers_agree_on_simple_case():
    cost = np.array([[0.1, 5.0, np.inf], [4.0, 0.2, 3.0], [np.inf, np.inf, np.inf]])
    g = greedy_match(cost, 1e9)
    h = hungarian_match(cost, 1e9)
    assert sorted(map(tuple, g)) == sorted(map(tuple, h)) == [(0, 0), (1, 1)]


def _det(xy, label=1, score=0.8, vel=(0.0, 0.0)):
    return Detections(np.array([[xy[0], xy[1], 0.9, 0.6, 0.6, 1.7, 0.0]]), np.array([vel]), np.array([score]),
                      np.array([label]), "world")


def test_lifecycle_coasting_and_low_score_recovery():
    trk = MultiObjectTracker(TrackerConfig())
    t, pos, ids = 0.0, np.array([0.0, 0.0]), []
    for k in range(40):
        t += 0.1
        pos = pos + np.array([0.1, 0.0])           # 1 m/s pedestrian
        if 10 <= k < 18:                           # 0.8 s occlusion -> coast
            d = Detections(frame="world")
        elif 18 <= k < 22:                         # comes back with low scores only
            d = _det(pos, score=0.2, vel=(1.0, 0.0))
        else:
            d = _det(pos, vel=(1.0, 0.0))
        out = trk.step(d, t)
        ids += [s.track_id for s in out]
        if 10 <= k < 18:
            assert len(out) == 1 and out[0].coasting
    assert set(ids) == {1}                         # one identity throughout
    s = trk.outputs()[0]
    assert abs(s.velocity[0] - 1.0) < 0.15 and abs(s.position[0] - pos[0]) < 0.2
    assert s.history.shape[1] == 6 and s.history[-1, 5] == 1.0


def test_low_score_detections_do_not_spawn_tracks_and_tentative_die():
    trk = MultiObjectTracker(TrackerConfig())
    for k in range(5):
        assert trk.step(_det((5.0, 5.0), score=0.2), 0.1 * (k + 1)) == []
    assert trk.tracks == []
    trk.step(_det((0.0, 0.0)), 1.0)                # single high-score blip
    for k in range(3):
        trk.step(Detections(frame="world"), 1.1 + 0.1 * k)
    assert trk.tracks == []


def test_static_pedestrian_static_in_world_while_robot_turns():
    """Tracking in the world frame: the robot spinning in place must not create motion."""
    robot = RobotTrajectory(GaitModel(speed=0.0, yaw_rate=1.0))
    ped = SimObject(0, "pedestrian", (6.0, 2.0), 0.0, [(0.0, 0.0, 0.0)], 10.0)
    oracle = OracleDetector(OracleNoise(false_positive_rate=0.0, yaw_flip_prob=0.0), seed=0)
    trk = MultiObjectTracker(TrackerConfig())
    for k in range(50):
        t = 0.1 * (k + 1)
        T_wd = det_frame_from_body(robot.T(t), 0.45)
        c, y, v = ped.state([t])
        gt = [dict(id=0, label="pedestrian", center=c[0], size=ped.size, yaw=y[0], velocity=v[0], num_points=50)]
        out = trk.step(oracle(gt, inv_T(T_wd), t, use_points=False).transform(T_wd, "world"), t)
    assert len(out) == 1
    assert np.linalg.norm(out[0].velocity) < 0.2
    assert np.linalg.norm(out[0].position[:2] - [6.0, 2.0]) < 0.15
