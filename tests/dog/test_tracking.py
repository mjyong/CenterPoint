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
    # the predictors skip the mixed rollout; the per-mode part must not change
    r2 = imm.rollout(3.0, 0.5, mixed=False)
    assert "mean" not in r2
    for k in ("mode_means", "mode_covs", "mode_probs", "times"):
        assert np.array_equal(r[k], r2[k])
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


def _vehicles(rows):
    """rows: (x, y, length, width, yaw, score). Class 0 is vehicle."""
    boxes = np.array([[x, y, 1.5, length, width, 4.0, yaw] for x, y, length, width, yaw, score in rows])
    scores = np.array([score for *_, score in rows])
    n = len(rows)
    return Detections(boxes, np.zeros((n, 2)), scores, np.zeros(n, dtype=np.int64), "world")


def test_long_vehicle_two_boxes_stay_one_track():
    """A 16 m bus seen as two boxes a few metres apart must not open a second track."""
    trk = MultiObjectTracker(TrackerConfig())
    one = _vehicles([(0.0, 0.0, 16.0, 3.0, 0.0, 0.8)])
    trk.step(one, 0.1)
    trk.step(one, 0.2)                         # confirm the track
    two = _vehicles([(0.0, 0.0, 16.0, 3.0, 0.0, 0.8), (3.0, 0.0, 16.0, 3.0, 0.0, 0.5)])
    for k in range(10):
        out = trk.step(two, 0.3 + 0.1 * k)
    assert [s.track_id for s in out] == [1]
    assert len(trk.tracks) == 1
    # the same pair on the first frame, before any track exists, also stays one track
    born = MultiObjectTracker(TrackerConfig())
    born.step(two, 0.1)
    assert len(born.tracks) == 1


def test_vehicles_side_by_side_stay_two_tracks():
    trk = MultiObjectTracker(TrackerConfig())
    side = _vehicles([(0.0, 0.0, 4.5, 1.8, 0.0, 0.8), (0.0, 4.0, 4.5, 1.8, 0.0, 0.8)])
    trk.step(side, 0.1)
    out = trk.step(side, 0.2)
    assert len(out) == 2


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


def _drive(trk, xs, t0, dt, vx, rng, pos_noise=0.15, vel_noise=0.25):
    """One vehicle at x = xs[k] with velocity-head reading vx[k] (+ noise); returns the reported states."""
    out = []
    for k, x in enumerate(xs):
        box = [(x + rng.normal(0, pos_noise), 6.0 + rng.normal(0, pos_noise), 4.5, 1.9, 0.0, 0.8)]
        dets = _vehicles(box)
        dets.velocities = np.array([[vx[k] + rng.normal(0, vel_noise), rng.normal(0, vel_noise)]])
        out.append(trk.step(dets, t0 + dt * k))
    return out


def test_motion_state_parked_car_stays_static():
    """A parked car is never reported moving: its box drifts 0.3 m/s (the visible side changes as the
    robot walks past) and once slides 1.2 m in 0.6 s (the box jumps along a long vehicle) while the
    velocity head reads ~0; the velocity head reads 12 m/s in its first frames (a sample bag's start)
    and 1.2 m/s for 0.5 s later (a pose glitch) while the box stays put. Either signal alone would
    call it moving."""
    rng = np.random.default_rng(0)
    t = np.arange(150) * 0.1
    x = 10 + 0.3 * t + np.clip((t - 10) * 2.0, 0, 1.2)
    head = np.where((t >= 5) & (t < 5.5), 1.2, 0.0)
    head[:3] = [12.0, 8.0, 5.0]
    out = _drive(MultiObjectTracker(TrackerConfig()), x, 0.0, 0.1, head, rng, vel_noise=0.1)
    assert all(s[0].motion == "static" for s in out[1:])
    displacement_only = _drive(MultiObjectTracker(TrackerConfig(use_velocity=False)), x, 0.0, 0.1, head,
                               np.random.default_rng(0), vel_noise=0.1)
    assert any(s[0].motion == "moving" for s in displacement_only[1:])


def test_motion_state_start_stop_and_forget():
    rng = np.random.default_rng(1)
    trk = MultiObjectTracker(TrackerConfig(stop_memory=3.0))
    t = np.arange(150) * 0.1
    v = np.where((t >= 3) & (t < 8), 3.0, 0.0)              # drives 3 m/s from 3 s to 8 s
    x = 10 + np.cumsum(v) * 0.1
    out = _drive(trk, x, 0.0, 0.1, v, rng)
    motion = np.array([s[0].motion if s else "static" for s in out])       # confirmed from the 2nd frame
    t_moving = t[np.argmax(motion == "moving")]
    assert (motion[t < 3] == "static").all() and 3.0 < t_moving <= 3.5
    t_stopped = t[(t > 8) & (motion == "stopped")][0]
    assert 8.5 < t_stopped <= 9.6
    assert (motion[(t > t_stopped) & (t < t_stopped + 2)] == "stopped").all()   # may go again: not static yet
    assert (motion[t > 12] == "static").all() and out[-1][0].time_since_moving > 3
    # a car already driving when first seen is reported moving within 0.5 s
    drive = _drive(MultiObjectTracker(TrackerConfig()), 10 + 3.0 * t[:20], 0.0, 0.1, np.full(20, 3.0),
                   np.random.default_rng(2))
    assert [s[0].motion for s in drive[1:]].index("moving") + 1 <= 5


def test_motion_eval_counts_false_moving_and_latency():
    from dog_perception.tracking.motion_eval import evaluate_motion
    rng = np.random.default_rng(2)
    t = np.arange(100) * 0.1
    v = np.where(t >= 4, 3.0, 0.0)
    x = 10 + np.cumsum(v) * 0.1
    frames = []
    for k in range(100):
        creep = -40 + 0.5 * t[k]                 # a truck creeping 0.5 m/s, 40 m away: moving, but not relevant
        dets = _vehicles([(x[k] + rng.normal(0, 0.1), 6.0, 4.5, 1.9, 0.0, 0.8),
                          (30 + rng.normal(0, 0.1), -8.0, 4.5, 1.9, 0.0, 0.8),
                          (creep, 10.0, 12.0, 2.6, 0.0, 0.8)])
        dets.velocities = np.array([[v[k], 0.0], [0.0, 0.0], [0.5, 0.0]])
        gt = [dict(key="mover", label=0, xy=np.array([x[k], 6.0]), moving=bool(v[k] > 0), speed=v[k], range=12.0),
              dict(key="parked", label=0, xy=np.array([30.0, -8.0]), moving=False, speed=0.0, range=31.0),
              dict(key="creeper", label=0, xy=np.array([creep, 10.0]), moving=True, speed=0.5, range=40.0)]
        frames.append((t[k], dets, gt))
    m = evaluate_motion([frames])
    assert m["static_tracks"] == 1 and m["static_flagged"] == 0 and m["false_moving"] < 0.05
    assert m["segments"] == 2 and m["missed"] == 1 and m["latency_median"] <= 0.5
    assert m["relevant_segments"] == 1 and m["relevant_missed"] == 0 and m["relevant_latency_p90"] <= 0.5
    assert [(d["kind"], d["key"]) for d in m["details"]] == [("missed", "creeper")]
    assert abs(m["recording_h"] * 3600 - 10.0) < 1e-6
