import numpy as np

from dog_perception.autolabel import auto_label
from dog_perception.detection import CLASSES, OracleDetector, OracleNoise
from dog_perception.detection.evaluate import evaluate_detections
from dog_perception.geometry import inv_T
from dog_perception.preprocess import det_frame_from_body
from dog_perception.sim import GaitModel, RobotTrajectory, SimObject


def test_auto_label_beats_raw_teacher():
    rng = np.random.default_rng(0)
    objs = [SimObject.random(i, "pedestrian", rng, 21.0, (0, 0), (15, 10)) for i in range(8)]
    robot = RobotTrajectory(GaitModel(speed=0.3))
    det = OracleDetector(OracleNoise(false_positive_rate=0.5, pos_std=0.15, yaw_flip_prob=0.1), seed=2)
    frames, gts = [], []
    for k in range(200):
        t = (k + 1) / 10
        T_wd = det_frame_from_body(robot.T(t), 0.45)
        gt = []
        for o in objs:
            c, y, v = o.state([t])
            gt.append(dict(id=o.id, label=o.label, center=c[0], size=o.size, yaw=float(y[0]), velocity=v[0], num_points=12))
        frames.append((t, det(gt, inv_T(T_wd), t).transform(T_wd, "world")))
        gts.append(dict(boxes=np.array([[*g["center"], *g["size"], g["yaw"]] for g in gt]),
                        labels=np.array([CLASSES.index(g["label"]) for g in gt]),
                        velocities=np.array([g["velocity"] for g in gt])))
    labels = auto_label(frames)
    raw = [dict(boxes=d.boxes, labels=d.labels, scores=d.scores, velocities=d.velocities) for _, d in frames]
    al = []
    for t, _ in frames:
        L = labels[round(t, 6)]
        al.append(dict(boxes=np.array([l["box"] for l in L]).reshape(-1, 7), labels=np.array([l["label"] for l in L], int),
                       scores=np.array([l["score"] for l in L]), velocities=np.array([l["velocity"] for l in L]).reshape(-1, 2)))
    r_raw = evaluate_detections(raw, gts, CLASSES)["pedestrian"]
    r_al = evaluate_detections(al, gts, CLASSES)["pedestrian"]
    assert r_al["AP@0.5"] > r_raw["AP@0.5"] + 0.2
    assert r_al["ATE"] < 0.7 * r_raw["ATE"] and r_al["AVE"] < 0.7 * r_raw["AVE"]


def _obs(t, xy, yaw, vel, score=0.6, label=0, size=(4.5, 1.9, 1.6)):
    return [dict(t=float(tk), box=np.array([*p, 0.8, *size, yk]), vel=np.asarray(v, float), score=score, label=label)
            for tk, p, yk, v in zip(t, xy, yaw, vel)]


def test_refine_track_static_and_moving():
    from dog_perception.autolabel import AutoLabelConfig, refine_track
    rng = np.random.default_rng(0)
    t = np.arange(30) * 0.1
    cfg = AutoLabelConfig()
    # parked car seen from a passing robot: the box center drifts 0.6 m in 3 s (visible side changes), the
    # teacher's velocity head says ~0; old labels gave it 0.2-0.6 m/s and a heading from that drift
    xy = np.c_[10 + 0.2 * t, 5 + 0.1 * t] + rng.normal(0, 0.05, (30, 2))
    out = refine_track(_obs(t, xy, np.full(30, 0.3), rng.normal(0, 0.1, (30, 2))), t, cfg)
    v = np.array([o["velocity"] for o in out])
    b = np.array([o["box"] for o in out])
    assert np.all(v == 0) and not any(o["moving"] for o in out)
    assert np.ptp(b[:, 0]) == 0 and np.ptp(b[:, 1]) == 0 and np.allclose(b[:, 6], 0.3, atol=0.05)
    # walking pedestrian (1.3 m/s along +y): keeps its velocity, heading from motion
    xy = np.c_[np.full(30, 3.0), 1.3 * t] + rng.normal(0, 0.05, (30, 2))
    out = refine_track(_obs(t, xy, np.zeros(30), np.tile([0, 1.3], (30, 1)), label=1, size=(0.6, 0.6, 1.7)), t, cfg)
    v = np.array([o["velocity"] for o in out])
    assert all(o["moving"] for o in out) and np.allclose(v.mean(0), [0, 1.3], atol=0.15)
    assert np.allclose([o["box"][6] for o in out], np.pi / 2, atol=0.15)
    # driving car whose best box points backwards: the 180 deg flip is resolved against the motion
    xy = np.c_[5.0 * t, np.zeros(30)]
    yaw = np.zeros(30)
    out = refine_track(_obs(t, xy, yaw, np.tile([5.0, 0], (30, 1)), score=0.5), t, cfg)
    assert np.allclose([o["box"][6] for o in out], 0.0, atol=0.05)
    obs = _obs(t, xy, np.where(np.arange(30) == 3, np.pi, 0.0), np.tile([5.0, 0], (30, 1)))
    obs[3]["score"] = 0.95
    assert np.allclose([o["box"][6] for o in refine_track(obs, t, cfg)[5:25]], 0.0, atol=0.05)


def test_refine_track_votes_vehicle_subclass():
    """A vehicle track's subclass (car / truck / bus head of the teacher) is a score-weighted vote over the
    observations that carry one; observations of another class do not count; no subclass at all -> -1."""
    from dog_perception.autolabel import AutoLabelConfig, refine_track
    from dog_perception.detection.boxes import SUBCLASSES
    t = np.arange(10) * 0.1
    xy = np.c_[np.full(10, 12.0), np.full(10, 4.0)]
    obs = _obs(t, xy, np.zeros(10), np.zeros((10, 2)), size=(9.0, 2.6, 3.2))
    subs = ["truck"] * 6 + ["car"] * 3 + ["pedestrian"]       # flickering teacher heads
    for o, s, sc in zip(obs, subs, [0.6] * 6 + [0.9] * 3 + [0.2]):
        o["subclass"], o["score"] = SUBCLASSES.index(s), sc
    obs[-1]["label"] = 1
    out = refine_track(obs, t, AutoLabelConfig())
    assert {o["label"] for o in out} == {0} and {o["subclass"] for o in out} == {SUBCLASSES.index("truck")}
    for o in obs:
        o["subclass"] = -1                  # teacher without vehicle subclasses (older 3-class dog model)
    assert {o["subclass"] for o in refine_track(obs, t, AutoLabelConfig())} == {-1}


def test_review_flag_trusts_tracks_seen_clearly_a_few_times():
    """Mean score below 0.5 alone no longer flags a track: 5 detections at 0.6 or more make it sure.
    Each label carries the teacher's score of its own frame (0 where interpolated)."""
    from dog_perception.autolabel import AutoLabelConfig, refine_track
    t = np.arange(40) * 0.1
    xy = np.c_[np.full(40, 30.0), np.full(40, 4.0)]
    scores = np.r_[np.full(34, 0.35), np.full(6, 0.7)]          # far most of the time, then clear
    obs = [o for k, o in enumerate(_obs(t, xy, np.zeros(40), np.zeros((40, 2)))) if k != 10]   # one miss
    for o, s in zip(obs, np.delete(scores, 10)):
        o["score"] = float(s)
    out = refine_track(obs, t, AutoLabelConfig())
    assert np.mean(np.delete(scores, 10)) < 0.5 and not any(o["review"] for o in out)
    assert {o["confident_frames"] for o in out} == {6}
    assert out[10]["interpolated"] and out[10]["det_score"] == 0.0 and out[39]["det_score"] == 0.7
    for o in obs[-2:]:
        o["score"] = 0.4                                         # only 4 clear detections left
    assert all(o["review"] for o in refine_track(obs, t, AutoLabelConfig()))


def test_short_moving_bursts_are_static_and_short_stops_are_moving():
    from dog_perception.autolabel import AutoLabelConfig, _clean_moving
    cfg = AutoLabelConfig()
    t = np.arange(100) * 0.1
    burst = np.zeros(100, bool)
    burst[96:] = True                                            # last 0.4 s of a parked truck's track
    assert not _clean_moving(t, burst, cfg).any()
    drive = np.zeros(100, bool)
    drive[20:60] = True                                          # 4 s drive with a 0.3 s hiccup
    drive[35:38] = False
    out = _clean_moving(t, drive, cfg)
    assert out[20:60].all() and not out[:20].any() and not out[60:].any()
    short = np.ones(8, bool)                                     # a car crossing the edge of view
    assert _clean_moving(t[:8], short, cfg).all()


def test_parked_truck_with_a_late_burst_is_pinned():
    """The burst kept the track from being pinned to one position before the clean-up."""
    from dog_perception.autolabel import AutoLabelConfig, refine_track
    rng = np.random.default_rng(1)
    t = np.arange(60) * 0.1
    xy = np.c_[np.full(60, 20.0), np.full(60, 6.0)] + rng.normal(0, 0.05, (60, 2))
    xy[-5:, 1] += np.arange(1, 6) * 0.25                         # box slides along the truck at the end
    vel = np.zeros((60, 2))
    vel[-5:] = [0, 0.8]
    out = refine_track(_obs(t, xy, np.zeros(60), vel, size=(12.0, 2.6, 3.5)), t, AutoLabelConfig())
    b = np.array([o["box"] for o in out])
    assert not any(o["moving"] for o in out) and np.ptp(b[:, 1]) == 0
