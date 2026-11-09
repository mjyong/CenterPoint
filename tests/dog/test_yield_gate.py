"""dog_perception/yield_gate.py (the robot's go / wait at a crossing) and yield_eval.py (offline evaluation
against labels). docs/PLANNING_INTEGRATION.md section 4."""
import numpy as np
import pytest

from dog_perception.detection.boxes import CLASSES, Detections
from dog_perception.yield_eval import (Occupancy, evaluate_path, gap_after, gap_both, gt_tracks, summarize,
                                       track_objects)
from dog_perception.yield_gate import YieldConfig, YieldGate, assess, required_sight_range, rollout

PATH = np.array([[0.0, 0.0], [0.0, 8.0]])          # the robot crosses 8 m along +y
KMH30 = 30 / 3.6


def car(x, y, vx, vy=0.0, oid=1, motion="moving", yaw=None, **kw):
    return dict(id=oid, label="vehicle", position=np.array([x, y]), velocity=np.array([vx, vy]),
                yaw=np.arctan2(vy, vx) if yaw is None else yaw, size=np.array([4.5, 1.9, 1.6]), motion=motion, **kw)


def test_a_car_coming_across_the_path_makes_the_robot_wait():
    a = assess(PATH, [car(-30.0, 4.0, KMH30)], 0.0, 0.0)          # reaches the path in 3.6 s; robot there 3.5-4.5 s
    assert [c["id"] for c in a.conflicts] == [1] and a.conflicts[0]["gap"] < 0
    assert 1 in a.approaching
    d = YieldGate().update(a, 0.0)
    assert not d.go and d.reason == "vehicle" and d.ids == [1]


def test_cars_that_passed_or_are_too_far_do_not():
    assert assess(PATH, [car(10.0, 4.0, KMH30)], 0.0, 0.0).conflicts == []             # past, going away
    assert assess(PATH, [car(-100.0, 4.0, KMH30)], 0.0, 0.0).conflicts == []           # 12 s away > horizon
    assert assess(PATH, [car(-30.0, 30.0, KMH30)], 0.0, 0.0).conflicts == []           # beyond the 8 m path
    assert assess(PATH, [dict(car(-30.0, 4.0, KMH30), label="pedestrian")], 0.0, 0.0).present == set()
    assert YieldGate().update(assess(PATH, [car(10.0, 4.0, KMH30)], 0.0, 0.0), 0.0).go


def test_gap_is_post_encroachment_time():
    # a car crossing at y = 6.5 m: its box grows to half-length 2.25 + 0.5 + 0.7 = 3.45 m, half-width 2.15 m,
    # so it covers s = 4.35 .. 8 m of the path; from x = -45.05 m its front reaches the path after 41.6 m
    cfg = YieldConfig(sigma_k=0.0)

    def gap(arrive):
        a = assess(PATH, [car(-45.05, 6.5, 41.6 / arrive)], 0.0, 0.0, cfg)
        return a.conflicts[0] if a.conflicts else None
    # there at 5.0 s, gone at 5.83 s; the robot (1 m long, 1 m/s) is on s = 5.4 m from 4.9 to 5.9 s
    assert gap(5.0)["gap"] == pytest.approx(-0.92, abs=0.1)
    # there at 10.0 s: 1.5 s after the robot left the far end (8.5 s), less than pet; beyond 8 s, still checked
    assert gap(10.0)["gap"] == pytest.approx(1.5, abs=0.1) and gap(10.0)["s"] == pytest.approx(8.0)
    assert gap(11.0) is None                                 # 2.5 s after


def test_waits_clear_time_after_the_car_and_until_it_has_passed():
    gate = YieldGate()
    go_at = None
    for k in range(120):                                   # 12 s at 10 Hz
        t = 0.1 * k
        d = gate.update(assess(PATH, [car(-30.0 + KMH30 * t, 4.0, KMH30)], t, t), t)
        if d.go:
            go_at = t
            break
    # the car leaves the corridor (x > 2.25 + 0.5 + 0.7 + 2 sigma) about 4 s in; then 2 s clear
    assert go_at is not None and 5.5 < go_at < 7.5, go_at
    a = assess(PATH, [car(-30.0 + KMH30 * go_at, 4.0, KMH30)], go_at, go_at)
    assert a.conflicts == [] and 1 not in a.approaching


def test_a_car_that_slows_but_has_not_passed_keeps_the_robot_waiting():
    gate = YieldGate()
    gate.update(assess(PATH, [car(-20.0, 4.0, KMH30)], 0.0, 0.0), 0.0)
    # 3 s later it crawls at 0.6 m/s 18 m away: no conflict within the horizon, but still coming
    a = assess(PATH, [car(-18.0, 4.0, 0.6)], 3.0, 3.0)
    assert a.conflicts == [] and 1 in a.approaching
    d = gate.update(a, 3.0)
    assert not d.go and d.reason == "not_passed"


def test_a_standing_car_facing_the_path_is_watched_for_three_seconds():
    standing = car(-6.0, 4.0, 0.0, motion="stopped", yaw=0.0)
    gate = YieldGate()
    ds = [gate.update(assess(PATH, [standing], t, t), t) for t in np.arange(0.0, 6.0, 0.1)]
    first_go = next(i for i, d in enumerate(ds) if d.go)
    assert ds[0].reason == "standing" and 0.1 * first_go == pytest.approx(5.0, abs=0.15)   # 3 s + 2 s clear
    # seen standing 5 s already (tracker): no wait
    assert YieldGate().update(assess(PATH, [dict(standing, standing_for=5.0)], 0.0, 0.0), 0.0).go
    # parked parallel to the path, 6 m to the side: not facing it
    assert assess(PATH, [car(-6.0, 4.0, 0.0, motion="static", yaw=np.pi / 2)], 0.0, 0.0).standing == []
    # it starts to move towards the path: a conflict again
    a = assess(PATH, [car(-6.0, 4.0, 2.0)], 4.0, 4.0)
    assert a.conflicts and a.standing == []


def test_stale_perception_waits_and_needs_clear_time_after():
    gate = YieldGate()
    assert gate.update(assess(PATH, [], 1.0, 0.5), 1.0).reason == "stale"
    assert gate.update(assess(PATH, [], 1.1, 1.1), 1.1).reason == "clearing"
    assert gate.update(assess(PATH, [], 3.1, 3.1), 3.1).go


def test_never_stops_in_front_of_a_car_once_in_its_lane():
    a = assess(PATH, [car(-12.0, 0.0, KMH30)], 0.0, 0.0)    # the robot stands where the car is going
    assert a.in_lane and a.conflicts
    assert YieldGate().update(a, 0.0, dog_moving=True).reason == "committed"
    assert not YieldGate().update(a, 0.0, dog_moving=False).go          # standing at the curb: wait
    assert not assess(PATH, [car(-30.0, 4.0, KMH30)], 0.0, 0.0).in_lane


def test_prediction_modes_below_min_prob_are_dropped():
    # most likely: straight on along y = -10 (never near the path); the other turns up into it
    times = np.arange(0.5, 3.01, 0.5)
    straight = np.column_stack([-20.0 + 5.0 * times, np.full(len(times), -10.0)])
    turn = np.column_stack([-20.0 + 5.0 * times, -10.0 + 4.0 * times])
    obj = car(-20.0, -10.0, 5.0, pred_times=times, pred_modes=np.stack([straight, turn]))
    assert assess(PATH, [dict(obj, pred_probs=np.array([0.95, 0.05]))], 0.0, 0.0).conflicts == []
    assert assess(PATH, [dict(obj, pred_probs=np.array([0.8, 0.2]))], 0.0, 0.0).conflicts


def test_rollout_moves_on_by_the_message_age_and_continues_at_constant_velocity():
    cfg = YieldConfig()
    (centers, head, sigma), = rollout(car(0.0, 0.0, 5.0), 10.2, 10.0, cfg)[0]
    assert centers[0] == pytest.approx([1.0, 0.0]) and centers[-1] == pytest.approx([1.0 + 5.0 * cfg.horizon, 0.0])
    assert head[0] == pytest.approx(0.0) and sigma[-1] > sigma[0]


def test_required_sight_range():
    # 8 m at 1 m/s (8.5 s to leave), 2 s gap, 1.1 s latency, 30 km/h: about 105 m
    assert required_sight_range(8.0, KMH30, 1.1) == pytest.approx(8.0 + KMH30 * 11.6)
    assert required_sight_range(8.0, KMH30, 1.1, YieldConfig(dog_speed=2.0, pet=1.0)) == pytest.approx(
        8.0 + KMH30 * (4.25 + 1.0 + 1.1))


# ------------------------------------------------------------------ evaluation
def test_gaps_from_a_label_occupancy():
    grid = np.arange(0.0, 10.0, 0.1)
    occ = np.zeros((2, len(grid)), dtype=bool)
    occ[1, 50:60] = True                                    # point 1 covered from 5.0 to 5.9 s by track 7
    o = Occupancy(grid, occ, np.where(occ, 7, -1))
    lo, hi = np.array([0.0, 2.0]), np.array([1.0, 3.0])     # the robot covers point 1 from 2 to 3 s
    assert gap_after(o, lo, hi) == (pytest.approx(2.0), 7)
    assert gap_after(o, lo + 3.0, hi + 3.0)[0] < 0          # robot there 5-6 s: both at once
    assert gap_after(o, lo + 4.0, hi + 4.0)[0] == np.inf    # the car left before the robot came
    assert gap_both(o, 4.0, lo + 3.0 + 3.5, hi + 3.0 + 3.5) == pytest.approx(8.5 - 5.9)   # right after it
    assert gap_both(o, 6.5, lo + 6.5, hi + 6.5) == np.inf   # the gate cannot see what passed before now


def _scene(detect=True, speed=5.0, seconds=40.0):
    """A car on y = 4 m (det frame: 4 m ahead of the robot) from x = -60 m, labels at 2 Hz, detections at 10 Hz."""
    labels, results = [], []
    for k in range(int(seconds * 10)):
        t = 100.0 + 0.1 * k
        x = -60.0 + speed * 0.1 * k
        box = np.array([x, 4.0, 0.8, 4.5, 1.9, 1.6, 0.0])
        dets = Detections(boxes=box[None] if detect else np.zeros((0, 7)),
                          velocities=np.array([[speed, 0.0]]) if detect else np.zeros((0, 2)),
                          scores=np.array([0.8]) if detect else np.zeros(0),
                          labels=np.array([CLASSES.index("vehicle")]) if detect else np.zeros(0, dtype=np.int64),
                          stamp=t)
        results.append(dict(stamp=t, T_world_det=np.eye(4), detections=dets))
        if k % 5 == 0:
            labels.append(dict(stamp=t, T_world_det=np.eye(4), objects=[
                dict(track_id=3, label="vehicle", box=box, velocity=np.array([speed, 0.0]))]))
    return dict(frames=labels), results


def test_evaluation_with_a_perfect_detector_is_safe_and_waits_for_the_car():
    labels, results = _scene()
    tracks = gt_tracks(labels)
    frames = track_objects(results, pred_horizon=3.0)
    eps, info = evaluate_path("s", "fwd", PATH, frames, tracks, labels["frames"][-1]["stamp"])
    m = summarize(eps)
    assert m["episodes"] >= 8 and m["unsafe"] == 0 and m["timeouts"] == 0, m
    assert any(e.go > e.arrive for e in eps if e.known and e.go is not None)     # some waited for the car
    assert all(e.go >= e.oracle - 0.11 for e in eps if e.known and e.go is not None and e.oracle is not None)
    assert info["parked_frac"] == 0.0 and info["crossing_s"] == pytest.approx(8.5)


def test_evaluation_catches_a_blind_robot():
    labels, results = _scene(detect=False)
    eps, _ = evaluate_path("s", "fwd", PATH, track_objects(results), gt_tracks(labels),
                           labels["frames"][-1]["stamp"])
    m = summarize(eps)
    assert m["unsafe"] > 0 and m["wait_max"] == 0.0, m      # never waits, walks in front of the car
    assert all(e.gap_track == 3 for e in eps if e.unsafe)
    why = [e.why for e in eps if e.unsafe][0]                # e.g. went at 5 s: the car 35 m to the left, 18 km/h
    assert why["labelled"] and not why["tracked"] and why["speed_kmh"] == pytest.approx(18.0, abs=0.1)
    assert 80.0 < why["bearing"] < 100.0 and 25.0 < why["range"] < 45.0


def test_the_command_line_tool(tmp_path, capsys):
    import os
    import pickle
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))
    import yield_eval as tool
    for name, detect in (("seen", True), ("blind", False)):
        labels, results = _scene(detect=detect)
        (tmp_path / name).mkdir()
        pickle.dump(labels, open(tmp_path / name / "labels_sustech2.pkl", "wb"))
        pickle.dump(results, open(tmp_path / name / "results.pkl", "wb"))
    pat = str(tmp_path / "{scene}")
    argv = ["--labels", pat + "/labels_sustech2.pkl", "--results", pat + "/results.pkl", "--fan", "4",
            "--details", str(tmp_path / "ep.csv"), "--scenes"]
    assert tool.main(argv + ["seen"]) == 0
    out = capsys.readouterr().out
    assert "| fan000 |" in out and "| all |" in out and "## Unsafe episodes (0)" in out
    assert "| 30 km/h | 105 m |" in out                     # 8 m at 1 m/s, as in the design doc
    rows = open(tmp_path / "ep.csv").read().splitlines()
    assert rows[0].startswith("scene,path,arrive") and len(rows) > 4 * 20
    assert tool.main(argv + ["blind", "missing"]) == 1      # walks in front of the car; a missing scene is skipped
    out = capsys.readouterr().out
    assert "missing: SKIPPED" in out and "fan000: arrived" in out and "NOT tracked" in out
