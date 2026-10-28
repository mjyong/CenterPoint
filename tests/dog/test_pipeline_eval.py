"""dog_perception/pipeline_eval.py and tools/dog/pipeline_eval.py on a scripted scene.

The robot walks along world +y at 0.5 m/s (det frame = world axes, shifted). Vehicles:
  1 / 5  one parked car at (8, 20), labelled as two tracks with no label for 3 s in between
  2      a car crossing at y = 30 from x = -30 at 5 m/s (t = 5 .. 17 s)
  3      a car coming down x = 2 towards the robot at 4 m/s from t = 10 s: within 5 m of it at t ~ 17 s
  9      an ignore box at (-12, 15)
The detector sees all of them (and the ignore box) and, for 2 s, a clutter box at (-15, 25).
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from dog_perception import pipeline_eval as pe
from dog_perception.detection.boxes import Detections
from dog_perception.detection.filters import DetectionFilterConfig, filter_detections, filter_reasons, \
    filter_measurements

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIZE = (4.5, 1.9, 1.6)


def _world_state(t):
    """[(label track id, x, y, vx, vy, moving)] at time t (world)."""
    out = [(1 if t < 12.0 else (None if t < 15.0 else 5), 8.0, 20.0, 0.0, 0.0, False)]   # unlabelled for 3 s
    if 5.0 <= t <= 17.0:
        out.append((2, -30.0 + 5.0 * (t - 5.0), 30.0, 5.0, 0.0, True))
    if 10.0 <= t <= 22.0:
        out.append((3, 2.0, 40.0 - 4.0 * (t - 10.0), 0.0, -4.0, True))
    out.append((9, -12.0, 15.0, 0.0, 0.0, False))
    return out


def make_scene(n=300, clutter=(8.0, 10.0)):
    rng = np.random.default_rng(0)
    rows, key, dense = [], [], []
    for k in range(n):
        t = 1000.0 + 0.1 * k
        rel = t - 1000.0
        T = np.eye(4)
        T[1, 3] = 0.5 * rel
        objs, boxes, vels = [], [], []
        for tid, x, y, vx, vy, moving in _world_state(rel):
            box = np.array([x, y - T[1, 3], 0.8, *SIZE, 0.0 if vx else (-np.pi / 2 if vy else 0.0)])
            if tid is not None:
                objs.append(dict(track_id=tid, label="vehicle", box=box, velocity=np.array([vx, vy]), moving=moving,
                                 ignore=tid == 9, points=100, points_all=300, interpolated=k % 5 != 0))
            boxes.append(box[:7] + np.r_[rng.normal(0, 0.15, 2), 0, 0, 0, 0, 0])
            vels.append([vx, vy] + rng.normal(0, 0.2, 2))
        if clutter[0] <= rel < clutter[1]:
            boxes.append(np.array([-15.0, 25.0 - T[1, 3], 0.8, *SIZE, 0.0]))
            vels.append([0.0, 0.0])
        n_det = len(boxes)
        dets = Detections(np.array(boxes), np.array(vels), np.full(n_det, 0.8), np.zeros(n_det, np.int64), "det", t)
        # the raw output also has a floating box the filters drop
        raw = Detections.concat([dets, Detections(np.array([[5.0, 5.0, 3.0, *SIZE, 0.0]]), np.zeros((1, 2)),
                                                  np.array([0.6]), np.zeros(1, np.int64), "det", t)])
        measured = (raw.boxes[:, 2] - raw.boxes[:, 5] / 2, np.full(len(raw), 50))
        rows.append(dict(stamp=t, T_world_det=T, T_world_body=T, pose_bad=False, detections=dets,
                         raw_detections=raw, raw_measured=measured))
        frame = dict(stamp=t, T_world_det=T, objects=objs)
        dense.append(frame)
        if k % 5 == 0:
            key.append(dict(frame, objects=[dict(o, interpolated=False) for o in objs]))
    return rows, dict(frames=key), dict(frames=dense)


@pytest.fixture(scope="module")
def scene():
    rows, key, dense = make_scene()
    return pe.load_scene("scene_a", rows, key, dense)


def test_filter_reasons_match_the_filters():
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-20, 20, (4000, 2)), rng.uniform(-0.3, -0.2, 4000), np.zeros(4000), np.zeros(4000)]
    car = np.c_[rng.uniform(4, 6, (300, 2)), rng.uniform(0.2, 1.4, 300), np.zeros(300), np.zeros(300)]
    pts = np.r_[pts, car]
    dets = Detections(np.array([[5, 5, 0.8, 4, 2, 1.6, 0], [5, 5, 0.8, 1, 1, 1.7, 0], [-10, 3, 2.5, 4, 2, 1.6, 0],
                                [12, -8, 0.8, 4, 2, 1.6, 0]]), np.zeros((4, 2)), np.array([0.9, 0.5, 0.6, 0.4]),
                      np.array([0, 2, 0, 0]))
    cfg = DetectionFilterConfig()
    why = filter_reasons(dets, pts, cfg)
    assert list(why) == ["", "duplicate", "floating", "few points"]
    kept = filter_detections(dets, pts, cfg)
    assert np.allclose(kept.boxes, dets.boxes[why == ""])
    assert list(filter_reasons(dets, None, cfg, measured=filter_measurements(dets, pts, cfg))) == list(why)


def test_stitch_joins_a_parked_car_and_a_continued_track():
    t = np.arange(0, 5, 0.5)
    tracks = {1: (t, np.tile([8.0, 20.0], (len(t), 1))),                     # parked
              5: (t + 10, np.tile([8.2, 20.1], (len(t), 1))),                # same place, 5 s later
              2: (t, np.c_[5.0 * t, np.full(len(t), 30.0)]),                 # moving
              7: (t + 5.5, np.c_[5.0 * (t + 5.5), np.full(len(t), 30.0)]),   # continues it after 1 s
              8: (t, np.c_[5.0 * t, np.full(len(t), 33.0)])}                 # beside it, same time: never one
    ids, merges = pe.stitch_gt(tracks)
    assert ids[5] == ids[1] == 1 and ids[7] == ids[2] == 2 and ids[8] == 8 and merges == 2


def test_overrides_and_grid():
    cfg = pe.apply_overrides(pe.EvalConfig(), {"tracker.vehicle.gate": 2.0, "tracker.high_score": 0.25,
                                               "tracker.motion_on_frames": 4, "filter.min_points.vehicle": 8,
                                               "predict.merge_dist": 0.5, "eval.hazard_radius": 8.0,
                                               "tracker.vehicle.models": "cv:1.0+ct:1.0:0.3"})
    assert cfg.tracker.class_params["vehicle"].gate == 2.0 and cfg.tracker.class_params["pedestrian"].gate == 1.5
    assert cfg.tracker.high_score["vehicle"] == 0.25 and cfg.tracker.high_score["cyclist"] == 0.4
    assert cfg.filters.min_points["vehicle"] == 8 and cfg.predictor["merge_dist"] == 0.5
    assert cfg.hazard_radius == 8.0 and cfg.tracker.motion_on_frames == 4
    assert cfg.tracker.class_params["vehicle"].models == (("cv", 1.0), ("ct", 1.0, 0.3))
    assert pe.EvalConfig().tracker.class_params["vehicle"].gate == 3.0          # the default is untouched
    with pytest.raises(KeyError):
        pe.apply_overrides(pe.EvalConfig(), {"tracker.nope": 1})
    grid = pe.parse_grid(["tracker.vehicle.gate=2,3", "tracker.matching=greedy,hungarian"])
    assert len(grid) == 4 and grid[0] == {"tracker.vehicle.gate": 2, "tracker.matching": "greedy"}
    assert pe.needs_clouds(pe.apply_overrides(pe.EvalConfig(), {"filter.box_margin": 0.2}))
    fixed_then_swept = pe.parse_grid(["tracker.high_score=0.3", "tracker.vehicle.gate=2", "tracker.vehicle.gate=3,4"])
    assert fixed_then_swept == [{"tracker.high_score": 0.3, "tracker.vehicle.gate": 3},
                                {"tracker.high_score": 0.3, "tracker.vehicle.gate": 4}]


def test_scene_metrics(scene):
    assert scene.merges == 1 and scene.gt_id[5] == 1
    assert scene.dense_id[3] == 3 and 9 in scene.dense_id                  # 10 Hz ids matched at keyframes
    res = pe.run_scene(scene, pe.EvalConfig())
    m = pe.summarize([res])
    assert res.keyframes == 60
    assert m["det_recall_near"] > 0.95 and m["det_precision_near"] < 1.0           # the clutter box
    p, r = m["det_precision_near"], m["det_recall_near"]
    assert m["det_f1_near"] == pytest.approx(2 * p * r / (p + r))
    assert m["MOTA"] > 0.85 and m["IDF1"] > 0.85 and res.mot["IDSW"] == 0
    assert res.fp_episodes >= 1 and any(i["kind"] == "false track" for i in res.issues)
    assert res.counts["fp_cause|nothing labelled nearby"] >= 1
    # cars 2 and 3 vanish at the end of their labels: the tracks coasting on are not false ones (only
    # the clutter track's coasting is)
    strict = pe.run_scene(scene, pe.apply_overrides(pe.EvalConfig(), {"eval.unlabelled_grace": 0.0}))
    key = "fp_cause|coasting after the vehicle was lost"
    assert strict.counts[key] >= res.counts[key] + 2 and strict.fp_episodes > res.fp_episodes
    assert m["motion_relevant_segments"] >= 1 and m["motion_relevant_missed"] == 0
    assert m["pred_samples_moving"] > 10 and m["pred_FDE3_moving"] < 3.0
    assert m["CV_FDE3_moving"] < 3.0
    # the 3 s error split along / across the labelled path; it is all there is to FDE on a straight path
    r3 = m["tables"]["prediction"]["moving"]["IMM"][3.0]
    assert r3["along"] <= r3["FDE1"] + 1e-9 and r3["across"] <= r3["FDE1"] + 1e-9
    assert np.hypot(r3["along"], r3["across"]) >= 0.7 * r3["FDE1"]
    assert "along" not in m["tables"]["prediction"]["all matched"]["IMM"][3.0]      # parked cars have no path
    # the training targets cut from this run's track log: a few cm off the labels (detections 0.15 m noisy)
    for source in pe.TARGET_SOURCES:
        assert m["target_%s_err3_moving" % source] < 0.15 and m["target_%s_kept_moving" % source] > 0.8
        assert m["tables"]["targets"][source]["static"]["by_h"][3.0]["mean"] < 0.15
    hz = m["tables"]["hazard"]
    assert len(hz) == 1 and hz[0]["gt"] == 3 and hz[0]["reported"] and hz[0]["lead"] > 0
    assert m["AP@2.0"] > 0.9
    # without stitching the parked car would be two ground-truth tracks
    raw = pe.load_scene("scene_a", scene.rows, dict(frames=list(scene.keyframes.values())), max_gap=0.0,
                        same_place=0.0)
    assert raw.merges == 0


def test_sources_raw_and_labels(scene):
    r_raw = pe.run_scene(scene, pe.EvalConfig(), source="raw")
    r_res = pe.run_scene(scene, pe.EvalConfig(), source="results")
    assert pe.summarize([r_raw])["MOTA"] == pytest.approx(pe.summarize([r_res])["MOTA"])   # floating box dropped
    loose = pe.apply_overrides(pe.EvalConfig(), {"filter.max_float": 5.0})
    assert pe.summarize([pe.run_scene(scene, loose, source="raw")])["det_fp_per_frame"] > \
        pe.summarize([r_raw])["det_fp_per_frame"]
    oracle = pe.summarize([pe.run_scene(scene, pe.EvalConfig(), source="labels")])
    assert oracle["det_precision_near"] == 1.0 and oracle["MOTA"] > pe.summarize([r_res])["MOTA"]   # no clutter
    assert oracle["target_measured_err3_moving"] < 1e-6 and oracle["pred_across3_moving"] < 1e-6   # labels in, labels out


def test_bootstrap_is_paired(scene):
    a = pe.run_scene(scene, pe.EvalConfig())
    b = pe.run_scene(scene, pe.apply_overrides(pe.EvalConfig(), {"tracker.high_score": 0.9}))   # nothing starts
    boot = pe.bootstrap({"a": [a, a], "b": [b, b]}, n=20)
    assert boot["a"]["IDF1"].shape == (20,) and (boot["a"]["IDF1"] > boot["b"]["IDF1"]).all()


def test_tool_writes_report_mcap_and_sweep(tmp_path):
    rows, key, dense = make_scene(n=120)
    for d in ("runs/scene_a/pillar", "labels/scene_a", "runs/scene_a/frames"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(tmp_path / "runs/scene_a/pillar/results.pkl", "wb"))
    pickle.dump(key, open(tmp_path / "labels/scene_a/labels_sustech2.pkl", "wb"))
    pickle.dump(dense, open(tmp_path / "labels/scene_a/labels_full2.pkl", "wb"))
    for r in rows[::7]:
        np.save(tmp_path / "runs/scene_a/frames" / ("%.6f.npy" % r["stamp"]),
                np.random.default_rng(0).random((200, 5)).astype(np.float32) * [40, 40, 2, 1, 0])
    tool = os.path.join(ROOT, "tools", "dog", "pipeline_eval.py")
    common = ["--scenes", "scene_a", "--results", str(tmp_path / "runs/{scene}/pillar/results.pkl"),
              "--labels", str(tmp_path / "labels/{scene}/labels_sustech2.pkl"),
              "--dense-labels", str(tmp_path / "labels/{scene}/labels_full2.pkl"),
              "--frames", str(tmp_path / "runs/{scene}/frames"), "--bootstrap", "20"]
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = tmp_path / "eval"
    subprocess.run([sys.executable, tool] + common + ["--out", str(out)], check=True, env=env, capture_output=True)
    report = (out / "report.md").read_text()
    assert "MOTA" in report and "IDF1" in report and "FDE" in report
    assert "## Training targets" in report and "| measured | moving |" in report and "| across |" in report
    assert (out / "issues.csv").exists() and (out / "report.json").exists() and (out / "scene_a_bev.png").exists()
    from mcap.reader import make_reader
    with open(out / "scene_a.mcap", "rb") as f:
        topics = {c.topic for c in make_reader(f).get_summary().channels.values()}
    assert {"/lidar", "/gt/boxes", "/det/boxes", "/track/boxes", "/pred/paths", "/blind", "/metrics/frame"} <= topics
    sweep = tmp_path / "sweep"
    subprocess.run([sys.executable, tool] + common + ["--out", str(sweep), "--set", "tracker.high_score=0.35,0.9",
                                                      "--detections", "raw"], check=True, env=env, capture_output=True)
    text = (sweep / "sweep.md").read_text()
    assert "tracker.high_score=0.9" in text and (sweep / "sweep.csv").exists()
    rows = tmp_path / "rows.json"
    rows.write_text('[{"tracker.high_score": 0.9}, {}]')
    subprocess.run([sys.executable, tool] + common + ["--out", str(tmp_path / "pair"), "--configs", str(rows),
                                                      "--set", "tracker.vehicle.gate=2.5"], check=True, env=env,
                   capture_output=True)
    text = (tmp_path / "pair" / "sweep.md").read_text()
    assert "tracker.vehicle.gate=2.5, tracker.high_score=0.9" in text and "| 2 | tracker.vehicle.gate=2.5 |" in text


def test_learned_predictor_and_track_logs(scene, tmp_path):
    """A tier-2 model (untrained here) predicts instead of the IMM; the track logs feed the trainer."""
    from dog_perception.prediction import SampleConfig, TrackLogger, build_samples
    from dog_perception.prediction.model import ModelConfig, TrajectoryPredictor, save_model
    path = str(tmp_path / "gru.pt")
    save_model(TrajectoryPredictor(ModelConfig()), path)
    imm_log = TrackLogger()
    imm = pe.run_scene(scene, pe.EvalConfig(), logger=imm_log)
    learned = pe.run_scene(scene, pe.apply_overrides(pe.EvalConfig(), {"eval.learned_model": path}))
    assert imm.predictor == "IMM" and learned.predictor == "learned"
    assert len(learned.pred) == len(imm.pred) > 10              # the same matched tracks get predictions
    m = pe.summarize([learned])
    assert np.isfinite(m["pred_FDE3_moving"]) and "learned" in m["tables"]["prediction"]["moving"]
    samples = build_samples(imm_log.to_dict(), SampleConfig())
    assert len(samples["future"]) > 50 and "imm_modes" in samples   # the online IMM is logged for comparison
    assert len(build_samples(imm_log.to_dict(), SampleConfig(targets="measured"))["future"]) == len(samples["future"])
    grid = pe.parse_grid(["eval.learned_model=," + path])
    assert grid == [{"eval.learned_model": ""}, {"eval.learned_model": path}]
