"""Evaluation metrics, the nuScenes clip loader (on a synthetic nuScenes-format
dataset) and the clip visualizer."""
import json
import os

import numpy as np
import pytest

from dog_perception.detection.evaluate import evaluate_detections, match_frame, scale_error
from dog_perception.tracking.evaluate import MotAccumulator


def test_mot_metrics_on_a_scripted_sequence():
    """GT a (5 frames) and b (5 frames); tracker: a gets T1 then T3 (one ID
    switch), b is missed in frame 2 (FN + fragmentation), one false track."""
    acc = MotAccumulator(["vehicle"])
    events = []
    for f in range(5):
        gt_ids, gt_xy = [1, 2], [[0.0, f], [10.0, f]]
        trk = [(1 if f < 3 else 3, [0.3, f])]
        if f != 2:
            trk.append((2, [10.0, f + 0.5]))
        if f == 4:
            trk.append((9, [30.0, 0.0]))
        ids, xy = zip(*trk)
        events.append(acc.update(gt_ids, gt_xy, [0, 0], list(ids), list(xy), [0] * len(ids)))
    r = acc.summary()["vehicle"]
    assert (r["TP"], r["FN"], r["FP"], r["IDSW"], r["FRAG"]) == (9, 1, 1, 1, 1)
    assert r["MOTA"] == pytest.approx(1 - 3 / 10)
    assert r["MOTP"] == pytest.approx((5 * 0.3 + 4 * 0.5) / 9)
    # IDF1: best map a<->T1 (3 frames), b<->T2 (4) -> IDTP 7; 10 GT and 10 track observations
    assert r["IDF1"] == pytest.approx(14 / (14 + 3 + 3))
    assert events[3]["idsw"] == [(1, 1, 3)] and events[2]["fn"] == [2] and events[4]["fp"] == [9]
    assert r["MT"] == 2 and r["ML"] == 0


def test_mot_keeps_previous_match_over_a_closer_track():
    """CLEAR-MOT continuity: an established match is kept while within the
    threshold, even if another track comes closer."""
    acc = MotAccumulator(["vehicle"])
    acc.update([1], [[0, 0]], [0], [5], [[0.5, 0]], [0])
    ev = acc.update([1], [[0, 0]], [0], [5, 6], [[1.5, 0], [0.1, 0]], [0, 0])
    assert ev["matches"][0][1] == 5 and ev["idsw"] == [] and ev["fp"] == [6]


def test_detection_error_metrics():
    assert scale_error([4, 2, 1.5], [4, 2, 1.5]) == pytest.approx(0.0)
    assert scale_error([4, 2, 1.0], [2, 2, 1.0]) == pytest.approx(0.5)
    gt = [dict(boxes=np.array([[10.0, 0, 0, 4, 2, 1.5, 0.0]]), labels=np.array([0]), velocities=np.array([[5.0, 0]]))]
    pred = [dict(boxes=np.array([[10.5, 0, 0, 4, 2, 1.5, 0.3], [30, 0, 0, 4, 2, 1.5, 0]]), labels=np.array([0, 0]),
                 scores=np.array([0.9, 0.5]), velocities=np.array([[4.0, 0], [0, 0]]))]
    r = evaluate_detections(pred, gt, ["vehicle"])["vehicle"]
    assert r["ATE"] == pytest.approx(0.5) and r["AOE"] == pytest.approx(0.3) and r["AVE"] == pytest.approx(1.0)
    assert r["AP@2.0"] == pytest.approx(1.0) and r["AP@0.5"] == pytest.approx(1.0)
    tp, gt_m = match_frame(pred[0], gt[0])
    assert list(tp) == [True, False] and list(gt_m) == [0]


def test_ignored_ground_truth_is_not_scored():
    """An ignored box is no miss, and a prediction on it (center within the threshold or inside the
    footprint) is no false positive; a prediction elsewhere still is."""
    gt = [dict(boxes=np.array([[10.0, 0, 0, 4, 2, 1.5, 0.0], [20.0, 5, 0, 12, 3, 4, 0.0]]), labels=np.array([0, 0]),
               ignore=np.array([False, True]))]
    hit = dict(boxes=np.array([[10.2, 0, 0, 4, 2, 1.5, 0]]), labels=np.array([0]), scores=np.array([0.5]))
    on_ign = dict(boxes=np.array([[24.5, 5.5, 0, 4, 2, 1.5, 0]]), labels=np.array([0]), scores=np.array([0.9]))
    far = dict(boxes=np.array([[-15.0, 0, 0, 4, 2, 1.5, 0]]), labels=np.array([0]), scores=np.array([0.9]))
    cat = lambda *ds: [{k: np.concatenate([d[k] for d in ds]) for k in ds[0]}]    # noqa: E731
    r = evaluate_detections(cat(hit, on_ign), gt, ["vehicle"])["vehicle"]
    assert r["num_gt"] == 1 and r["AP@0.5"] == pytest.approx(1.0) and r["AP@2.0"] == pytest.approx(1.0)
    r = evaluate_detections(cat(hit, far), gt, ["vehicle"])["vehicle"]
    assert r["AP@2.0"] == pytest.approx(0.5)            # the higher-scored false positive comes first
    r = evaluate_detections(cat(hit, on_ign), [dict(gt[0], ignore=np.array([False, False]))], ["vehicle"])["vehicle"]
    assert r["num_gt"] == 2 and r["AP@2.0"] < 0.5        # not ignored: a miss and a false positive


def test_threshold_sweep():
    """Raising the reporting threshold drops the low-scored false positive first, then a true one."""
    from dog_perception.detection.evaluate import threshold_sweep
    gt = [dict(boxes=np.array([[10.0, 0, 0, 4, 2, 1.5, 0.0], [30.0, 0, 0, 4, 2, 1.5, 0.0]]), labels=np.array([0, 0]))]
    pred = [dict(boxes=np.array([[10.2, 0, 0, 4, 2, 1.5, 0], [30.3, 0, 0, 4, 2, 1.5, 0], [-15.0, 0, 0, 4, 2, 1.5, 0]]),
                 labels=np.array([0, 0, 0]), scores=np.array([0.8, 0.4, 0.2]))]
    r = {x["threshold"]: x for x in threshold_sweep(pred, gt, 0, [0.1, 0.3, 0.5])}
    assert (r[0.1]["tp"], r[0.1]["fp"], r[0.1]["recall"]) == (2, 1, 1.0)
    assert (r[0.3]["tp"], r[0.3]["fp"], r[0.3]["precision"]) == (2, 0, 1.0)
    assert (r[0.5]["tp"], r[0.5]["recall"]) == (1, 0.5) and r[0.5]["f1"] == pytest.approx(2 / 3)
    near = threshold_sweep(pred, gt, 0, [0.1], rng_bin=(0, 20))[0]
    assert (near["n_gt"], near["tp"], near["fp"]) == (1, 1, 1)


def _random_frames(rng, n, miss=0.2, jitter=0.6, fp=1):
    """Ground truth within 40 m and a detector that misses some, jitters the rest and adds false positives."""
    preds, gts = [], []
    for _ in range(n):
        k = rng.integers(1, 6)
        xy = rng.uniform(-40, 40, (k, 2))
        gts.append(dict(boxes=np.c_[xy, np.zeros(k), np.tile([4.0, 2, 1.5, 0], (k, 1))], labels=np.zeros(k, np.int64)))
        keep = rng.random(k) > miss
        pxy = np.r_[xy[keep] + rng.normal(0, jitter, (keep.sum(), 2)), rng.uniform(-40, 40, (fp, 2))]
        m = len(pxy)
        preds.append(dict(boxes=np.c_[pxy, np.zeros(m), np.tile([4.0, 2, 1.5, 0], (m, 1))],
                          labels=np.zeros(m, np.int64), scores=rng.random(m)))
    return preds, gts


def test_bootstrap_over_recordings():
    """AP from per-frame matches equals the global matching; the same model twice differs by nothing,
    and a clearly better detector is ahead in (nearly) every resample."""
    from dog_perception.detection.evaluate import ap_of_frames, bootstrap_ap, frame_matches
    rng = np.random.default_rng(0)
    preds, gts = _random_frames(rng, 60)
    full = evaluate_detections(preds, gts, ["vehicle"])["vehicle"]
    for thr in (0.5, 2.0):
        assert ap_of_frames(frame_matches(preds, gts, 0, thr), np.arange(60)) == pytest.approx(full["AP@%.1f" % thr])
    rec = np.repeat(np.arange(10), 6)                                    # 10 recordings of 6 frames
    boot, val = bootstrap_ap({"a": (preds, gts, rec), "a2": (preds, gts, rec)}, 0, n=200)
    assert val["a"]["mAP"] == pytest.approx(full["mAP"])
    assert np.all(boot["a"]["mAP"] == boot["a2"]["mAP"]) and np.std(boot["a"]["mAP"]) > 0
    # a better detector on the same ground truth
    better = []
    for p, g in zip(preds, gts):
        k = len(g["labels"])
        better.append(dict(boxes=g["boxes"] + np.c_[rng.normal(0, 0.1, (k, 2)), np.zeros((k, 5))],
                           labels=g["labels"], scores=rng.uniform(0.5, 1, k)))
    boot, val = bootstrap_ap({"a": (preds, gts, rec), "b": (better, gts, rec)}, 0, n=200)
    assert val["b"]["mAP"] > val["a"]["mAP"] + 0.2 and np.mean(boot["b"]["mAP"] > boot["a"]["mAP"]) > 0.99


# ---------------------------------------------------------------- synthetic nuScenes
def _q_yaw(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def make_fake_nuscenes(root, n_key=4):
    """One scene: ego driving +x at 5 m/s with a slight turn, LIDAR_TOP at 20 Hz
    (keyframes every 0.5 s), one static wall and one car driving +y at 2 m/s.
    The lidar 'sees' the wall as points fixed in the world."""
    meta = os.path.join(root, "v1.0-mini")
    os.makedirs(meta)
    os.makedirs(os.path.join(root, "samples", "LIDAR_TOP"))
    os.makedirs(os.path.join(root, "sweeps", "LIDAR_TOP"))
    T0 = 1_600_000_000_000_000
    tables = {k: [] for k in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor",
                              "sample_annotation", "instance", "category", "attribute", "visibility", "log")}
    tables["sensor"].append(dict(token="s_lidar", channel="LIDAR_TOP", modality="lidar"))
    tables["calibrated_sensor"].append(dict(token="cs_lidar", sensor_token="s_lidar", translation=[0.9, 0.0, 1.84],
                                            rotation=_q_yaw(-np.pi / 2), camera_intrinsic=[]))
    tables["category"].append(dict(token="cat_car", name="vehicle.car", description=""))
    tables["category"].append(dict(token="cat_cone", name="movable_object.trafficcone", description=""))
    tables["visibility"].append(dict(token="4", level="v80-100", description=""))
    tables["attribute"].append(dict(token="att_moving", name="vehicle.moving", description=""))
    tables["log"].append(dict(token="log0", logfile="", vehicle="", date_captured="", location=""))
    tables["instance"].append(dict(token="inst_car", category_token="cat_car", nbr_annotations=n_key,
                                   first_annotation_token="ann_car_0", last_annotation_token="ann_car_%d" % (n_key - 1)))
    tables["instance"].append(dict(token="inst_cone", category_token="cat_cone", nbr_annotations=n_key,
                                   first_annotation_token="ann_cone_0", last_annotation_token="ann_cone_%d" % (n_key - 1)))
    wall = np.column_stack([np.full(200, 30.0), np.linspace(-10, 10, 200), np.full(200, 1.0)])   # world frame
    n_sweeps = 10 * (n_key - 1) + 1
    for k in range(n_sweeps):
        t = k * 0.05
        ts = T0 + int(round(t * 1e6))
        ego_xy, ego_yaw = np.array([5.0 * t, 0.0]), 0.1 * t
        tables["ego_pose"].append(dict(token="ep%d" % k, timestamp=ts, translation=[*ego_xy, 0.0],
                                       rotation=_q_yaw(ego_yaw)))
        key = k % 10 == 0
        if key:
            j = k // 10
            tables["sample"].append(dict(token="smp%d" % j, timestamp=ts, scene_token="sc0",
                                         prev="smp%d" % (j - 1) if j else "",
                                         next="smp%d" % (j + 1) if j < n_key - 1 else ""))
            for name, cat, xy, yaw, vel in (("car", "cat_car", [20.0, -5.0 + 2.0 * t], np.pi / 2, None),
                                            ("cone", "cat_cone", [15.0, 3.0], 0.0, None)):
                tables["sample_annotation"].append(dict(
                    token="ann_%s_%d" % (name, j), sample_token="smp%d" % j, instance_token="inst_" + name,
                    translation=[*xy, 0.8], size=[1.9, 4.5, 1.6], rotation=_q_yaw(yaw), num_lidar_pts=50,
                    num_radar_pts=0, visibility_token="4", attribute_tokens=["att_moving"] if name == "car" else [],
                    prev="ann_%s_%d" % (name, j - 1) if j else "",
                    next="ann_%s_%d" % (name, j + 1) if j < n_key - 1 else ""))
        # wall points in this sweep's lidar frame
        T_ge = np.eye(4)
        T_ge[:2, :2] = [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]]
        T_ge[:2, 3] = ego_xy
        T_el = np.eye(4)
        T_el[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]           # yaw -90 deg
        T_el[:3, 3] = [0.9, 0.0, 1.84]
        T_lg = np.linalg.inv(T_ge @ T_el)
        pts = wall @ T_lg[:3, :3].T + T_lg[:3, 3]
        data = np.column_stack([pts, np.full(len(pts), 10.0), np.zeros(len(pts))]).astype(np.float32)
        fn = "%s/LIDAR_TOP/sd%d.pcd.bin" % ("samples" if key else "sweeps", k)
        data.tofile(os.path.join(root, fn))
        tables["sample_data"].append(dict(token="sd%d" % k, sample_token="smp%d" % (k // 10 if key else (k // 10)),
                                          ego_pose_token="ep%d" % k, calibrated_sensor_token="cs_lidar",
                                          timestamp=ts, fileformat="pcd", is_key_frame=key, height=0, width=0,
                                          filename=fn, prev="sd%d" % (k - 1) if k else "",
                                          next="sd%d" % (k + 1) if k < n_sweeps - 1 else ""))
    tables["scene"].append(dict(token="sc0", name="scene-fake", description="synthetic", log_token="log0",
                                nbr_samples=n_key, first_sample_token="smp0", last_sample_token="smp%d" % (n_key - 1)))
    for name, rows in tables.items():
        with open(os.path.join(meta, name + ".json"), "w") as f:
            json.dump(rows, f)
    return wall


def test_nuscenes_clip_loader(tmp_path):
    from dog_perception.nuscenes_clip import NuscScene, NuscTables

    wall = make_fake_nuscenes(str(tmp_path))
    tb = NuscTables(str(tmp_path), "v1.0-mini")
    sc = NuscScene(tb, "scene-fake", hz=10, cameras=())
    stamps = [1e-6 * s["timestamp"] for s in sc.selected]
    assert np.allclose(np.diff(stamps), 0.1) and len(sc) == 16
    frames = list(sc.frames())
    f = frames[-1]
    assert f.is_key and f.points[:, 4].max() == pytest.approx(0.45)
    # every sweep's wall points land on the same world points: ego motion is compensated
    xyz_w = f.points[:, :3] @ f.T_global_D[:3, :3].T + f.T_global_D[:3, 3]
    d = np.min(np.linalg.norm(xyz_w[:, None, :2] - wall[None, :, :2], axis=2), axis=1)
    assert d.max() < 1e-3
    # D frame: ground at z = 0 (the lidar is 1.84 m up, the wall at z = 1 m)
    assert np.allclose(xyz_w[:, 2], 1.0, atol=1e-3)
    assert abs(f.points[:f.n_current, 2].mean() - 1.0) < 1e-3
    gt = f.gt
    assert sorted(gt.labels.tolist()) == [-1, 0]         # car -> vehicle, cone -> not evaluated
    car = int(np.nonzero(gt.labels == 0)[0][0])
    assert np.allclose(gt.velocities[car], [0.0, 2.0])
    assert gt.boxes[car, 3] == pytest.approx(4.5) and gt.boxes[car, 6] == pytest.approx(np.pi / 2)
    fut = sc.future(gt.ids[car], frames[0].stamp, [0.5, 1.0, 1.5])
    assert np.allclose(fut, [[20, -4], [20, -3], [20, -2]])
    assert sc.future(gt.ids[car], frames[0].stamp, [0.5, 1.0, 1.5, 2.0]) is None     # beyond the annotations
    boxes, labels, ids = sc.interpolated(frames[0].stamp + 0.25)
    assert np.allclose(boxes[labels == 0, :2], [[20.0, -4.5]])


def test_clip_visualizer_writes_all_layers(tmp_path):
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    from dog_perception.clip_viz import ClipVisualizer
    from dog_perception.detection.boxes import Detections
    from dog_perception.foxglove import McapWriter
    from dog_perception.prediction.kinematic import IMMPredictor
    from dog_perception.tracking import MultiObjectTracker

    w = McapWriter(str(tmp_path / "x.mcap"))
    viz = ClipVisualizer(w)
    trk = MultiObjectTracker()
    for k in range(4):
        t = 0.1 * k
        d = Detections(np.array([[10.0 + 0.5 * k, 0, 0.8, 4.5, 1.9, 1.6, 0.0]]), np.array([[5.0, 0.0]]),
                       np.array([0.9]), np.array([0]), "world", t)
        tracks = trk.step(d, t)
        preds = IMMPredictor()(trk)
        viz.transforms(t, [("map", "ego", np.eye(4))])
        viz.lidar(t, np.random.default_rng(0).normal(size=(100, 3)), np.ones(100))
        viz.ego(t, np.eye(4), 0.0)
        viz.ground_truth(t, [dict(box=np.array([10.0 + 0.5 * k, 0, 0.8, 4.5, 1.9, 1.6, 0]), id=1, label=0,
                                  status="tracked", vel=np.array([5.0, 0]), past=None, future=np.zeros((3, 3)),
                                  meta={"gt_id": 1})], True)
        viz.detections(t, d, np.array([True]), np.array([True]), {0: 1})
        viz.tracks(t, tracks, {})
        viz.predictions(t, preds, {s.track_id: 0.0 for s in tracks}, None)
        viz.events(t, dict(matches=[], fp=[], fn=[], idsw=[(1, 1, 2)]), {1: np.zeros(3)}, {})
        viz.metrics(t, dict(t=t, mota_cum=float("nan")))
    w.close()
    with open(tmp_path / "x.mcap", "rb") as f:
        topics = {ch.topic for _, ch, _ in make_reader(f).iter_messages()}
    assert {"/tf", "/lidar", "/ego", "/gt/boxes", "/gt/paths", "/det/boxes", "/track/boxes", "/track/history",
            "/pred/paths", "/events", "/metrics/frame", "/log"} <= topics


def test_moving_velocity_reads_the_velocity_head():
    """A head that reads a moving car at a tenth of its speed is told apart from one that reads it right;
    parked cars and ignored labels do not count."""
    from dog_perception.detection.evaluate import moving_velocity
    gts, good, slow = [], [], []
    for k in range(10):
        boxes = np.array([[10.0 + k, 0, 0.8, 4.5, 1.9, 1.6, 0], [5, 8, 0.8, 4.5, 1.9, 1.6, 0], [-9, 3, 0.8, 4, 2, 2, 0]])
        vel = np.array([[3.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        gts.append(dict(boxes=boxes, labels=np.zeros(3, np.int64), velocities=vel, ignore=np.array([0, 0, 1], bool)))
        for out, f in ((good, 1.0), (slow, 0.03)):
            out.append(dict(boxes=boxes[:2] + [0.3, 0, 0, 0, 0, 0, 0], labels=np.zeros(2, np.int64),
                            scores=np.array([0.8, 0.9]), velocities=vel[:2] * f))
    g = moving_velocity(good, gts, 0)
    s = moving_velocity(slow, gts, 0)
    assert g["labels"] == g["matched"] == 10 and g["speed_ratio"] == pytest.approx(1.0) and g["read_moving"] == 1.0
    assert s["speed_ratio"] == pytest.approx(0.03) and s["read_moving"] == 0.0 and s["error"] == pytest.approx(2.91)
