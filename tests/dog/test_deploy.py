"""Deployment tools (docs/DEPLOY_ORIN.md): the det3d-free voxelization, the model metadata, the exported
detector against the PyTorch one, the backend comparison and the engine / run_sequence plumbing.
TensorRT itself is only exercised where it is installed (the server, the Orin)."""
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from .conftest import ROOT, requires_det3d

PP_CFG = "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"


@requires_det3d
def test_voxelize_matches_det3d():
    from det3d.core.input.voxel_generator import VoxelGenerator
    from dog_perception.detection.voxelize import grid_size, voxelize

    rng = np.random.default_rng(0)
    vs, r = [0.2, 0.2, 8.0], [-40.8, -40.8, -5.0, 40.8, 40.8, 3.0]
    for n, max_voxels, max_points in ((40000, 30000, 20), (20000, 300, 20), (30000, 30000, 2)):
        pts = np.c_[rng.uniform(-45, 45, (n, 2)), rng.uniform(-6, 4, n), rng.random((n, 2))].astype(np.float32)
        pts[: n // 3, :2] = rng.normal(0, 1.5, (n // 3, 2))          # dense pillars: more than max_points
        ref = VoxelGenerator(vs, r, max_points, max_voxels).generate(pts)
        got = voxelize(pts, vs, r, max_points, max_voxels)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    assert grid_size(vs, r).tolist() == [408, 408, 1]
    empty = voxelize(np.zeros((0, 5), np.float32), vs, r, 20, 100)
    assert [a.shape for a in empty] == [(0, 20, 5), (0, 3), (0,)]


PP_CKPT = os.path.join(ROOT, "work_dirs", "pretrained", "nusc_pp.pth")


@pytest.fixture(scope="module")
def exported(sim_frames, tmp_path_factory):
    """The deployed nuScenes Pillar exported to ONNX with its metadata. Real weights: with random ones
    every score sits near the threshold, and float noise decides which boxes pass."""
    pytest.importorskip("onnxruntime")
    if not os.path.exists(PP_CKPT):
        pytest.skip("no %s" % PP_CKPT)
    sys.path.insert(0, os.path.join(ROOT, "tools", "dog"))
    from export_pillar_onnx import export
    from dog_perception.detection.centerpoint import CenterPointDetector
    from dog_perception.detection.model_meta import meta_from_detector

    det = CenterPointDetector(PP_CFG, PP_CKPT, device="cpu")
    out = str(tmp_path_factory.mktemp("deploy"))
    export(det, out, 20000)
    meta = meta_from_detector(det, 20000)
    meta.source = dict(config=PP_CFG, checkpoint=PP_CKPT, xy_range=40.8)
    meta.save(os.path.join(out, "model_meta.json"))
    return det, out


@pytest.mark.slow
@requires_det3d
def test_exported_onnx_detector_matches_pytorch(exported, sim_frames):
    from dog_perception.detection.exported import ExportedPillarDetector
    from dog_perception.detection.model_meta import ModelMeta

    det, out = exported
    meta = ModelMeta.load(os.path.join(out, "model_meta.json"))
    assert meta.grid_size == [408, 408, 1] and meta.max_pillars == 20000 and len(meta.tasks) == 6
    assert meta.rpn_outputs[:2] == ["task0_reg", "task0_height"] and meta.onnx["rpn"]["outputs"] == meta.rpn_outputs
    ex = ExportedPillarDetector(out, "onnx", keep_stages=True)
    for fr in sim_frames["frames"][-2:]:
        ref, got = det(fr.points, fr.stamp), ex(fr.points, fr.stamp)
        assert len(ref) == len(got) > 0 and len(det.last_raw) == len(ex.last_raw)
        a, b = np.argsort(-ref.scores), np.argsort(-got.scores)
        assert np.abs(ref.scores[a] - got.scores[b]).max() < 1e-4
        assert np.abs(ref.boxes[a] - got.boxes[b]).max() < 1e-3
        assert np.array_equal(ref.labels[a], got.labels[b]) and np.array_equal(ref.subclasses[a], got.subclasses[b])
    used, total = ex.last_pillars
    assert used == total > 0 and ex.last_stages["bev"].shape == (1, 64, 408, 408)
    assert {"voxelize_ms", "network_ms", "pfn_ms", "rpn_ms", "post_ms"} <= set(ex.last_timing)
    with pytest.raises(ValueError):                    # metadata of another format version
        bad = dict(json.load(open(os.path.join(out, "model_meta.json"))), version=0)
        json.dump(bad, open(os.path.join(out, "bad.json"), "w"))
        ModelMeta.load(os.path.join(out, "bad.json"))


@pytest.mark.slow
@requires_det3d
def test_compare_backends_and_pillar_stats_tools(exported, sim_frames, tmp_path):
    det, out = exported
    frames = tmp_path / "frames"
    frames.mkdir()
    for k, fr in enumerate(sim_frames["frames"][-2:]):
        np.save(frames / ("%d.npy" % k), fr.points)
    env = dict(os.environ, PYTHONPATH=ROOT)
    report = tmp_path / "cmp.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "compare_backends.py"), "--model-dir", out,
                    "--frames", str(frames / "*.npy"), "--backends", "onnx,onnx", "--out", str(report)],
                   check=True, env=env, capture_output=True)
    r = json.load(open(report))["backends"]["onnx#2"]
    assert r["gates"]["passed"] and r["stages"]["heads_rel"] == 0.0 and r["final"]["reported_match"] == 1.0
    assert r["final"]["reported_match_tol"] == 1.0 and sum(r["final"]["missed_by"].values()) == 0
    # a PyTorch reference has no intermediate tensors: stages are taken against the first exported backend
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "compare_backends.py"), "--model-dir", out,
                    "--frames", str(frames / "*.npy"), "--backends", "torch,onnx,onnx", "--out", str(report)],
                   check=True, env=env, capture_output=True)
    r = json.load(open(report))
    assert r["stage_reference"] == "onnx" and r["backends"]["onnx"]["stages"] == {}
    assert r["backends"]["onnx#2"]["stages"]["heads_rel"] == 0.0 and r["backends"]["onnx"]["gates"]["passed"]
    # across machines: one saves its detections, the other compares with them
    dump = tmp_path / "dumps"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "compare_backends.py"), "--model-dir", out,
                    "--frames", str(frames / "*.npy"), "--backends", "onnx", "--dump", str(dump)],
                   check=True, env=env, capture_output=True)
    assert len(list((dump / "onnx").glob("*.npz"))) == 2
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "compare_backends.py"), "--model-dir", out,
                    "--frames", str(frames / "*.npy"), "--backends", "dump:%s" % (dump / "onnx") + ",onnx",
                    "--out", str(report)], check=True, env=env, capture_output=True)
    r = json.load(open(report))["backends"]["onnx"]
    assert r["gates"]["passed"] and r["raw"]["matched"] == r["raw"]["n_ref"] > 0 and r["final"]["score_max"] == 0.0
    stats = tmp_path / "pillars.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "pillar_stats.py"), "--model-dir", out,
                    "--frames", str(frames / "*.npy"), "--out", str(stats)], check=True, env=env, capture_output=True)
    s = json.load(open(stats))
    assert s["frames"] == 2 and 0 < s["max"] <= s["suggested_max_pillars"] and s["frames_over_current"] == 0


def test_compare_classifies_reported_misses():
    """Misses / extras at the reporting threshold: near (score noise across it), filter, score, class, none."""
    from collections import defaultdict
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from compare_backends import det_stats, summarize_dets
    from dog_perception.detection.boxes import Detections

    def dets(xy, scores, labels):
        b = np.zeros((len(xy), 7))
        b[:, :2], b[:, 3:6] = xy, (4.0, 2.0, 1.5)
        return Detections(b, np.zeros((len(xy), 2)), np.asarray(scores, float), np.asarray(labels))

    xy = [[10.0 * k, 0.0] for k in range(6)]               # vehicles, reported from 0.4
    ref = dets(xy, [0.8, 0.41, 0.6, 0.7, 0.9, 0.39], [0] * 6)
    # 0 same; 1 just below (near); 2 far below (score); 3 now a cyclist (class); 4 gone (none);
    # 5 just above (near extra); and a new box at 100 m (none extra)
    got = dets(xy[:4] + [xy[5], [100.0, 0.0]], [0.8, 0.395, 0.2, 0.7, 0.405, 0.5], [0, 0, 0, 2, 0, 0])
    acc = defaultdict(int, score=[], center=[], yaw=[], size=[])
    det_stats(ref, got, 0.5, acc, tol=0.02)
    r = summarize_dets(acc)
    assert r["reported_ref"] == 5 and (r["reported_missed"], r["reported_extra"]) == (4, 3)
    assert r["missed_by"] == {"near": 1, "filter": 0, "score": 1, "class": 1, "none": 1}
    assert r["extra_by"] == {"near": 1, "filter": 0, "score": 0, "class": 1, "none": 1}
    assert r["reported_match"] == pytest.approx(1 - 7 / 5) and r["reported_match_tol"] == pytest.approx(1 - 5 / 5)

    # after the filters: box 4 was detected (got raw) and filtered; the new box at 100 m was in ref raw
    # at 0.45 and filtered there; box 2's raw partner at 0.2 is a score difference, not a filter flip
    got_raw = dets(xy[:5] + [xy[5], [100.0, 0.0]], [0.8, 0.395, 0.2, 0.7, 0.9, 0.405, 0.5], [0, 0, 0, 2, 0, 0, 0])
    ref_raw = dets(xy + [[100.0, 0.0]], [0.8, 0.41, 0.6, 0.7, 0.9, 0.39, 0.45], [0] * 7)
    acc = defaultdict(int, score=[], center=[], yaw=[], size=[])
    det_stats(ref, got, 0.5, acc, 0.02, ref_raw, got_raw)
    r = summarize_dets(acc)
    assert r["missed_by"] == {"near": 1, "filter": 1, "score": 1, "class": 1, "none": 0}
    assert r["extra_by"] == {"near": 1, "filter": 1, "score": 0, "class": 1, "none": 0}
    assert r["reported_match_tol"] == pytest.approx(1 - 3 / 5)


def test_engine_names_and_run_sequence_specs(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    sys.path.insert(0, os.path.join(ROOT, "tools", "dog"))
    from bench_engines import engine_paths, parse_shapes
    from build_engines import onnx_for
    from run_sequence import parse_exported
    from dog_perception.detection.exported import graph_file

    assert graph_file("d", "pfn", "trt", "fp16") == os.path.join("d", "pfn.fp16.engine")
    assert graph_file("d", "rpn_head", "onnx", "fp16") == os.path.join("d", "rpn_head.onnx")
    assert onnx_for("d", "rpn_head", "int8") == os.path.join("d", "rpn_head.int8.onnx")
    assert onnx_for("d", "pfn", "fp16") == os.path.join("d", "pfn.onnx")
    assert parse_exported("a=x/y") == ("a", "x/y", "trt", "fp16")
    assert parse_exported("b=x:onnx") == ("b", "x", "onnx", "fp16")
    assert parse_exported("c=x:trt:int8") == ("c", "x", "trt", "int8")
    with pytest.raises(ValueError):
        parse_exported("nodir")
    for name in ("pfn.fp16.engine", "rpn_head.int8.engine", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    found = engine_paths(str(tmp_path), None)
    assert found == [str(tmp_path / "pfn.fp16.engine"), str(tmp_path / "rpn_head.int8.engine")]
    named = engine_paths("d", ["pfn.fp16", "rpn_head.fp16.engine"])
    assert named == [os.path.join("d", "pfn.fp16.engine"), os.path.join("d", "rpn_head.fp16.engine")]
    assert parse_shapes("pillars:1x10x10000x20,b:1x3") == {"pillars": (1, 10, 10000, 20), "b": (1, 3)}
    assert parse_shapes(None) == {}


@pytest.mark.slow
@requires_det3d
def test_dynamic_pillar_pfn_matches_static(exported, sim_frames, tmp_path):
    """--dynamic-pillars: the PFN gets only the used pillars; detections as with the padded graph."""
    from export_pillar_onnx import export
    from dog_perception.detection.exported import ExportedPillarDetector
    from dog_perception.detection.model_meta import ModelMeta, meta_from_detector

    det, out = exported
    dyn = str(tmp_path / "dyn")
    export(det, dyn, 20000, dynamic_pillars=True)
    meta = meta_from_detector(det, 20000, dynamic_pillars=True)
    meta.source = dict(config=PP_CFG, checkpoint=PP_CKPT, xy_range=40.8)
    meta.save(os.path.join(dyn, "model_meta.json"))
    assert ModelMeta.load(os.path.join(dyn, "model_meta.json")).onnx["pfn"]["dynamic"]
    assert not ModelMeta.load(os.path.join(out, "model_meta.json")).onnx["pfn"]["dynamic"]
    a, b = ExportedPillarDetector(out, "onnx", keep_stages=True), ExportedPillarDetector(dyn, "onnx", keep_stages=True)
    assert not a.pfn_dynamic and b.pfn_dynamic
    for fr in sim_frames["frames"][-2:]:
        ra, rb = a(fr.points, fr.stamp), b(fr.points, fr.stamp)
        assert len(ra) == len(rb) > 0
        ia, ib = np.argsort(-ra.scores), np.argsort(-rb.scores)
        assert np.abs(ra.scores[ia] - rb.scores[ib]).max() < 1e-5 and np.abs(ra.boxes[ia] - rb.boxes[ib]).max() < 1e-4
        used, _ = b.last_pillars
        assert b.last_stages["pfn_in"].shape[2] == used < a.last_stages["pfn_in"].shape[2] == 20000
        assert np.abs(a.last_stages["bev"] - b.last_stages["bev"]).max() < 1e-5


def test_trt_runner_on_a_tiny_engine(tmp_path, monkeypatch):
    """Only where TensorRT and a GPU are installed (the server, the Orin)."""
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from build_engines import benchmark, build
    from dog_perception.detection.exported import TrtRunner
    try:
        from dog_perception.detection.exported import _CudaRuntime
        _CudaRuntime()
    except OSError:
        pytest.skip("no CUDA runtime")
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU()).eval()
    x = torch.randn(1, 3, 8, 8)
    ref = net(x).detach().numpy()
    onnx_path = str(tmp_path / "tiny.onnx")
    kw = dict(dynamo=False) if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(net, x, onnx_path, input_names=["x"], output_names=["y"], opset_version=17, **kw)
    # the same graph with its weights in a side file (torch >= 2.9's default exporter does this),
    # built from another working directory: the parser must look next to the graph
    ext_path = str(tmp_path / "tiny_ext.onnx")
    onnx.save_model(onnx.load(onnx_path), ext_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="tiny_ext.onnx.data", size_threshold=0)
    assert os.path.exists(ext_path + ".data")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for src, eng in ((onnx_path, "tiny.fp32.engine"), (ext_path, "tiny_ext.fp32.engine")):
        build(src, str(tmp_path / eng), "fp32", workspace_mb=256)
        y = TrtRunner(str(tmp_path / eng))({"x": x.numpy()})["y"]
        assert trt.__version__ and np.abs(y - ref).max() < 1e-4
    b = benchmark(str(tmp_path / "tiny.fp32.engine"), 20, warmup_ms=50)
    assert b["runs"] == 20 and 0 < b["gpu_min_ms"] <= b["gpu_median_ms"] <= b["gpu_p99_ms"] <= b["gpu_max_ms"]
    assert b["host_median_ms"] > 0
    # a dynamic dimension (the pillars of a --dynamic-pillars PFN) needs an optimization profile
    dnet = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU()).eval()
    dyn_path = str(tmp_path / "tiny_dyn.onnx")
    torch.onnx.export(dnet, torch.randn(1, 3, 50, 4), dyn_path, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamic_axes={"x": {2: "p"}, "y": {2: "p"}}, **kw)
    with pytest.raises(ValueError):
        build(dyn_path, str(tmp_path / "tiny_dyn.fp32.engine"), "fp32", workspace_mb=256)
    prof = {"x": ((1, 3, 1, 4), (1, 3, 30, 4), (1, 3, 50, 4))}
    info = build(dyn_path, str(tmp_path / "tiny_dyn.fp32.engine"), "fp32", workspace_mb=256, profile=prof)
    assert info["profile"]["x"][2] == [1, 3, 50, 4]
    r = TrtRunner(str(tmp_path / "tiny_dyn.fp32.engine"))
    assert r.dynamic == ["x"] and r.profiles["x"][1] == (1, 3, 30, 4)
    for p in (1, 17, 50):
        xi = torch.randn(1, 3, p, 4)
        y = r({"x": xi.numpy()})["y"]
        assert y.shape == (1, 4, p, 4) and np.abs(y - dnet(xi).detach().numpy()).max() < 1e-4
    with pytest.raises(ValueError):
        r({"x": np.zeros((1, 3, 51, 4), np.float32)})
    assert r.benchmark(10, warmup_ms=10, shapes={"x": (1, 3, 20, 4)})["shapes"]["x"] == [1, 3, 20, 4]
    assert r.benchmark(10, warmup_ms=10)["shapes"]["x"] == [1, 3, 30, 4]
    report = tmp_path / "bench.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "bench_engines.py"), "--model-dir",
                    str(tmp_path), "--engines", "tiny.fp32", "--runs", "10", "--warmup-ms", "20", "--out", str(report)],
                   check=True, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True)
    r = json.load(open(report))
    assert [e["engine"] for e in r["engines"]] == ["tiny.fp32.engine"] and r["trt"] == trt.__version__


def test_export_cpp_config(tmp_path):
    """The C++ parameters come from the dataclasses: every model's q_omega resolved, per-class values for all
    three classes, a sequence's meta.json reflected, the YAML the C++ loader reads."""
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from export_cpp_config import config_dict, to_yaml
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig
    from dog_perception.tracking import TrackerConfig

    d = config_dict(PreprocessConfig(det_frame=DetFrameConfig(base_height=0.72, auto_level=True)),
                    tracker=TrackerConfig(classes=("vehicle", "pedestrian")))
    assert d["version"] == 1 and d["preprocess"]["det_frame"]["base_height"] == 0.72
    assert d["preprocess"]["det_frame"]["auto_level"] is True and len(d["preprocess"]["T_body_lidar"]) == 16
    assert d["tracker"]["classes"] == ["vehicle", "pedestrian"]
    assert d["tracker"]["class_params"]["vehicle"]["models"] == [["cv", 1.5, 0.001], ["ct", 1.5, 0.25]]
    assert d["tracker"]["high_score"] == {"vehicle": 0.4, "pedestrian": 0.3, "cyclist": 0.4}
    assert d["detection_filter"]["min_points"]["cyclist"] == 3.0 and d["predictor"]["horizon"] == 3.0
    text = to_yaml(d)
    assert "  T_body_lidar: [1.0, 0.0, 0.0, 0.2," in text and "matching: \"greedy\"" in text
    out = tmp_path / "kygi.yaml"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "export_cpp_config.py"), "--robot", "kygi",
                    "--out", str(out)], check=True, capture_output=True, env=dict(os.environ, PYTHONPATH=ROOT))
    t = out.read_text()
    assert "base_height: 0.72" in t and "localization: \"inspvax\"" in t and "ins_lever_arm: [-0.06, -0.54, 0.0]" in t


def test_objects_bag_against_a_replay(tmp_path):
    """tools/deploy/objects_bag.py reads a recorded ~/objects bag (the message definitions of
    deploy/ros2/dog_perception_msgs) and finds dropped frames against a replay of the same input."""
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from objects_bag import read_objects, typestore

    ts = typestore()
    Obj, Arr = ts.types["dog_perception_msgs/msg/Object"], ts.types["dog_perception_msgs/msg/ObjectArray"]
    Header, Time = ts.types["std_msgs/msg/Header"], ts.types["builtin_interfaces/msg/Time"]
    Point, Vec, Pose = (ts.types["geometry_msgs/msg/" + n] for n in ("Point", "Vector3", "Pose"))
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    frames = [(100.1, [3, 5]), (100.2, [3]), (100.3, [3, 7])]

    def message(stamp, ids):
        objs = [Obj(id=i, label=0, label_name="vehicle", motion=Obj.MOTION_MOVING, score=0.8, coasting=False, age=1.0,
                    position=Point(x=1.0, y=2.0, z=0.8), yaw=0.1, size=Vec(x=4.5, y=1.9, z=1.6),
                    velocity=Vec(x=1.0, y=0.0, z=0.0), covariance=np.eye(4).ravel(),
                    prediction_times=np.array([0.5, 1.0], np.float32), predictions=[]) for i in ids]
        sec = int(stamp)
        return Arr(header=Header(stamp=Time(sec=sec, nanosec=int(round((stamp - sec) * 1e9))), frame_id="dog_det"),
                   world_frame="camera_init", world_pose=Pose(position=Point(x=0.0, y=0.0, z=0.0),
                                                              orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0)),
                   objects=objs)

    bag = tmp_path / "objects"
    with Writer(bag, version=8) as w:
        c = w.add_connection("/dog_perception/objects", Arr.__msgtype__, typestore=ts)
        for stamp, ids in frames[:2]:                          # the third frame was dropped
            w.write(c, int((stamp + 0.03) * 1e9), ts.serialize_cdr(message(stamp, ids), Arr.__msgtype__))
    rows = read_objects(bag)
    assert [r["ids"] for r in rows] == [[3, 5], [3]] and rows[0]["frame"] == "dog_det"
    replay = tmp_path / "replay.npz"
    np.savez(replay, stamp=np.array([f[0] for f in frames]), trk_offsets=np.array([0, 2, 3, 5]),
             trk_id=np.array([3, 5, 3, 3, 7]))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "objects_bag.py"), "--bag", str(bag),
                        "--replay", str(replay)], capture_output=True, text=True)
    rep = json.loads(r.stdout)
    assert r.returncode == 1 and rep["missing_frames"] == 1 and rep["same_ids_frames"] == 2 and not rep["passed"]

    # ros2 bag play --loop: three passes (the stamps jump back 4 s), the recording stopped during the third one
    loop = [(100.0 + 2 * k, ids) for k, (_, ids) in enumerate(frames)]
    replay_loop = tmp_path / "replay_loop.npz"
    np.savez(replay_loop, stamp=np.array([f[0] for f in loop]), trk_offsets=np.array([0, 2, 3, 5]),
             trk_id=np.array([3, 5, 3, 3, 7]))

    def looped(name, drop=None):
        path = tmp_path / name
        with Writer(path, version=8) as w:
            c = w.add_connection("/dog_perception/objects", Arr.__msgtype__, typestore=ts)
            for k, (stamp, ids) in enumerate(loop * 2 + loop[:1]):
                if k != drop:
                    w.write(c, int((10 * k + 0.03) * 1e9), ts.serialize_cdr(message(stamp, ids), Arr.__msgtype__))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "deploy", "objects_bag.py"), "--bag",
                            str(path), "--replay", str(replay_loop)], capture_output=True, text=True)
        return r.returncode, json.loads(r.stdout)

    code, rep = looped("loop")
    assert code == 0 and rep["passes"] == 3 and rep["passed"] and rep["matched_frames"] == 7
    assert rep["received_gap_max"] == pytest.approx(10.0)
    assert [p["partial"] for p in rep["per_pass"]] == [False, False, True]
    code, rep = looped("loop_drop", drop=5)                    # the second pass's last frame
    assert code == 1 and rep["missing_frames"] == 1 and rep["per_pass"][1]["missing_frames"] == 1


def test_cut_bag_keeps_the_node_messages(tmp_path):
    """tools/deploy/cut_bag.py: the node's topics up to N clouds, in the source's rosbag2 version (5 here, what
    ROS 2 Humble writes), the messages a replay of the source with --max-clouds N reads."""
    pytest.importorskip("rosbags")
    bag = os.path.join(ROOT, "samples", "rosbag2_2026_09_21-16_41_27_moving")
    if not os.path.isdir(bag):
        pytest.skip("sample bag not checked out")
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    import yaml
    from cut_bag import cut
    from message_log import bag_messages, node_topics
    from dog_perception.node_config import node_parameters

    params = node_parameters("kygi")
    out = str(tmp_path / "cut")
    counts = cut(bag, out, node_topics(params), params["lidar_topic"], 5)
    assert counts[params["lidar_topic"]] == 5
    with open(os.path.join(out, "metadata.yaml")) as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    assert info["version"] == 5 and info["message_count"] == sum(counts.values())
    assert info["topics_with_message_count"][0]["topic_metadata"]["offered_qos_profiles"].startswith("- history")
    ref = []
    for kind, m in bag_messages(bag, params):
        if kind == "cloud" and sum(k == "cloud" for k, _ in ref) == 5:
            break
        ref.append((kind, m))
    got = list(bag_messages(out, params))
    assert [k for k, _ in got] == [k for k, _ in ref]
    assert all(m.header.stamp == r.header.stamp for (_, m), (_, r) in zip(got, ref))
    clouds = [(m, r) for (k, m), (_, r) in zip(got, ref) if k == "cloud"]
    assert all(np.array_equal(m.data, r.data) for m, r in clouds)


def test_soak_report_windows_and_warnings(tmp_path):
    """tools/deploy/soak_report.py on made-up orin_soak.sh logs: windows with and without the load."""
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from soak_report import read_load, read_node_log, read_tegrastats, windows

    lines = ["[dogperc_node-1] [INFO] [0000001000.000000000] [dog_perception]: dog perception up: data/kygi.yaml"]
    for k in range(20):
        t = 1000.0 + 30 * k
        lines.append("[dogperc_node-1] [INFO] [%013.9f] [dog_perception]: %d dets, %d tracks | preprocess_ms 6.0 "
                     "det_gpu_ms 10.0 total_ms %.1f" % (t, 40 + k, 5, 18.0 if t < 1300 else 25.0))
    lines += ["[dogperc_node-1] [WARN] [0000001100.500000000] [dog_perception]: inspvax stamps jumped back 59.9 s "
              "(bag played again or clock reset): starting over"] * 3
    (tmp_path / "node.log").write_text("\n".join(lines) + "\n")
    (tmp_path / "load.log").write_text("load on 1300\nwarn: something\nload off 1700\n")
    (tmp_path / "tegrastats.log").write_text(
        "RAM 5000/30000MB CPU [5%@2201] GR3D_FREQ 40% gpu@45.5C tj@50C\n"
        "RAM 5100/30000MB CPU [9%@2201] GR3D_FREQ 60% gpu@47C tj@52.5C\n")
    frames, warnings = read_node_log(tmp_path / "node.log")
    assert len(frames) == 20 and frames[0]["dets"] == 40 and frames[0]["det_gpu_ms"] == 10.0
    assert warnings == {"WARN: inspvax stamps jumped back # s (bag played again or clock reset): starting over": 3}
    rows = windows(frames, 300.0, read_load(tmp_path / "load.log"))
    assert [(r["load"], r["frames"], r["total_median"]) for r in rows] == [(False, 10, 18.0), (True, 10, 25.0)]
    teg = read_tegrastats(tmp_path / "tegrastats.log")
    assert teg["gr3d_mean"] == 50 and teg["ram_mb_max"] == 5100 and teg["temp_max_c"] == dict(gpu=47.0, tj=52.5)
