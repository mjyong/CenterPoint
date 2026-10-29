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
    """Misses / extras at the reporting threshold: near (score noise across it), score, class, none."""
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
    assert r["missed_by"] == {"near": 1, "score": 1, "class": 1, "none": 1}
    assert r["extra_by"] == {"near": 1, "score": 0, "class": 1, "none": 1}
    assert r["reported_match"] == pytest.approx(1 - 7 / 5) and r["reported_match_tol"] == pytest.approx(1 - 5 / 5)


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
