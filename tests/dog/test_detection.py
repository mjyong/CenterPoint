import numpy as np
import pytest

from dog_perception.detection import (CLASSES, Detections, ModelFrame, circle_nms, classwise_circle_nms,
                                      det3d_to_standard, standard_to_det3d)
from dog_perception.detection.evaluate import evaluate_detections
from dog_perception.geometry import make_T, rot_z

from .conftest import requires_det3d

PP_CFG = "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py"


def test_box_convention_roundtrip_and_nuscenes_yaw():
    rng = np.random.default_rng(0)
    std = np.column_stack([rng.normal(0, 5, (10, 3)), rng.uniform(0.5, 5, (10, 3)), rng.uniform(-3, 3, 10)])
    vel = rng.normal(0, 2, (10, 2))
    b9 = standard_to_det3d(std, vel)
    back, v = det3d_to_standard(b9)
    assert np.allclose(back[:, :6], std[:, :6]) and np.allclose(v, vel)
    assert np.allclose(np.cos(back[:, 6]), np.cos(std[:, 6])) and np.allclose(np.sin(back[:, 6]), np.sin(std[:, 6]))
    # nusc_common stores  -yaw - pi/2  and  (w, l, h)
    assert np.allclose(b9[:, 3], std[:, 4]) and np.allclose(b9[:, 4], std[:, 3])


def test_model_frame_roundtrip_and_semantics():
    mf = ModelFrame()
    pts = np.array([[10.0, 0.0, 0.0, 5.0, 0.1]], dtype=np.float32)   # 10 m ahead, on the ground
    m = mf.points_to_model(pts)
    assert np.allclose(m[0, :3], [0.0, 10.0, -1.84019], atol=1e-4)   # nuScenes: forward = +y, lidar 1.84 m up
    assert np.allclose(mf.points_from_model(m), pts, atol=1e-5)
    boxes = np.array([[10.0, 2.0, 0.9, 4.5, 1.9, 1.6, 0.3]])
    vel = np.array([[1.0, 0.5]])
    bm, vm = mf.boxes_to_model(boxes, vel)
    bd, vd = mf.boxes_from_model(bm, vm)
    assert np.allclose(bd, boxes) and np.allclose(vd, vel)


def test_detections_transform_and_nms():
    d = Detections(np.array([[1.0, 0, 0, 1, 1, 1, 0.0], [1.2, 0, 0, 1, 1, 1, 0.0], [5, 5, 0, 1, 1, 1, 0.0]]),
                   np.array([[1.0, 0], [1.0, 0], [0, 0]]), np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]))
    T = make_T(rot_z(np.pi / 2), [10, 0, 0])
    w = d.transform(T, "world")
    assert np.allclose(w.boxes[0, :3], [10, 1, 0]) and np.isclose(w.boxes[0, 6], np.pi / 2)
    assert np.allclose(w.velocities[0], [0, 1])
    assert list(circle_nms(d.boxes[:, :2], d.scores, 0.5)) == [0, 2]
    kept = classwise_circle_nms(d, {"vehicle": 0.5})
    assert len(kept) == 2


def test_center_distance_ap():
    gt = [dict(boxes=np.array([[0, 0, 0, 1, 1, 1, 0], [10, 0, 0, 1, 1, 1, 0]]), labels=np.array([1, 1]))]
    pred = [dict(boxes=np.array([[0.3, 0, 0, 1, 1, 1, 0], [30, 0, 0, 1, 1, 1, 0]]), labels=np.array([1, 1]),
                 scores=np.array([0.9, 0.8]))]
    r = evaluate_detections(pred, gt, CLASSES)
    assert np.isclose(r["pedestrian"]["AP@0.5"], 0.5) and np.isclose(r["pedestrian"]["ATE"], 0.3)
    assert np.isnan(r["vehicle"]["mAP"])


@requires_det3d
def test_decoder_matches_det3d_predict():
    """NumPy decoder (BPU/ONNX path) == CenterHead.predict on peaked random maps."""
    import torch
    from det3d.torchie import Config
    from det3d.models import build_detector
    from dog_perception.detection.centerpoint import configure_for_dog
    from dog_perception.detection.decode import decode_centerpoint

    cfg = configure_for_dog(Config.fromfile(PP_CFG))
    net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg).eval()
    rng = np.random.default_rng(0)
    H = W = 102
    preds, maps = [], []
    for t, nc in enumerate(net.bbox_head.num_classes):
        m = {k: rng.normal(0, 0.3, (c, H, W)).astype(np.float32)
             for k, c in (("reg", 2), ("height", 1), ("dim", 3), ("rot", 2), ("vel", 2))}
        hm = np.full((nc, H, W), -8.0, np.float32)
        idx = rng.choice(H * W, 30, replace=False)
        hm.reshape(nc, -1)[rng.integers(0, nc, 30), idx] = rng.uniform(-1, 3, 30)
        m["hm"] = hm
        maps.append(m)
        preds.append({k: torch.from_numpy(v[None].copy()) for k, v in m.items()})
    out = net.bbox_head.predict({}, preds, net.test_cfg)[0]
    box, score, label = decode_centerpoint(maps, net.bbox_head.num_classes, dict(net.test_cfg))
    o = np.argsort(-out["scores"].numpy())
    n = np.argsort(-score)
    assert len(o) == len(n) > 0
    assert np.allclose(out["scores"].numpy()[o], score[n], atol=1e-5)
    assert np.array_equal(out["label_preds"].numpy()[o], label[n])
    assert np.allclose(out["box3d_lidar"].numpy()[o], box[n], atol=1e-4)


@requires_det3d
def test_pillar_export_parity_and_detector(sim_frames):
    import torch
    from dog_perception.detection.centerpoint import CenterPointDetector
    from dog_perception.detection.decode import HEAD_ORDER
    from dog_perception.detection.pillar_export import (PFNExport, RPNHeadExport, decorate_pillars, pad_pillars,
                                                        scatter_pillars)

    torch.manual_seed(0)
    det = CenterPointDetector(PP_CFG, device="cpu")
    frame = sim_frames["frames"][-1]
    dets = det(frame.points, frame.stamp)
    assert set(det.last_timing) == {"voxelize_ms", "network_ms", "post_ms"}
    assert dets.boxes.shape[1] == 7 and (dets.labels >= 0).all()

    cfg, net = det.cfg, det.net
    pts = det.frame.points_to_model(det._crop(frame.points))
    v, c, n = det.voxel_generator.generate(pts)
    feats = decorate_pillars(v, n, c, cfg.voxel_generator.voxel_size, cfg.voxel_generator.range)
    x, cpad, nv = pad_pillars(feats, c, 30000)
    with torch.no_grad():
        pf = PFNExport(net.reader, cfg.voxel_generator.max_points_in_voxel).eval()(torch.from_numpy(x))[0, :, :, 0].numpy()
        ref = net.reader(torch.from_numpy(v), torch.from_numpy(n), torch.from_numpy(np.pad(c, ((0, 0), (1, 0))))).numpy()
        assert np.abs(pf[:, :nv].T - ref).max() < 1e-4
        canvas = scatter_pillars(pf, cpad, nv, det.voxel_generator.grid_size)
        ref_canvas = net.backbone(torch.from_numpy(ref), torch.from_numpy(np.pad(c, ((0, 0), (1, 0)))), 1,
                                  det.voxel_generator.grid_size).numpy()
        assert np.abs(canvas - ref_canvas).max() < 1e-4
        outs = RPNHeadExport(net).eval()(torch.from_numpy(canvas))
        assert len(outs) == len(HEAD_ORDER) * len(net.bbox_head.tasks)


@requires_det3d
def test_configure_for_dog_validates_range():
    from det3d.torchie import Config
    from dog_perception.detection.centerpoint import configure_for_dog

    cfg = configure_for_dog(Config.fromfile(PP_CFG), 40.8)
    assert cfg.voxel_generator.range[3] == 40.8 and cfg.model.reader.pc_range[0] == -40.8
    assert cfg.test_cfg.circular_nms and len(cfg.test_cfg.min_radius) == 6
    with pytest.raises(ValueError):
        configure_for_dog(Config.fromfile(PP_CFG), 40.0 + 0.1)


@requires_det3d
def test_nusc_to_dog_head_surgery_loads_strictly():
    import torch
    from det3d.models import build_detector
    from det3d.torchie import Config
    from tools.dog.convert_nusc_ckpt import convert_state_dict

    src = build_detector(Config.fromfile(PP_CFG).model, train_cfg=None, test_cfg=None)
    dst_cfg = Config.fromfile("configs/dog/dog_centerpoint_pp_xt32.py")
    dst = build_detector(dst_cfg.model, train_cfg=None, test_cfg=dst_cfg.test_cfg)
    sd = convert_state_dict(src.state_dict())
    dst.load_state_dict(sd, strict=True)
    # pedestrian head's heat-map comes from nuScenes task 5 channel 0, cyclist from task 4 channel 1
    assert torch.equal(dst.bbox_head.tasks[1].hm[-1].weight, src.bbox_head.tasks[5].hm[-1].weight[[0]])
    assert torch.equal(dst.bbox_head.tasks[2].hm[-1].bias, src.bbox_head.tasks[4].hm[-1].bias[[1]])
    assert torch.equal(dst.bbox_head.tasks[2].vel[0].weight, src.bbox_head.tasks[4].vel[0].weight)


@requires_det3d
def test_tta_batches_four_flips(sim_frames):
    from dog_perception.detection.centerpoint import CenterPointDetector
    det = CenterPointDetector(PP_CFG, device="cpu", tta=True)
    ex = det.build_example(det.frame.points_to_model(sim_frames["frames"][-1].points))
    assert len(ex["num_voxels"]) == 4 and int(ex["coordinates"][:, 0].max()) == 3
    d = det(sim_frames["frames"][-1].points)
    assert d.boxes.shape[1] == 7


@requires_det3d
def test_voxel_detector_builds():
    import torch
    from dog_perception.detection import build_detector
    try:
        import spconv  # noqa: F401
    except ImportError:
        pytest.skip("spconv not installed")
    det = build_detector("voxel", device="cuda" if torch.cuda.is_available() else "cpu")
    assert det.arch == "VoxelNet" and list(det.voxel_generator.grid_size) == [1088, 1088, 40]
    if not torch.cuda.is_available():
        pytest.skip("spconv CPU kernels do not support bias; forward needs CUDA")
    pts = np.random.default_rng(0).uniform([-30, -30, -0.5, 0, 0], [30, 30, 2, 100, 0.4], (20000, 5)).astype(np.float32)
    assert det(pts).boxes.shape[1] == 7


@requires_det3d
def test_standard_boxes_agree_with_det3d_points_in_rbbox():
    """Our [x,y,z,l,w,h,yaw] <-> det3d [x,y,z,w,l,h,vx,vy,r] conversion selects the same points
    as det3d itself (used by the GT database / GT sampling)."""
    from det3d.core.bbox import box_np_ops
    from dog_perception.preprocess.self_filter import points_in_boxes_oriented

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-10, 10, (100000, 2)), rng.uniform(-1, 2, 100000)]).astype(np.float32)
    std = np.array([[2.0, 1.0, 0.5, 4.5, 1.9, 1.6, 0.6], [-3.0, 4.0, 0.3, 0.8, 0.6, 1.7, -2.2],
                    [5.0, -5.0, 0.2, 1.8, 0.6, 1.7, 1.2]])
    inside = box_np_ops.points_in_rbbox(pts, standard_to_det3d(std))
    for i in range(len(std)):
        mine = points_in_boxes_oriented(pts, std[i:i + 1])
        assert mine.sum() > 50 and np.array_equal(inside[:, i], mine)
