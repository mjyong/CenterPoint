"""CenterPoint-Pillar / CenterPoint-Voxel (this repo's det3d models) behind one API.

    det = CenterPointDetector("configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py",
                              "work_dirs/.../latest.pth")
    dets = det(frame.points)          # Detections in the det frame

Both architectures consume the same preprocessed cloud, so they can be
swapped or run side by side for comparison.
"""
import copy
import time

import numpy as np
import torch

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint

from .boxes import CLASSES, NUSC_TO_DOG, Detections, class_index, classwise_circle_nms, det3d_to_standard
from .frame_adapter import ModelFrame

# center-distance NMS radii (squared meters, det3d convention), per class
CIRCLE_NMS_SQ_RADIUS = {
    "car": 4.0, "truck": 12.0, "construction_vehicle": 12.0, "bus": 10.0, "trailer": 10.0,
    "barrier": 1.0, "motorcycle": 0.85, "bicycle": 0.85, "pedestrian": 0.175, "traffic_cone": 0.175,
    "vehicle": 4.0, "cyclist": 0.85,
}

# merged-class NMS after nuScenes -> dog mapping (car/truck/bus heads are
# separate tasks and can fire on the same object)
MERGED_NMS_RADIUS = {"vehicle": 2.0, "cyclist": 0.9, "pedestrian": 0.0}


def _required_divisor(model_cfg):
    ds = model_cfg["backbone"].get("ds_factor", 1)
    return int(ds * np.prod(model_cfg["neck"].get("ds_layer_strides", [1])))


def configure_for_dog(cfg, xy_range=40.8, score_threshold=0.1):
    """Adapt a det3d config in place: square BEV range and CPU-friendly circle NMS.

    z range is left untouched on purpose: VoxelNet's sparse backbone needs 41
    z-cells to produce the 256-channel BEV map its pretrained neck expects.
    """
    vg = cfg.voxel_generator
    vs = vg.voxel_size
    div = _required_divisor(cfg.model)
    cells = 2 * xy_range / vs[0]
    if abs(cells - round(cells)) > 1e-6 or round(cells) % div:
        raise ValueError("2*xy_range/voxel_size = %.3f must be an integer multiple of %d" % (cells, div))
    r = list(vg.range)
    r[0], r[1], r[3], r[4] = -xy_range, -xy_range, xy_range, xy_range
    vg.range = r
    if "pc_range" in cfg.model.reader:
        cfg.model.reader.pc_range = tuple(r)
    tc = cfg.test_cfg
    tc.pc_range = [-xy_range, -xy_range]
    lim = list(tc.post_center_limit_range)
    lim[0], lim[1], lim[3], lim[4] = -xy_range, -xy_range, xy_range, xy_range
    tc.post_center_limit_range = lim
    tc.score_threshold = score_threshold
    if not tc.get("circular_nms", False):
        tasks = cfg.model.bbox_head.tasks
        tc.circular_nms = True
        tc.min_radius = [max(CIRCLE_NMS_SQ_RADIUS.get(n, 1.0) for n in t["class_names"]) for t in tasks]
    tc.double_flip = False
    return cfg


class CenterPointDetector:
    def __init__(self, config, checkpoint=None, device=None, frame=None, z_range=(-1.0, 3.0),
                 xy_range=40.8, score_threshold=0.1, class_map=None, merged_nms=None, tta=False):
        cfg = Config.fromfile(config) if isinstance(config, str) else copy.deepcopy(config)
        configure_for_dog(cfg, xy_range, score_threshold)
        self.cfg = cfg
        self.frame = frame or ModelFrame()
        self.z_range = z_range
        self.tta = tta
        self.merged_nms = MERGED_NMS_RADIUS if merged_nms is None else merged_nms
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.arch = cfg.model.type

        names = [n for t in cfg.model.bbox_head.tasks for n in t["class_names"]]
        if class_map is None:
            class_map = NUSC_TO_DOG if any(n in NUSC_TO_DOG for n in names) else {n: n for n in names}
        self.model_class_names = names
        self.label_map = np.array([class_index(class_map.get(n, "")) for n in names], dtype=np.int64)

        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        if checkpoint:
            load_checkpoint(self.net, checkpoint, map_location="cpu")
        self.net = self.net.to(self.device).eval()

        vg = cfg.voxel_generator
        self.voxel_generator = VoxelGenerator(
            voxel_size=vg.voxel_size, point_cloud_range=vg.range,
            max_num_points=vg.max_points_in_voxel, max_voxels=vg.max_voxel_num[1])
        self.num_point_features = cfg.model.reader.get("num_input_features", 5)
        self.last_timing = {}

    # ------------------------------------------------------------------ input
    def _crop(self, points_det):
        z = points_det[:, 2]
        return points_det[(z >= self.z_range[0]) & (z <= self.z_range[1])]

    def _flips(self, pts):
        if not self.tta:
            return [pts]
        out = [pts]
        for fx, fy in ((1, -1), (-1, 1), (-1, -1)):   # det3d DoubleFlip order: y, x, xy
            p = pts.copy()
            p[:, 0] *= fx
            p[:, 1] *= fy
            out.append(p)
        return out

    def build_example(self, points_model):
        clouds = self._flips(points_model[:, :self.num_point_features].astype(np.float32))
        vox, coors, npts, nvox = [], [], [], []
        for b, pts in enumerate(clouds):
            v, c, n = self.voxel_generator.generate(pts)
            vox.append(v)
            coors.append(np.pad(c, ((0, 0), (1, 0)), constant_values=b))
            npts.append(n)
            nvox.append(len(v))
        dev = self.device
        grid = self.voxel_generator.grid_size
        return dict(
            voxels=torch.from_numpy(np.concatenate(vox)).to(dev),
            coordinates=torch.from_numpy(np.concatenate(coors)).int().to(dev),
            num_points=torch.from_numpy(np.concatenate(npts)).to(dev),
            num_voxels=torch.tensor(nvox, dtype=torch.int64, device=dev),
            shape=[grid] * len(clouds),
            points=[torch.from_numpy(p).to(dev) for p in clouds],
        )

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def __call__(self, points_det, stamp=None):
        t0 = time.perf_counter()
        pts = self.frame.points_to_model(self._crop(points_det))
        if len(pts) == 0:
            return Detections(stamp=stamp)
        example = self.build_example(pts)
        self.net.test_cfg.double_flip = self.tta
        t1 = time.perf_counter()
        out = self.net(example, return_loss=False)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        dets = self.postprocess(out["box3d_lidar"].float().cpu().numpy(), out["scores"].float().cpu().numpy(),
                                out["label_preds"].cpu().numpy(), stamp)
        t3 = time.perf_counter()
        self.last_timing = {"voxelize_ms": 1e3 * (t1 - t0), "network_ms": 1e3 * (t2 - t1),
                            "post_ms": 1e3 * (t3 - t2)}
        return dets

    def postprocess(self, box9, scores, model_labels, stamp=None):
        labels = self.label_map[model_labels.astype(np.int64)]
        keep = labels >= 0
        std, vel = det3d_to_standard(box9[keep])
        std, vel = self.frame.boxes_from_model(std, vel)
        dets = Detections(std, vel, scores[keep].astype(np.float64), labels[keep], "det", stamp)
        return classwise_circle_nms(dets, self.merged_nms)

    @property
    def classes(self):
        return CLASSES
