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

from .boxes import (CLASSES, NUSC_TO_DOG, NUSC_TO_SUBCLASS, SUBCLASS_TO_CLASS, SUBCLASSES, Detections, class_index,
                    classwise_circle_nms, det3d_to_standard, subclass_index)
from .filters import DetectionFilterConfig, filter_detections
from .frame_adapter import ModelFrame

# center-distance NMS radii (squared meters, det3d convention), per class
CIRCLE_NMS_SQ_RADIUS = {
    "car": 4.0, "truck": 12.0, "construction_vehicle": 12.0, "bus": 10.0, "trailer": 10.0,
    "barrier": 1.0, "motorcycle": 0.85, "bicycle": 0.85, "pedestrian": 0.175, "traffic_cone": 0.175,
    "vehicle": 4.0, "cyclist": 0.85,
}

# merged-class NMS after nuScenes -> dog mapping (car/truck/bus heads are
# separate tasks and can fire on the same object). Pedestrians: merge within
# 0.6 m, and up to 1.2 m when the weaker box scores <= 0.6x the stronger one
# (a weak second peak on the same person; two people next to each other both
# score high). Chosen on nuScenes mini_val, see docs/NUSC_CLIP_EVAL.md.
MERGED_NMS_RADIUS = {"vehicle": 2.0, "cyclist": 0.9, "pedestrian": (0.6, 1.2, 0.6)}


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


def _load_state_dict(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)   # trusted local file
    except TypeError:  # torch < 1.13 (older JetPack wheels) has no weights_only
        ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def default_class_map(model_class_names):
    """Model class -> dog class. Dog models (configs/dog/: car / truck / bus / pedestrian / cyclist,
    or the older vehicle / pedestrian / cyclist) go through SUBCLASS_TO_CLASS, nuScenes models
    through NUSC_TO_DOG. (Testing 'any name in NUSC_TO_DOG' is not enough: 'pedestrian' is in both,
    and a dog model's cyclist would map to nothing and be dropped.)"""
    if set(model_class_names) <= set(CLASSES) | set(SUBCLASSES):
        return {n: SUBCLASS_TO_CLASS.get(n, n) for n in model_class_names}
    return NUSC_TO_DOG


def model_subclass(name):
    """Subclass of a model class name (None for the older dog models' 'vehicle')."""
    return name if name in SUBCLASSES else NUSC_TO_SUBCLASS.get(name)


def checkpoint_arch(path):
    """Preset name from the parameter names of a det3d checkpoint: 'pillar' / 'voxel' for the
    nuScenes models (6 head tasks), 'dog_pillar' / 'dog_voxel' for the fine-tuned dog models
    (5 tasks: car, truck, bus, pedestrian, cyclist), 'dog_pillar_3cls' / 'dog_voxel_3cls' for the
    older 3-task ones (vehicle, pedestrian, cyclist), None if unknown."""
    keys = list(_load_state_dict(path).keys())
    if any(k.startswith("reader.pfn_layers") for k in keys):
        arch = "pillar"
    elif any(k.startswith("backbone.conv_input") for k in keys):
        arch = "voxel"
    else:
        return None
    tasks = {k.split(".")[2] for k in keys if k.startswith("bbox_head.tasks.")}
    return {5: "dog_" + arch, 3: "dog_%s_3cls" % arch}.get(len(tasks), arch)


def check_checkpoint(net, path, arch=""):
    """Refuse checkpoints that do not fit the network.

    det3d's ``load_checkpoint`` only *warns* on missing / mismatched weights,
    so e.g. pillar weights with a VoxelNet config silently leave the sparse
    backbone random and the detector outputs a flood of garbage boxes.
    (Element counts are compared so spconv 1.x -> 2.x transposes still pass.)
    """
    sd = _load_state_dict(path)
    own = {k: v for k, v in net.state_dict().items() if "num_batches_tracked" not in k}
    missing = [k for k in own if k not in sd]
    bad = [k for k in own if k in sd and sd[k].numel() != own[k].numel()]
    if missing or bad:
        found = checkpoint_arch(path)
        raise ValueError(
            "checkpoint %s does not match the %s config: %d/%d weights missing, %d with wrong shape (e.g. %s). "
            "The checkpoint looks like a %s model; pass the matching config (PRESET_CONFIGS in "
            "dog_perception/detection/__init__.py)." % (path, arch, len(missing), len(own), len(bad),
                                                        (missing + bad)[0], found or "unknown"))


class CenterPointDetector:
    def __init__(self, config, checkpoint=None, device=None, frame=None, z_range=(-1.0, 5.0),
                 xy_range=40.8, score_threshold=0.1, class_map=None, merged_nms=None, tta=False,
                 filter_cfg="default"):
        """``score_threshold`` stays low (0.1) on purpose: the tracker uses low-score
        boxes to continue tracks. Report with ``Detections.above(DEFAULT_SCORE_THRESHOLDS)``.
        ``filter_cfg``: DetectionFilterConfig, "default", or None to disable the
        geometric filters (point count, ground support, cross-class duplicates)."""
        cfg = Config.fromfile(config) if isinstance(config, str) else copy.deepcopy(config)
        configure_for_dog(cfg, xy_range, score_threshold)
        self.cfg = cfg
        self.filter_cfg = DetectionFilterConfig() if filter_cfg == "default" else filter_cfg
        self.frame = frame or ModelFrame()
        self.z_range = z_range
        self.tta = tta
        self.merged_nms = MERGED_NMS_RADIUS if merged_nms is None else merged_nms
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.arch = cfg.model.type

        names = [n for t in cfg.model.bbox_head.tasks for n in t["class_names"]]
        if class_map is None:
            class_map = default_class_map(names)
        self.model_class_names = names
        self.label_map = np.array([class_index(class_map.get(n, "")) for n in names], dtype=np.int64)
        # subclass only where it agrees with the mapped class (a custom class_map may disagree)
        subs = [model_subclass(n) for n in names]
        self.subclass_map = np.array([subclass_index(s) if s and SUBCLASS_TO_CLASS[s] == class_map.get(n) else -1
                                      for n, s in zip(names, subs)], dtype=np.int64)

        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        if checkpoint:
            check_checkpoint(self.net, checkpoint, self.arch)
            load_checkpoint(self.net, checkpoint, map_location="cpu")
        self.net = self.net.to(self.device).eval()

        vg = cfg.voxel_generator
        self.voxel_generator = VoxelGenerator(
            voxel_size=vg.voxel_size, point_cloud_range=vg.range,
            max_num_points=vg.max_points_in_voxel, max_voxels=vg.max_voxel_num[1])
        self.num_point_features = cfg.model.reader.get("num_input_features", 5)
        r = np.asarray(vg.range, np.float64)
        self._mirror = bool(np.allclose(r[0], -r[3]) and np.allclose(r[1], -r[4]))   # grid symmetric in x, y
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
        """Voxels of the cloud and, with TTA, of its three flips. The flips are mirrored from the
        voxels of the cloud (coordinates reflected, point x / y negated) instead of voxelising
        each flipped cloud again: the grid is symmetric about 0, so this gives the same voxels
        (a point within float rounding of a voxel border can land in the neighbour: about 1
        voxel in 6,000 on the kygi data) at a quarter of the CPU time (0.8 s -> 0.14 s per
        frame for 5 sweeps of the XT32)."""
        clouds = self._flips(points_model[:, :self.num_point_features].astype(np.float32))
        if not self._mirror:
            return self._build_example_slow(clouds)
        v0, c0, n0 = self.voxel_generator.generate(clouds[0])
        gx, gy = self.voxel_generator.grid_size[:2]
        vox, coors, npts, nvox = [], [], [], []
        for b, (fx, fy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))[:len(clouds)]):
            v, c = v0, c0
            if b:
                v, c = v0.copy(), c0.copy()                    # coordinates are (z, y, x)
                v[:, :, 0] *= fx
                v[:, :, 1] *= fy
                if fx < 0:
                    c[:, 2] = gx - 1 - c[:, 2]
                if fy < 0:
                    c[:, 1] = gy - 1 - c[:, 1]
            vox.append(v)
            coors.append(np.pad(c, ((0, 0), (1, 0)), constant_values=b))
            npts.append(n0)
            nvox.append(len(v))
        return self._example(clouds, vox, coors, npts, nvox)

    def _build_example_slow(self, clouds):
        """Each cloud voxelised on its own (grids that are not symmetric about 0)."""
        vox, coors, npts, nvox = [], [], [], []
        for b, pts in enumerate(clouds):
            v, c, n = self.voxel_generator.generate(pts)
            vox.append(v)
            coors.append(np.pad(c, ((0, 0), (1, 0)), constant_values=b))
            npts.append(n)
            nvox.append(len(v))
        return self._example(clouds, vox, coors, npts, nvox)

    def _example(self, clouds, vox, coors, npts, nvox):
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
        self.last_raw_count = len(dets)
        if self.filter_cfg is not None:
            dets = filter_detections(dets, points_det, self.filter_cfg)
        t3 = time.perf_counter()
        self.last_timing = {"voxelize_ms": 1e3 * (t1 - t0), "network_ms": 1e3 * (t2 - t1),
                            "post_ms": 1e3 * (t3 - t2)}
        return dets

    def postprocess(self, box9, scores, model_labels, stamp=None):
        model_labels = model_labels.astype(np.int64)
        labels = self.label_map[model_labels]
        keep = labels >= 0
        std, vel = det3d_to_standard(box9[keep])
        std, vel = self.frame.boxes_from_model(std, vel)
        dets = Detections(std, vel, scores[keep].astype(np.float64), labels[keep], "det", stamp,
                          self.subclass_map[model_labels[keep]])
        return classwise_circle_nms(dets, self.merged_nms)

    @property
    def classes(self):
        return CLASSES
