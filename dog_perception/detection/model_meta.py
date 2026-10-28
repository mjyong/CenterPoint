"""What a deployed Pillar model needs besides its two engines, without det3d or PyTorch.

``tools/dog/export_pillar_onnx.py`` writes ``model_meta.json`` next to ``pfn.onnx`` /
``rpn_head.onnx`` from the PyTorch detector (``meta_from_detector``). The exported detector
(``exported.py``, ONNX Runtime or TensorRT) and the C++ pipeline (``deploy/cpp``) read it, so a
retrained or differently configured model needs no code change: grid, pillar limits, the head
layout per task, the decoder settings, the class mapping to the dog classes and the merged-class NMS
all come from here.

``postprocess`` is the part of ``CenterPointDetector`` after the network (shared by both).
"""
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .boxes import Detections, classwise_circle_nms, det3d_to_standard
from .decode import HEAD_ORDER
from .frame_adapter import ModelFrame

META_VERSION = 1


@dataclass
class ModelMeta:
    voxel_size: list                 # [3] m
    pc_range: list                   # [6] xyz min, xyz max, model frame
    grid_size: list                  # [3] nx, ny, nz
    max_points: int                  # points per pillar
    max_voxels: int                  # det3d's limit (test)
    max_pillars: int                 # P_max of the PFN engine
    num_point_features: int          # x, y, z, intensity, dt
    pfn_channels: int                # decorated features (10)
    bev_channels: int                # PFN output = BEV canvas channels (64)
    z_range: list                    # [2] det-frame crop
    frame: dict                      # ModelFrame fields
    tasks: list                      # per task: {"class_names": [...]}
    head_keys: list                  # present keys, HEAD_ORDER
    test_cfg: dict                   # decoder (decode.decode_centerpoint)
    label_map: list                  # model class -> dog class index (-1: dropped)
    subclass_map: list               # model class -> vehicle subclass index (-1: none)
    merged_nms: dict                 # dog class -> radius or [radius, weak radius, weak ratio]
    # pfn "dynamic": the pillar dimension is free (export_pillar_onnx.py --dynamic-pillars), max_pillars caps it
    onnx: dict = field(default_factory=lambda: {"pfn": {"input": "pillars", "output": "pillar_features",
                                                        "dynamic": False},
                                                "rpn": {"input": "bev", "outputs": []}})
    source: dict = field(default_factory=dict)
    version: int = META_VERSION

    @property
    def num_classes(self):
        return [len(t["class_names"]) for t in self.tasks]

    @property
    def model_frame(self):
        return ModelFrame(**self.frame)

    @property
    def rpn_outputs(self):
        return ["task%d_%s" % (t, k) for t in range(len(self.tasks)) for k in self.head_keys]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(_plain(asdict(self)), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        if d.get("version", 0) != META_VERSION:
            raise ValueError("%s: model_meta version %s, this code reads %d (re-export the ONNX)"
                             % (path, d.get("version"), META_VERSION))
        return cls(**d)


def _plain(x):
    """JSON-safe copy (numpy scalars / arrays and tuples to Python lists and numbers)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def meta_from_detector(det, max_pillars, dynamic_pillars=False):
    """ModelMeta of a ``CenterPointDetector`` (Pillar)."""
    cfg = det.cfg
    vg, tc = cfg.voxel_generator, cfg.test_cfg
    net = det.net
    tasks = [{"class_names": list(t["class_names"])} for t in cfg.model.bbox_head.tasks]
    keys = [k for k in HEAD_ORDER if k in net.bbox_head.tasks[0].heads]
    test_cfg = dict(pc_range=list(tc.pc_range[:2]), voxel_size=list(tc.voxel_size[:2]),
                    out_size_factor=tc.out_size_factor, score_threshold=tc.score_threshold,
                    post_center_limit_range=list(tc.post_center_limit_range), min_radius=list(tc.min_radius),
                    nms={"nms_post_max_size": tc.nms.nms_post_max_size}, pre_max_size=tc.get("pre_max_size", 1000))
    meta = ModelMeta(
        voxel_size=list(vg.voxel_size), pc_range=list(vg.range), grid_size=list(det.voxel_generator.grid_size),
        max_points=vg.max_points_in_voxel, max_voxels=vg.max_voxel_num[1], max_pillars=int(max_pillars),
        num_point_features=det.num_point_features, pfn_channels=net.reader.pfn_layers[0].linear.in_features,
        bev_channels=cfg.model.neck.num_input_features, z_range=list(det.z_range), frame=asdict(det.frame),
        tasks=tasks, head_keys=keys, test_cfg=test_cfg, label_map=det.label_map.tolist(),
        subclass_map=det.subclass_map.tolist(), merged_nms={k: v for k, v in det.merged_nms.items()})
    meta.onnx["rpn"]["outputs"] = meta.rpn_outputs
    meta.onnx["pfn"]["dynamic"] = bool(dynamic_pillars)
    return meta


def postprocess(box9, scores, model_labels, label_map, subclass_map, frame, merged_nms, stamp=None):
    """Decoded boxes (model frame, det3d box9) -> dog-class Detections in the det frame, merged-class
    NMS applied (car and truck heads firing on one object, a second peak on a pedestrian)."""
    model_labels = np.asarray(model_labels).astype(np.int64)
    label_map, subclass_map = np.asarray(label_map, np.int64), np.asarray(subclass_map, np.int64)
    labels = label_map[model_labels]
    keep = labels >= 0
    std, vel = det3d_to_standard(box9[keep])
    std, vel = frame.boxes_from_model(std, vel)
    dets = Detections(std, vel, np.asarray(scores)[keep].astype(np.float64), labels[keep], "det", stamp,
                      subclass_map[model_labels[keep]])
    return classwise_circle_nms(dets, {k: tuple(v) if isinstance(v, list) else v for k, v in merged_nms.items()})
