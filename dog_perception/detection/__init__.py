from .boxes import (CLASSES, NUSC_TO_DOG, Detections, circle_nms, class_index, classwise_circle_nms,
                    det3d_to_standard, score_threshold_array, standard_to_det3d)
from .filters import DEFAULT_SCORE_THRESHOLDS, DetectionFilterConfig, filter_detections
from .frame_adapter import NUSC_LIDAR_HEIGHT, ModelFrame
from .oracle import OracleDetector, OracleNoise


PRESET_CONFIGS = {
    "pillar": "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py",
    "voxel": "configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py",
    "dog_pillar": "configs/dog/dog_centerpoint_pp_xt32.py",
    "dog_voxel": "configs/dog/dog_centerpoint_voxel_xt32.py",
}


def build_detector(kind, **kwargs):
    """kind: a PRESET_CONFIGS key or a path to a det3d config. Lazily imports det3d."""
    import os
    from .centerpoint import CenterPointDetector
    if kind in PRESET_CONFIGS:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        kind = os.path.join(root, PRESET_CONFIGS[kind])
    return CenterPointDetector(kind, **kwargs)
