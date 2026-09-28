"""Deployment split of CenterPoint-Pillar for BPU (RDK S100P) / TensorRT.

    CPU  : voxelize -> pillar decoration (10 features)       [numpy]
    NPU  : PFN as 1x1 Conv2d + BN + ReLU + max-pool           [pfn.onnx]
    CPU  : scatter pillars into the BEV canvas                [numpy]
    NPU  : RPN neck + CenterHead (dense 2D convs only)        [rpn_head.onnx]
    CPU  : decode + circle NMS                                [decode.py]

The PFN's ``Linear/BatchNorm1d`` is rewritten as ``Conv2d(1x1)/BatchNorm2d``
over a fixed ``(1, C, P_max, N_max)`` tensor so every op is a plain dense op
with a static shape, which is what the BPU quantizer wants.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .decode import HEAD_ORDER


def decorate_pillars(voxels, num_points, coords, voxel_size, pc_range):
    """NumPy port of ``PillarFeatureNet`` feature decoration.

    voxels (P, N, C), num_points (P,), coords (P, 3) as (z, y, x).
    Returns (P, N, C + 5): [raw, xyz - pillar mean, xy - pillar center].
    """
    voxels = voxels.astype(np.float32)
    P, N, _ = voxels.shape
    mean = voxels[:, :, :3].sum(axis=1, keepdims=True) / np.maximum(num_points, 1)[:, None, None]
    f_cluster = voxels[:, :, :3] - mean
    cx = coords[:, 2].astype(np.float32) * voxel_size[0] + voxel_size[0] / 2 + pc_range[0]
    cy = coords[:, 1].astype(np.float32) * voxel_size[1] + voxel_size[1] / 2 + pc_range[1]
    f_center = np.stack([voxels[:, :, 0] - cx[:, None], voxels[:, :, 1] - cy[:, None]], axis=-1)
    feats = np.concatenate([voxels, f_cluster, f_center], axis=-1)
    mask = np.arange(N)[None, :] < num_points[:, None]
    return (feats * mask[..., None]).astype(np.float32)


def pad_pillars(features, coords, max_pillars):
    """Pad / truncate to a static pillar count. Returns (1, C, P, N) and coords."""
    P, N, C = features.shape
    out = np.zeros((max_pillars, N, C), dtype=np.float32)
    n = min(P, max_pillars)
    out[:n] = features[:n]
    c = np.zeros((max_pillars, 3), dtype=np.int64)
    c[:n] = coords[:n]
    return out.transpose(2, 0, 1)[None], c, n


def scatter_pillars(pillar_features, coords, num_valid, grid_size):
    """(C, P) features -> (1, C, ny, nx) BEV canvas."""
    nx, ny = int(grid_size[0]), int(grid_size[1])
    C = pillar_features.shape[0]
    canvas = np.zeros((C, ny * nx), dtype=np.float32)
    idx = coords[:num_valid, 1] * nx + coords[:num_valid, 2]
    canvas[:, idx] = pillar_features[:, :num_valid]
    return canvas.reshape(1, C, ny, nx)


class PFNExport(nn.Module):
    """``PillarFeatureNet`` layers as Conv2d(1x1) over (1, C, P, N)."""

    def __init__(self, pillar_feature_net, max_points):
        super().__init__()
        self.max_points = max_points
        self.convs, self.bns, self.last = nn.ModuleList(), nn.ModuleList(), []
        for layer in pillar_feature_net.pfn_layers:
            lin, bn1d = layer.linear, layer.norm
            conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
            conv.weight.data.copy_(lin.weight.data[:, :, None, None])
            if lin.bias is not None:
                conv.bias.data.copy_(lin.bias.data)
            bn = nn.BatchNorm2d(bn1d.num_features, eps=bn1d.eps, momentum=bn1d.momentum)
            bn.load_state_dict(bn1d.state_dict())
            self.convs.append(conv)
            self.bns.append(bn)
            self.last.append(layer.last_vfe)

    def forward(self, x):
        for conv, bn, last in zip(self.convs, self.bns, self.last):
            y = F.relu(bn(conv(x)))
            y_max = F.max_pool2d(y, kernel_size=(1, self.max_points))
            x = y_max if last else torch.cat([y, y_max.repeat(1, 1, 1, self.max_points)], dim=1)
        return x  # (1, C_out, P, 1)


class RPNHeadExport(nn.Module):
    """Neck + CenterHead; returns the raw maps task by task in ``HEAD_ORDER``."""

    def __init__(self, detector):
        super().__init__()
        self.neck = detector.neck
        self.head = detector.bbox_head

    def forward(self, canvas):
        preds, _ = self.head(self.neck(canvas))
        outs = []
        for task in preds:
            outs += [task[k] for k in HEAD_ORDER if k in task]
        return tuple(outs)


def split_head_outputs(outputs, head_keys, num_tasks):
    """Flat tuple of (1, C, H, W) maps -> list of per-task dicts of (C, H, W)."""
    k = len(head_keys)
    return [{key: np.asarray(outputs[t * k + i])[0] for i, key in enumerate(head_keys)}
            for t in range(num_tasks)]
