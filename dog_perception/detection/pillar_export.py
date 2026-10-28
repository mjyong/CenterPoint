"""Deployment split of CenterPoint-Pillar for NVIDIA Orin (TensorRT).

    host / CUDA : voxelize -> pillar decoration (10 features)   [numpy reference here]
    TensorRT    : PFN as 1x1 Conv2d + BN + ReLU + max-pool       [pfn.onnx]
    host / CUDA : scatter pillars into the BEV canvas            [numpy reference here]
    TensorRT    : RPN neck + CenterHead (dense 2D convs only)    [rpn_head.onnx]
    host / CUDA : decode + circle NMS                            [decode.py]

This is the same two-engine split the community CenterPoint-Pillar
TensorRT ports use (pfe + rpn). The PFN's ``Linear/BatchNorm1d`` is rewritten
as ``Conv2d(1x1)/BatchNorm2d`` over a fixed ``(1, C, P_max, N_max)`` tensor so
both engines have static shapes and only plain dense ops (FP16 / INT8 friendly).
The numpy steps (``pillar_host.py``, no PyTorch) are the reference implementation; on Orin they belong
on the GPU (CUDA kernels, deploy/cpp) because the ARM cores are the bottleneck.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .decode import HEAD_ORDER
# the NumPy host side lives in pillar_host.py (no PyTorch); re-exported here for the older imports
from .pillar_host import decorate_pillars, pad_pillars, scatter_pillars, split_head_outputs  # noqa: F401


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
    """Neck + CenterHead; returns the raw maps task by task in ``HEAD_ORDER``.

    The RPN's ``ZeroPad2d(1) + Conv(padding=0)`` pairs are folded into
    ``Conv(padding=1)`` (identical result) on a copy of the neck, so the graph
    has no Pad op whose pads are computed by a Shape/Slice/Cast subgraph,
    which older TensorRT ONNX parsers reject.
    """

    def __init__(self, detector):
        super().__init__()
        self.neck = fold_zero_pad(copy.deepcopy(detector.neck))
        self.head = detector.bbox_head

    def forward(self, canvas):
        preds, _ = self.head(self.neck(canvas))
        outs = []
        for task in preds:
            outs += [task[k] for k in HEAD_ORDER if k in task]
        return tuple(outs)


def fold_zero_pad(module):
    """Replace ZeroPad2d(p) followed by Conv2d(padding=0) with Conv2d(padding=p) in place."""
    for seq in module.modules():
        children = list(seq.children())
        for a, b in zip(children, children[1:]):
            if isinstance(a, nn.ZeroPad2d) and isinstance(b, nn.Conv2d) and tuple(b.padding) == (0, 0):
                pad = a.padding
                if len(set(pad)) == 1:
                    b.padding = (pad[0], pad[0])
                    for name, child in seq.named_children():
                        if child is a:
                            setattr(seq, name, nn.Identity())
    return module
