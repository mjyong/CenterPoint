"""Per-stage shapes, parameters, MACs and latency of a CenterPoint detector on one frame.

    python tools/dog/profile_detector.py --ckpt work_dirs/pretrained/nusc_voxel.pth \
        --frame data/runs/rec_moving/frames/1789980192.298542.npy --device cuda:0 [--layers] [--tta]

``--frame`` is a model-input frame saved by ``run_sequence.py --save-frames`` (detection frame,
x y z intensity dt). The config comes from the checkpoint (``checkpoint_arch``) unless ``--config``
names a preset (PRESET_CONFIGS: pillar, voxel, dog_pillar, ...) or a det3d config file; without ``--ckpt`` the
weights are random, which is fine for shapes, parameters and MACs.

Dense convs count ``output elements x C_in x k x k`` multiply-adds. A sparse conv counts its real
gather-gemm-scatter work, ``(input site, output site, kernel offset) triples x C_in x C_out``, so the
sparse backbone's MACs depend on the frame. Latency: median over ``--runs`` after two warm-up runs,
with ``torch.cuda.synchronize`` around every stage; ``predict`` is the rest of ``net(example)``
(decoding and circle NMS). Numbers behind docs/CENTERPOINT_NETWORK.md and
docs/CENTERPOINT_VOXEL_NETWORK.md.

Voxel on the CPU: spconv 2.x refuses a fused bias there ("cpu don't support act and bias"); in
train mode it adds the bias after the gemm instead (same result), so the sparse convs are switched
to train mode (BatchNorm stays in eval). spconv's CPU path is not deterministic with several
threads, so the CPU run is single-threaded.
"""
import argparse
import itertools
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection import PRESET_CONFIGS, build_detector  # noqa: E402
from dog_perception.detection.centerpoint import checkpoint_arch  # noqa: E402

STAGES = ("reader", "backbone", "neck", "bbox_head")


def sparse_pairs(indices, spatial_shape, kernel, stride, padding, subm):
    """Number of (input site, output site, kernel offset) triples of one 3D sparse conv.

    ``indices``: (N, 4) int batch, z, y, x of the input sites. A submanifold conv keeps the input
    sites as outputs and pairs each with its occupied neighbours; a regular sparse conv pairs every
    input site with each output position whose kernel window covers it."""
    idx = np.asarray(indices, np.int64)
    k, s, p = (np.broadcast_to(np.asarray(v, np.int64), (3,)) for v in (kernel, stride, padding))
    if subm:
        dims = np.asarray(spatial_shape, np.int64) + k       # room for the offsets, no wrap-around

        def key(a):
            h = a[:, 0]
            for d in range(3):
                h = h * dims[d] + a[:, d + 1] + k[d] // 2
            return h
        base = np.sort(key(idx))
        n = 0
        for off in itertools.product(*[range(-(kk // 2), kk // 2 + 1) for kk in k]):
            q = key(idx + np.r_[0, off])
            pos = np.searchsorted(base, q).clip(max=len(base) - 1)
            n += int((base[pos] == q).sum())
        return n
    out_dim = (np.asarray(spatial_shape, np.int64) + 2 * p - k) // s + 1
    per = np.ones(len(idx), np.int64)
    for d in range(3):
        v = idx[:, d + 1, None] + p[d] - np.arange(k[d])     # (N, k): o * stride for each kernel tap
        per *= ((v >= 0) & (v % s[d] == 0) & (v // s[d] < out_dim[d])).sum(1)
    return int(per.sum())


def _shape(o):
    from spconv.pytorch import SparseConvTensor
    if isinstance(o, SparseConvTensor):
        return "sparse %d sites x %d ch, grid (z,y,x) %s" % (o.features.shape[0], o.features.shape[1],
                                                            "x".join(str(int(v)) for v in o.spatial_shape))
    if isinstance(o, (tuple, list)):
        return "%d x {%s}" % (len(o), _shape(o[0])) if isinstance(o[0], dict) else _shape(o[0])
    if isinstance(o, dict):
        return " ".join("%s%s" % (k, tuple(v.shape[1:2])) for k, v in o.items()) + " @%s" % "x".join(
            str(v) for v in next(iter(o.values())).shape[2:])
    return str(tuple(o.shape)) if torch.is_tensor(o) else type(o).__name__


class Profiler:
    def __init__(self, net, sync):
        self.net, self.sync = net, sync
        self.layers, self.outputs, self.times, self._t0 = [], OrderedDict(), {}, {}
        try:
            from spconv.pytorch.conv import SparseConvolution
        except ImportError:
            SparseConvolution = ()
        self.sparse_cls = SparseConvolution
        for name, m in net.named_modules():
            if SparseConvolution and isinstance(m, SparseConvolution):
                m.register_forward_hook(self._sparse_hook(name))
            elif isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                m.register_forward_hook(self._dense_hook(name))
            if name in self.stage_names():
                m.register_forward_hook(self._output_hook(name))
        for name in STAGES:
            mod = getattr(net, name)
            mod.register_forward_pre_hook(self._tic(name))
            mod.register_forward_hook(self._toc(name))
        self.record = True

    def stage_names(self):
        names = list(STAGES)
        for parent in ("backbone", "neck", "bbox_head"):
            for n, m in getattr(self.net, parent).named_children():
                if isinstance(m, torch.nn.ModuleList):
                    names += ["%s.%s.%s" % (parent, n, i) for i in range(len(m))]
                else:
                    names.append("%s.%s" % (parent, n))
        return names

    def _sparse_hook(self, name):
        def h(m, inp, out):
            if self.record:
                x = inp[0]
                pairs = sparse_pairs(x.indices.cpu().numpy(), x.spatial_shape, m.kernel_size, m.stride,
                                     m.padding, m.subm)
                self.layers.append(dict(name=name, kind="SubM" if m.subm else "SpConv", cin=m.in_channels,
                                        cout=m.out_channels, k=tuple(m.kernel_size), s=tuple(m.stride),
                                        n_in=x.features.shape[0], n_out=out.features.shape[0], pairs=pairs,
                                        out=_shape(out), mac=pairs * m.in_channels * m.out_channels))
        return h

    def _dense_hook(self, name):
        def h(m, inp, out):
            if not self.record:
                return
            if isinstance(m, torch.nn.ConvTranspose2d):
                mac = inp[0].numel() * m.out_channels * int(np.prod(m.kernel_size)) // m.groups
            elif isinstance(m, torch.nn.Linear):
                mac = out.numel() * m.in_features
            else:
                mac = out.numel() * m.in_channels * int(np.prod(m.kernel_size)) // m.groups
            self.layers.append(dict(name=name, kind=type(m).__name__, mac=int(mac), out=_shape(out)))
        return h

    def _output_hook(self, name):
        def h(m, inp, out):
            if self.record:
                self.outputs[name] = _shape(out)
        return h

    def _tic(self, name):
        def h(m, inp):
            self.sync()
            self._t0[name] = time.perf_counter()
        return h

    def _toc(self, name):
        def h(m, inp, out):
            self.sync()
            self.times[name] = time.perf_counter() - self._t0[name]
        return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True, help="model-input frame .npy (run_sequence.py --save-frames)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None, help="preset (%s) or det3d config; default: from --ckpt"
                    % ", ".join(PRESET_CONFIGS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="double-flip TTA (batch of 4)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--layers", action="store_true", help="also print every conv layer")
    args = ap.parse_args()

    config = args.config or (checkpoint_arch(args.ckpt) if args.ckpt else "pillar")
    det = build_detector(config, checkpoint=args.ckpt, device=args.device, tta=args.tta)
    net, dev = det.net, det.device
    if dev.type == "cpu":
        torch.set_num_threads(1)
    prof = Profiler(net, torch.cuda.synchronize if dev.type == "cuda" else (lambda: None))
    if dev.type == "cpu" and prof.sparse_cls:
        for m in net.modules():
            if isinstance(m, prof.sparse_cls):
                m.train()

    points = np.load(args.frame)
    pts = det.frame.points_to_model(det._crop(points))
    net.test_cfg.double_flip = det.tta
    timings = []
    for r in range(args.runs + 2):
        prof.record = r == 0
        t0 = time.perf_counter()
        example = det.build_example(pts)
        prof.sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = net(example, return_loss=False)[0]
        prof.sync()
        t2 = time.perf_counter()
        if r >= 2:
            row = dict(voxelize=t1 - t0, **prof.times)
            row["predict"] = (t2 - t1) - sum(prof.times[s] for s in STAGES)
            row["network"] = t2 - t1
            timings.append(row)
        if r == 0:
            n_pts = example["num_points"].float()

    vg = det.voxel_generator
    print("config %s, checkpoint %s, device %s%s" % (config, args.ckpt, dev, ", TTA" if det.tta else ""))
    print("points %d -> %d in range; voxels %s; grid (x,y,z) %s; points per voxel mean %.2f, %.1f%% capped at %d"
          % (len(points), len(pts), tuple(example["voxels"].shape), list(vg.grid_size), n_pts.mean(),
             100 * (n_pts == example["voxels"].shape[1]).float().mean(), example["voxels"].shape[1]))
    print("detections (score >= %.2f): %d" % (net.test_cfg.score_threshold, len(out["scores"])))

    print("\n%-26s %10s %9s  %s" % ("stage", "params", "GMAC", "output"))
    for name in [n for n in prof.stage_names() if n in prof.outputs]:
        shape, mod = prof.outputs[name], net.get_submodule(name)
        params = sum(p.numel() for p in mod.parameters())
        mac = sum(l["mac"] for l in prof.layers if l["name"] == name or l["name"].startswith(name + "."))
        indent = "  " if "." in name else ""
        print("%-26s %10d %9.3f  %s" % (indent + name, params, mac / 1e9, shape))
    print("%-26s %10d %9.3f" % ("total", sum(p.numel() for p in net.parameters()),
                                sum(l["mac"] for l in prof.layers) / 1e9))

    if args.layers:
        print("\nconv layers")
        for l in prof.layers:
            if "pairs" in l:
                print("  %-26s %-6s %3d->%3d k%s s%s  sites %6d -> %6d  pairs %8d (%.1f per output)  %.3f GMAC  %s"
                      % (l["name"], l["kind"], l["cin"], l["cout"], l["k"], l["s"], l["n_in"], l["n_out"],
                         l["pairs"], l["pairs"] / max(l["n_out"], 1), l["mac"] / 1e9, l["out"]))
            else:
                print("  %-26s %-16s %.3f GMAC  %s" % (l["name"], l["kind"], l["mac"] / 1e9, l["out"]))

    med = {k: 1e3 * float(np.median([t[k] for t in timings])) for k in timings[0]}
    print("\nlatency, median of %d runs [ms]: " % len(timings) + "  ".join("%s %.1f" % kv for kv in med.items()))
    if dev.type == "cuda":
        print("peak CUDA memory %.0f MB" % (torch.cuda.max_memory_allocated(dev) / 2 ** 20))


if __name__ == "__main__":
    main()
