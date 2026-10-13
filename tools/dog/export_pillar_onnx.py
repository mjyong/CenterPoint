"""Export CenterPoint-Pillar as two static-shape ONNX graphs for TensorRT on Orin.

    pfn.onnx       (1, 10, P_max, N_max) -> (1, 64, P_max, 1)
    rpn_head.onnx  (1, 64, ny, nx)       -> raw head maps (reg, height, dim, rot, vel, hm per task)

Voxelization, pillar decoration, scatter and decoding stay outside the
engines (``dog_perception.detection.pillar_export`` / ``decode``). With
``--check`` the ONNX graphs are run with onnxruntime on a real cloud and
compared to the PyTorch model stage by stage and end to end. The script
prints the ``trtexec`` commands to build FP16 engines on the Orin itself
(engines are specific to the GPU and TensorRT version, so build them there).

    python tools/dog/export_pillar_onnx.py --config configs/dog/dog_centerpoint_pp_xt32.py \
        --checkpoint work_dirs/dog_pp/latest.pth --out deploy/ --check sample.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import inspect  # noqa: E402

from dog_perception.detection.centerpoint import CenterPointDetector  # noqa: E402
from dog_perception.detection.filters import filter_detections  # noqa: E402
from dog_perception.detection.decode import HEAD_ORDER, decode_centerpoint  # noqa: E402
from dog_perception.detection.pillar_export import (PFNExport, RPNHeadExport, decorate_pillars,  # noqa: E402
                                                    pad_pillars, scatter_pillars, split_head_outputs)


def _onnx_export(model, args, path, **kw):
    # torch >= 2.5 defaults to the dynamo exporter; keep the TorchScript one
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    torch.onnx.export(model, args, path, **kw)


def export(det, out_dir, max_pillars, opset=13):
    os.makedirs(out_dir, exist_ok=True)
    cfg = det.cfg
    net = det.net.cpu().eval()
    n_pts = cfg.voxel_generator.max_points_in_voxel
    c_in = net.reader.pfn_layers[0].linear.in_features
    pfn = PFNExport(net.reader, n_pts).eval()
    _onnx_export(pfn, torch.zeros(1, c_in, max_pillars, n_pts), os.path.join(out_dir, "pfn.onnx"),
                 input_names=["pillars"], output_names=["pillar_features"], opset_version=opset)
    nx, ny = int(det.voxel_generator.grid_size[0]), int(det.voxel_generator.grid_size[1])
    c_bev = cfg.model.neck.num_input_features
    rpn = RPNHeadExport(net).eval()
    keys = [k for k in HEAD_ORDER if k in net.bbox_head.tasks[0].heads]
    names = ["task%d_%s" % (t, k) for t in range(len(net.bbox_head.tasks)) for k in keys]
    _onnx_export(rpn, torch.zeros(1, c_bev, ny, nx), os.path.join(out_dir, "rpn_head.onnx"),
                 input_names=["bev"], output_names=names, opset_version=opset)
    return keys


def run_onnx(det, out_dir, points_det, keys, max_pillars):
    import onnxruntime as ort
    cfg = det.cfg
    pts = det.frame.points_to_model(det._crop(points_det))[:, :det.num_point_features]
    v, c, n = det.voxel_generator.generate(pts)
    feats = decorate_pillars(v, n, c, cfg.voxel_generator.voxel_size, cfg.voxel_generator.range)
    x, cpad, nv = pad_pillars(feats, c, max_pillars)
    pfn = ort.InferenceSession(os.path.join(out_dir, "pfn.onnx"), providers=["CPUExecutionProvider"])
    pf = pfn.run(None, {"pillars": x})[0][0, :, :, 0]
    canvas = scatter_pillars(pf, cpad, nv, det.voxel_generator.grid_size)
    rpn = ort.InferenceSession(os.path.join(out_dir, "rpn_head.onnx"), providers=["CPUExecutionProvider"])
    outs = rpn.run(None, {"bev": canvas})
    maps = split_head_outputs(outs, keys, len(det.net.bbox_head.tasks))
    box9, scores, labels = decode_centerpoint(maps, det.net.bbox_head.num_classes, dict(cfg.test_cfg))

    # stage-by-stage reference: the unmodified PyTorch reader / scatter / neck / head
    with torch.no_grad():
        ex = det.build_example(pts)
        feat = det.net.reader(ex["voxels"], ex["num_points"], ex["coordinates"])
        bev = det.net.backbone(feat, ex["coordinates"], 1, ex["shape"][0])
        ref_preds, _ = det.net.bbox_head(det.net.neck(bev))
        ref = [t[k].numpy() for t in ref_preds for k in HEAD_ORDER if k in t]
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
    errors = {"pfn": rel(pf[:, :nv].T, feat.numpy()), "bev": rel(canvas, bev.numpy()),
              "heads": max(rel(a, b) for a, b in zip(outs, ref))}
    dets = filter_detections(det.postprocess(box9, scores, labels), points_det, det.filter_cfg) \
        if det.filter_cfg is not None else det.postprocess(box9, scores, labels)
    return dets, nv, len(v), errors


def trtexec_commands(out_dir, fp16=True):
    flag = " --fp16" if fp16 else ""
    return ["trtexec --onnx=%s --saveEngine=%s%s" % (os.path.join(out_dir, f + ".onnx"),
                                                    os.path.join(out_dir, f + ".engine"), flag)
            for f in ("pfn", "rpn_head")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default: from --checkpoint (nuScenes or dog model)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="deploy/centerpoint_pillar")
    ap.add_argument("--max-pillars", type=int, default=30000)
    ap.add_argument("--xy-range", type=float, default=40.8)
    ap.add_argument("--check", default=None, help="det-frame cloud (.npy, N x 5) to verify ONNX vs PyTorch")
    args = ap.parse_args()

    if args.config is None:
        from dog_perception.detection import PRESET_CONFIGS
        from dog_perception.detection.centerpoint import checkpoint_arch
        args.config = PRESET_CONFIGS[(checkpoint_arch(args.checkpoint) if args.checkpoint else None) or "pillar"]
    det = CenterPointDetector(args.config, args.checkpoint, device="cpu", xy_range=args.xy_range)
    keys = export(det, args.out, args.max_pillars)
    print("exported", sorted(os.listdir(args.out)))
    print("build the engines on the Orin (FP16 first; INT8 needs calibration frames):")
    for c in trtexec_commands(args.out):
        print("  " + c)
    if args.check:
        pts = np.load(args.check)
        ref = det(pts)
        got, nv, total, errors = run_onnx(det, args.out, pts, keys, args.max_pillars)
        if total > args.max_pillars:
            print("WARNING: %d pillars > max_pillars=%d, truncated" % (total, args.max_pillars))
        print("pillars used: %d / %d" % (nv, args.max_pillars))
        print("max |onnx - pytorch| / max |pytorch|: " + ", ".join("%s %.1e" % kv for kv in errors.items()))
        print("pytorch: %d dets, onnx+numpy decode: %d dets" % (len(ref), len(got)))
        if len(ref) and len(got) == len(ref):
            a, b = np.argsort(-ref.scores), np.argsort(-got.scores)
            print("max |score diff| %.2e, max |center diff| %.2e m"
                  % (np.abs(ref.scores[a] - got.scores[b]).max(),
                     np.abs(ref.boxes[a, :3] - got.boxes[b, :3]).max()))


if __name__ == "__main__":
    main()
