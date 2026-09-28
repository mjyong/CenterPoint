"""Export CenterPoint-Pillar as two static-shape ONNX graphs for BPU / TensorRT.

    pfn.onnx       (1, 10, P_max, N_max) -> (1, 64, P_max, 1)
    rpn_head.onnx  (1, 64, ny, nx)       -> raw head maps (reg, height, dim, rot, vel, hm per task)

Voxelization, pillar decoration, scatter and decoding stay on the CPU
(``dog_perception.detection.pillar_export`` / ``decode``). With ``--check``
the ONNX graphs are run with onnxruntime on a real cloud and compared to the
PyTorch model end to end.

    python tools/dog/export_pillar_onnx.py --config configs/dog/dog_centerpoint_pp_xt32.py \
        --checkpoint work_dirs/dog_pp/latest.pth --out deploy/ --check sample.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.centerpoint import CenterPointDetector  # noqa: E402
from dog_perception.detection.decode import HEAD_ORDER, decode_centerpoint  # noqa: E402
from dog_perception.detection.pillar_export import (PFNExport, RPNHeadExport, decorate_pillars,  # noqa: E402
                                                    pad_pillars, scatter_pillars, split_head_outputs)


def export(det, out_dir, max_pillars, opset=13):
    os.makedirs(out_dir, exist_ok=True)
    cfg = det.cfg
    net = det.net.cpu().eval()
    n_pts = cfg.voxel_generator.max_points_in_voxel
    c_in = net.reader.pfn_layers[0].linear.in_features
    pfn = PFNExport(net.reader, n_pts).eval()
    torch.onnx.export(pfn, torch.zeros(1, c_in, max_pillars, n_pts), os.path.join(out_dir, "pfn.onnx"),
                      input_names=["pillars"], output_names=["pillar_features"], opset_version=opset,
                      dynamo=False)
    nx, ny = int(det.voxel_generator.grid_size[0]), int(det.voxel_generator.grid_size[1])
    c_bev = cfg.model.neck.num_input_features
    rpn = RPNHeadExport(net).eval()
    keys = [k for k in HEAD_ORDER if k in net.bbox_head.tasks[0].heads]
    names = ["task%d_%s" % (t, k) for t in range(len(net.bbox_head.tasks)) for k in keys]
    torch.onnx.export(rpn, torch.zeros(1, c_bev, ny, nx), os.path.join(out_dir, "rpn_head.onnx"),
                      input_names=["bev"], output_names=names, opset_version=opset, dynamo=False)
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

    # map-level reference: the unmodified PyTorch reader/scatter/neck/head on the same pillars
    with torch.no_grad():
        ex = det.build_example(pts)
        feat = det.net.reader(ex["voxels"], ex["num_points"], ex["coordinates"])
        bev = det.net.backbone(feat, ex["coordinates"], 1, ex["shape"][0])
        ref = [o.numpy() for o in RPNHeadExport(det.net).eval()(bev)]
    map_diff = max(float(np.abs(a - b).max()) for a, b in zip(outs, ref))
    return det.postprocess(box9, scores, labels), nv, len(v), map_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="deploy/centerpoint_pillar")
    ap.add_argument("--max-pillars", type=int, default=30000)
    ap.add_argument("--xy-range", type=float, default=40.8)
    ap.add_argument("--check", default=None, help="det-frame cloud (.npy, N x 5) to verify ONNX vs PyTorch")
    args = ap.parse_args()

    det = CenterPointDetector(args.config, args.checkpoint, device="cpu", xy_range=args.xy_range)
    keys = export(det, args.out, args.max_pillars)
    print("exported", os.listdir(args.out))
    if args.check:
        pts = np.load(args.check)
        ref = det(pts)
        got, nv, total, map_diff = run_onnx(det, args.out, pts, keys, args.max_pillars)
        if total > args.max_pillars:
            print("WARNING: %d pillars > max_pillars=%d, truncated" % (total, args.max_pillars))
        print("pillars used: %d / %d" % (nv, args.max_pillars))
        print("max |onnx - pytorch| over all head maps: %.2e" % map_diff)
        print("pytorch: %d dets, onnx+numpy decode: %d dets" % (len(ref), len(got)))
        if len(ref) and len(got) == len(ref):
            a, b = np.argsort(-ref.scores), np.argsort(-got.scores)
            print("max |score diff| %.2e, max |center diff| %.2e m"
                  % (np.abs(ref.scores[a] - got.scores[b]).max(),
                     np.abs(ref.boxes[a, :3] - got.boxes[b, :3]).max()))


if __name__ == "__main__":
    main()
