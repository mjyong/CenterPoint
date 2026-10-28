"""INT8 post-training quantization of the exported Pillar graphs (NVIDIA ModelOpt, explicit Q/DQ).

TensorRT 10 builds INT8 engines from graphs that carry their own scales (Q/DQ nodes); the old
implicit calibrators are deprecated. This script computes the scales on real frames and writes
``<graph>.int8.onnx`` next to the FP32 graph; ``build_engines.py --precision int8`` turns it into
``<graph>.int8.engine`` on the server (accuracy) and on the Orin (latency, the deployed one).

    python tools/deploy/quantize_int8.py --model-dir data/deploy/nusc_pp \\
        --frames "data/runs/<train scene>/frames/*.npy" ... --num 128 --method entropy
    # keep the heatmap convolutions in FP16 (mixed precision) if INT8 costs detections
    python tools/deploy/quantize_int8.py ... --exclude ".*/hm/.*" --suffix int8_hmfp16

Calibration frames come from the **training** scenes (the validation scenes score the result). The
inputs of the graph (the BEV canvas for ``rpn_head``, the padded pillars for ``pfn``) are computed
with the FP32 ONNX graphs (detection/exported.py). ``rpn_head`` is where the time goes; the PFN is
small and stays FP16 by default. Memory: one BEV canvas is 42.6 MB (64 x 408 x 408 float32), 128
of them 5.4 GB.

Written against ``modelopt.onnx.quantization.quantize`` of nvidia-modelopt 0.47.0 (the server's):
one input array of N frames is split into N calibration steps; ``nodes_to_exclude`` are regexes
matched with ``re.match`` from the start of the node name; layers left out of INT8 are stored in FP16
(its default ``high_precision_dtype``) while the graph inputs and outputs stay FP32; the output
graph has opset 19.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.exported import ExportedPillarDetector  # noqa: E402

INPUT_STAGE = {"rpn_head": "bev", "pfn": "pfn_in"}


def calibration_inputs(model_dir, files, graph):
    det = ExportedPillarDetector(model_dir, "onnx", filter_cfg=None)
    out = []
    for k, f in enumerate(files):
        pts = det.frame.points_to_model(det._crop(np.load(f)))
        _, stages, _ = det.network(pts)
        out.append(stages[INPUT_STAGE[graph]].astype(np.float32))
        if (k + 1) % 16 == 0:
            print("calibration inputs: %d / %d" % (k + 1, len(files)), flush=True)
    return np.concatenate(out, axis=0)


def matching_nodes(onnx_path, patterns):
    """Node names each pattern excludes, matched as ModelOpt does (``re.match``)."""
    import onnx
    names = [n.name for n in onnx.load(onnx_path, load_external_data=False).graph.node]
    return {p: [n for n in names if re.match(p, n)] for p in patterns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="det-frame clouds of training scenes (globs)")
    ap.add_argument("--num", type=int, default=128, help="calibration frames, spread over --frames")
    ap.add_argument("--graph", default="rpn_head", choices=sorted(INPUT_STAGE))
    ap.add_argument("--method", default="entropy", choices=["entropy", "max"])
    ap.add_argument("--exclude", nargs="*", default=[], help="node-name regexes kept in high precision")
    ap.add_argument("--suffix", default="int8", help="output <graph>.<suffix>.onnx (build_engines reads .int8.onnx)")
    ap.add_argument("--eps", default="cuda:0,cpu", help="ONNX Runtime execution providers for calibration")
    args = ap.parse_args()

    import modelopt
    from modelopt.onnx.quantization import quantize

    files = sorted(f for p in args.frames for f in glob.glob(p))
    if not files:
        raise SystemExit("no frames match %s" % args.frames)
    pick = np.linspace(0, len(files) - 1, min(args.num, len(files))).round().astype(int)
    files = [files[i] for i in np.unique(pick)]
    src = os.path.join(args.model_dir, args.graph + ".onnx")
    dst = os.path.join(args.model_dir, "%s.%s.onnx" % (args.graph, args.suffix))
    if args.exclude:
        hits = matching_nodes(src, args.exclude)
        for p, n in hits.items():
            print("exclude %r: %d nodes%s" % (p, len(n), (" (e.g. %s)" % n[0]) if n else " -- matches nothing"))
    t0 = time.time()
    data = calibration_inputs(args.model_dir, files, args.graph)
    t1 = time.time()
    quantize(onnx_path=src, quantize_mode="int8", calibration_data=data, calibration_method=args.method,
             calibration_eps=args.eps.split(","), nodes_to_exclude=args.exclude or None, output_path=dst)
    info = dict(graph=args.graph, output=os.path.basename(dst), method=args.method, exclude=args.exclude,
                excluded_nodes=sum(len(n) for n in hits.values()) if args.exclude else 0,
                modelopt=modelopt.__version__, frames=len(files), first=files[0], last=files[-1],
                inputs_s=round(t1 - t0, 1), quantize_s=round(time.time() - t1, 1))
    print(json.dumps(info, indent=1))
    log = os.path.join(args.model_dir, "quantize.json")
    entries = json.load(open(log)) if os.path.exists(log) else []
    with open(log, "w") as f:
        json.dump(entries + [info], f, indent=1)


if __name__ == "__main__":
    main()
