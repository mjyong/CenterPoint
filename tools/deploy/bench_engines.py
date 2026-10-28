"""Latency of TensorRT engines already built, on the machine that runs them: what
``trtexec --loadEngine`` calls GPU Compute Time, where there is no trtexec (the server's pip
TensorRT; the Orin has /usr/src/tensorrt/bin/trtexec since T25).

    python3 tools/deploy/bench_engines.py --model-dir ~/dogperc/models/nusc_pp \\
        --engines pfn.fp16 rpn_head.fp16 rpn_head.int8 --runs 200 --out bench_orin.json

Each engine runs on random inputs (timing does not depend on the values): inputs copied once,
``--warmup-ms`` of warm-up, then ``--runs`` enqueues timed with CUDA events (median, p99, mean, min,
max). ``host_median_ms`` is one ``TrtRunner`` call with its input and output copies, which the C++
pipeline does not pay. Run with the clocks locked (``sudo jetson_clocks``) and nothing else on the GPU.

A dynamic engine (the PFN of ``export_pillar_onnx.py --dynamic-pillars``) runs at ``--shapes``, written
as for trtexec (``--shapes pillars:1x10x10000x20``), else at the optimum of its profile:

    python3 tools/deploy/bench_engines.py --model-dir ~/dogperc/models/nusc_pp_dyn --engines pfn.fp16 \
        --shapes pillars:1x10x10000x20
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from build_engines import machine  # noqa: E402
from dog_perception.detection.exported import TrtRunner  # noqa: E402


def parse_shapes(text):
    """``"a:1x10x100x20,b:1x3"`` -> {"a": (1, 10, 100, 20), "b": (1, 3)} (trtexec's --shapes)."""
    out = {}
    for item in filter(None, (text or "").split(",")):
        name, dims = item.rsplit(":", 1)
        out[name] = tuple(int(d) for d in dims.split("x"))
    return out


def engine_paths(model_dir, names):
    """``names``: engine file names with or without ``.engine`` (``pfn.fp16``); none: every engine."""
    if not names:
        return sorted(glob.glob(os.path.join(model_dir, "*.engine")))
    return [os.path.join(model_dir, n if n.endswith(".engine") else n + ".engine") for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--engines", nargs="*", default=None, help="default: every *.engine in --model-dir")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup-ms", type=float, default=500.0)
    ap.add_argument("--shapes", default=None, help="dynamic inputs, name:AxBxC[,...]; default: profile optimum")
    ap.add_argument("--out", default=None, help="JSON report")
    args = ap.parse_args()

    paths = engine_paths(args.model_dir, args.engines)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing or not paths:
        raise SystemExit("no engine: %s" % (", ".join(missing) or os.path.join(args.model_dir, "*.engine")))
    shapes = parse_shapes(args.shapes)
    results = []
    print("| engine | input shapes | GPU median ms | GPU p99 ms | GPU mean ms | min | max | host median ms |")
    print("|---|---|---|---|---|---|---|---|")
    for p in paths:
        runner = TrtRunner(p)
        r = runner.benchmark(args.runs, args.warmup_ms, shapes={n: v for n, v in shapes.items() if n in runner.dynamic})
        r["engine"] = os.path.basename(p)
        results.append(r)
        print("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            r["engine"], " ".join("x".join(map(str, v)) for v in r["shapes"].values()), r["gpu_median_ms"],
            r["gpu_p99_ms"], r["gpu_mean_ms"], r["gpu_min_ms"], r["gpu_max_ms"], r["host_median_ms"]), flush=True)
    if args.out:
        import tensorrt as trt
        report = dict(machine=machine(), trt=trt.__version__, runs=args.runs, warmup_ms=args.warmup_ms, shapes=shapes,
                      date=time.strftime("%Y-%m-%d %H:%M:%S"), engines=results)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
