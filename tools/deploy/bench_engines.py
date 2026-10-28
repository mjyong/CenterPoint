"""Latency of TensorRT engines already built, on the machine that runs them: what
``trtexec --loadEngine`` calls GPU Compute Time, where there is no trtexec (the server's pip
TensorRT; the Orin has /usr/src/tensorrt/bin/trtexec since T25).

    python3 tools/deploy/bench_engines.py --model-dir ~/dogperc/models/nusc_pp \\
        --engines pfn.fp16 rpn_head.fp16 rpn_head.int8 --runs 200 --out bench_orin.json

Each engine runs on random inputs (timing does not depend on the values): inputs copied once,
``--warmup-ms`` of warm-up, then ``--runs`` enqueues timed with CUDA events (median, p99, mean, min,
max). ``host_median_ms`` is one ``TrtRunner`` call with its input and output copies, which the C++
pipeline does not pay. Run with the clocks locked (``sudo jetson_clocks``) and nothing else on the GPU.
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
    ap.add_argument("--out", default=None, help="JSON report")
    args = ap.parse_args()

    paths = engine_paths(args.model_dir, args.engines)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing or not paths:
        raise SystemExit("no engine: %s" % (", ".join(missing) or os.path.join(args.model_dir, "*.engine")))
    results = []
    print("| engine | GPU median ms | GPU p99 ms | GPU mean ms | min | max | host median ms |")
    print("|---|---|---|---|---|---|---|")
    for p in paths:
        r = TrtRunner(p).benchmark(args.runs, args.warmup_ms)
        r["engine"] = os.path.basename(p)
        results.append(r)
        print("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            r["engine"], r["gpu_median_ms"], r["gpu_p99_ms"], r["gpu_mean_ms"], r["gpu_min_ms"], r["gpu_max_ms"],
            r["host_median_ms"]), flush=True)
    if args.out:
        import tensorrt as trt
        report = dict(machine=machine(), trt=trt.__version__, runs=args.runs, warmup_ms=args.warmup_ms,
                      date=time.strftime("%Y-%m-%d %H:%M:%S"), engines=results)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
