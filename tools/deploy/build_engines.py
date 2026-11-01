"""Build TensorRT engines from the exported Pillar graphs, on the machine that will run them.

Engines are tied to the GPU and the TensorRT version: build x86 engines on the server (accuracy
checks) and Orin engines on the Orin (latency, the deployed ones). Output names follow
``dog_perception/detection/exported.py``: ``<graph>.<precision>.engine``.

    # FP16 (and an FP32 reference) for both graphs
    python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --precision fp32,fp16
    # INT8 from a Q/DQ graph (tools/deploy/quantize_int8.py): rpn_head.int8.onnx -> rpn_head.int8.engine;
    # any int8* name works the same (rpn_head.int8_hmfp16.onnx -> rpn_head.int8_hmfp16.engine)
    python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --precision int8 --graphs rpn_head
    # the RPN on a DLA core (Orin), layers DLA cannot run fall back to the GPU
    python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --precision fp16 --graphs rpn_head --dla 0

``--benchmark N`` times every new engine N times on random inputs (GPU time from CUDA events, median
and p99: ``TrtRunner.benchmark``; ``tools/deploy/bench_engines.py`` times engines already built, the
same number as trtexec's GPU Compute Time, also where there is no trtexec). Each build appends to
``<model-dir>/engines.json``: TensorRT version, machine, graph, precision, build time, size.
The same build is also copied to ``<model-dir>/machines/<host>/`` so a later build on another
machine does not replace it. Deploy that directory for the machine that will run it
(``deploy/scripts/sync_to_orin.sh --model <dir> --engines <host>``); ``--stage <host>`` writes
one directory (ONNX, model_meta and that host's engines) for ``build_deb.sh --model``.

Graphs are parsed from their file, so ONNX weights stored in an external file next to the graph
(torch's dynamo exporter writes ``<name>.onnx.data``) are found whatever the working directory.

A PFN exported with ``--dynamic-pillars`` gets one optimization profile: 1 .. ``max_pillars`` pillars
(model_meta.json), kernels tuned for ``--opt-pillars`` (default: half of max_pillars, near the
median frame of the 0921 data).
"""
import argparse
import json
import os
import platform
import re
import shutil
import socket
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

GRAPHS = ("pfn", "rpn_head")


def is_int8(precision):
    return precision.startswith("int8")


def onnx_for(model_dir, graph, precision):
    """INT8 engines come from the Q/DQ graph quantize_int8.py writes (<graph>.<precision>.onnx); the
    others from the plain export."""
    if is_int8(precision):
        return os.path.join(model_dir, "%s.%s.onnx" % (graph, precision))
    return os.path.join(model_dir, "%s.onnx" % graph)


def build(onnx_path, engine_path, precision, workspace_mb=4096, dla=None, timing_cache=None, profile=None):
    """``profile``: {input name: (min shape, opt shape, max shape)}, required for every dynamic input."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)            # TensorRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(os.path.abspath(onnx_path)):     # external weights: next to the graph
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("%s: ONNX parse failed:\n  %s" % (onnx_path, "\n  ".join(errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    dynamic = [t for t in inputs if any(d < 0 for d in t.shape)]
    if dynamic:
        missing = [t.name for t in dynamic if t.name not in (profile or {})]
        if missing:
            raise ValueError("%s: dynamic inputs %s need a profile (min, opt, max)" % (onnx_path, missing))
        opt_profile = builder.create_optimization_profile()
        for t in dynamic:
            opt_profile.set_shape(t.name, *[tuple(int(d) for d in x) for x in profile[t.name]])
        config.add_optimization_profile(opt_profile)
    if precision == "fp16" or is_int8(precision):
        config.set_flag(trt.BuilderFlag.FP16)        # INT8: layers left unquantized run in FP16
    if is_int8(precision):
        config.set_flag(trt.BuilderFlag.INT8)        # scales come from the graph's Q/DQ nodes
    if dla is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    cache = None
    if timing_cache:
        blob = open(timing_cache, "rb").read() if os.path.exists(timing_cache) else b""
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("%s: TensorRT build failed (see the log above)" % onnx_path)
    with open(engine_path, "wb") as f:
        f.write(plan)
    if cache is not None:
        with open(timing_cache, "wb") as f:
            f.write(cache.serialize())
    info = dict(trt=trt.__version__, seconds=round(time.time() - t0, 1), bytes=os.path.getsize(engine_path))
    if dynamic:
        info["profile"] = {t.name: [list(x) for x in profile[t.name]] for t in dynamic}
    return info


def pillar_profile(meta, opt_pillars=None):
    """Optimization profile of a dynamic PFN: 1 .. max_pillars pillars."""
    shape = lambda p: (1, meta.pfn_channels, int(p), meta.max_points)  # noqa: E731
    return {meta.onnx["pfn"]["input"]: (shape(1), shape(opt_pillars or meta.max_pillars // 2), shape(meta.max_pillars))}


def benchmark(engine_path, runs, warmup_ms=500.0, shapes=None):
    from dog_perception.detection.exported import TrtRunner
    return TrtRunner(engine_path).benchmark(runs, warmup_ms, shapes=shapes)


def machine():
    m = dict(host=socket.gethostname(), arch=platform.machine())
    model = "/proc/device-tree/model"
    if os.path.exists(model):                       # Jetson
        m["model"] = open(model).read().strip("\x00\n ")
    rel = "/etc/nv_tegra_release"
    if os.path.exists(rel):                         # distinguishes two Orins with the same TensorRT
        m["l4t"] = open(rel).read().strip()
    return m


def l4t_release(text):
    """"R36.4.4" from /etc/nv_tegra_release ("# R36 (release), REVISION: 4.4, GCID: ...") or a short form; None
    without one (not a Jetson, or an empty record)."""
    if not text:
        return None
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*(\d+(?:\.\d+)*)", text)
    if m:
        return "R%s.%s" % m.groups()
    m = re.search(r"R\d+(?:\.\d+)+", text)
    return m.group(0) if m else None


def check_target(log, arch, l4t_text):
    """The engines of a machines/<host>/engines.json load on the target: same arch and, when recorded, same L4T
    (engines only load on the TensorRT that built them; deploy/scripts/sync_to_orin.sh --engines)."""
    arches = {e.get("arch") for e in log}
    if arches != {arch}:
        raise SystemExit("engines are %s, target is %s" % (", ".join(sorted(map(str, arches))), arch))
    want = {l4t_release(e.get("l4t")) for e in log} - {None}
    got = l4t_release(l4t_text)
    if want and got not in want:
        raise SystemExit("engines are L4T %s, target is %s" % (", ".join(sorted(want)), got or "not a Jetson"))


def _host_dir(model_dir, host):
    return os.path.join(model_dir, "machines", host)


def archive_engine(model_dir, engine_path, info):
    """Keep this engine under machines/<host>/, replacing only that host's copy of the same file."""
    dest = _host_dir(model_dir, info["host"])
    os.makedirs(dest, exist_ok=True)
    shutil.copy2(engine_path, os.path.join(dest, os.path.basename(engine_path)))
    log_path = os.path.join(dest, "engines.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    log = [e for e in log if e.get("engine") != info["engine"]]
    log.append(info)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    print("  archived %s -> machines/%s/" % (info["engine"], info["host"]), flush=True)


def archive_current(model_dir):
    """File the engines sitting in model_dir under the host recorded for each file."""
    log_path = os.path.join(model_dir, "engines.json")
    if not os.path.exists(log_path):
        raise SystemExit("%s missing" % log_path)
    newest = {}
    for entry in json.load(open(log_path)):
        newest[entry["engine"]] = entry          # the file on disk is the last build of that name
    n = 0
    for engine, info in newest.items():
        src = os.path.join(model_dir, engine)
        if os.path.isfile(src):
            archive_engine(model_dir, src, info)
            n += 1
    if not n:
        raise SystemExit("%s: engines.json names no engine file in this directory" % model_dir)


def stage_machine(model_dir, host, out):
    """One directory for the packager: shared ONNX and model_meta, plus this host's engines."""
    src = _host_dir(model_dir, host)
    if not os.path.isdir(src):
        raise SystemExit("no engines for %s (%s)" % (host, src))
    os.makedirs(out, exist_ok=True)
    copied = []
    for name in os.listdir(model_dir):
        if name == "model_meta.json" or name.endswith(".onnx") or name.endswith(".onnx.data"):
            shutil.copy2(os.path.join(model_dir, name), os.path.join(out, name))
            copied.append(name)
    for name in os.listdir(src):
        shutil.copy2(os.path.join(src, name), os.path.join(out, name))
        copied.append(name)
    if not any(name.endswith(".engine") for name in copied):
        raise SystemExit("%s has no .engine files" % src)
    print("staged %s -> %s (%s)" % (host, out, ", ".join(sorted(copied))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="export_pillar_onnx.py output")
    ap.add_argument("--precision", default="fp16", help="comma list of fp32, fp16, int8 (or int8_<suffix>)")
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--dla", type=int, default=None, help="DLA core (Orin); the engine name gets .dla<k>")
    ap.add_argument("--timing-cache", default=None, help="file to reuse tactic timings between builds")
    ap.add_argument("--benchmark", type=int, default=0, help="time every new engine N times (random inputs)")
    ap.add_argument("--warmup-ms", type=float, default=500.0, help="warm-up before --benchmark timing")
    ap.add_argument("--opt-pillars", type=int, default=None,
                    help="dynamic PFN: pillar count the kernels are tuned for (default max_pillars / 2)")
    ap.add_argument("--archive", action="store_true",
                    help="copy the engines already in --model-dir into machines/<host>/ and exit")
    ap.add_argument("--stage", default=None, metavar="HOST",
                    help="write ONNX, model_meta and machines/HOST engines to --stage-out, then exit")
    ap.add_argument("--stage-out", default=None, help="directory written by --stage")
    args = ap.parse_args()
    if args.archive:
        archive_current(args.model_dir)
        return
    if args.stage:
        if not args.stage_out:
            raise SystemExit("--stage needs --stage-out")
        stage_machine(args.model_dir, args.stage, args.stage_out)
        return

    from dog_perception.detection.model_meta import ModelMeta
    meta = ModelMeta.load(os.path.join(args.model_dir, "model_meta.json"))

    log_path = os.path.join(args.model_dir, "engines.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    for prec in args.precision.split(","):
        for graph in args.graphs.split(","):
            onnx_path = onnx_for(args.model_dir, graph, prec)
            if not os.path.exists(onnx_path):
                raise SystemExit("%s missing%s" % (onnx_path, " (tools/deploy/quantize_int8.py)" if is_int8(prec) else ""))
            name = "%s.%s%s.engine" % (graph, prec, "" if args.dla is None else ".dla%d" % args.dla)
            engine_path = os.path.join(args.model_dir, name)
            print("building %s from %s ..." % (name, os.path.basename(onnx_path)), flush=True)
            dyn = graph == "pfn" and meta.onnx["pfn"].get("dynamic")
            profile = pillar_profile(meta, args.opt_pillars) if dyn else None
            info = build(onnx_path, engine_path, prec, args.workspace_mb, args.dla, args.timing_cache, profile)
            info.update(graph=graph, precision=prec, engine=name, onnx=os.path.basename(onnx_path), dla=args.dla,
                        built=time.strftime("%Y-%m-%d %H:%M:%S"), **machine())
            if args.benchmark:
                info.update(benchmark(engine_path, args.benchmark, args.warmup_ms))
            log.append(info)
            print("  %s" % json.dumps(info), flush=True)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=1)
            archive_engine(args.model_dir, engine_path, info)


if __name__ == "__main__":
    main()
