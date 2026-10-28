"""Compare detector backends on the same saved frames: PyTorch, ONNX Runtime, TensorRT FP32 / FP16 / INT8.

The first backend is the reference. Per frame, every other backend is compared with it

* stage by stage (exported backends only: ONNX / TensorRT keep their intermediate tensors): PFN
  output, BEV canvas, every head map; max |a - b| / max |ref| and cosine similarity;
* on the detections, before the geometric filters (``raw``) and after them (``final``): boxes
  matched by center (same class, within ``--match-dist``), score / center / yaw / size differences,
  and at the reporting thresholds (DEFAULT_SCORE_THRESHOLDS) the reference boxes the backend misses
  and the boxes it adds.

    python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp \\
        --frames "data/runs/0921_164127_s000/frames/*.npy" --stride 3 \\
        --backends torch,onnx,trt:fp32,trt:fp16 --out data/reports/T26/compare_s000.json

Backends: ``torch`` (det3d + the checkpoint in model_meta.json), ``onnx``, ``trt:<precision>``
(``<graph>.<precision>.engine``; for ``int8*`` the PFN falls back to ``pfn.fp16.engine`` when there
is no INT8 PFN), ``dump:<dir>`` (detections another machine saved with ``--dump``). The gates of
docs/DEPLOY_ORIN.md (P1) are checked against the reference and printed.

Across machines (the Orin has neither PyTorch nor the full data): on the server
``--backends torch --dump data/deploy/dumps`` saves the reference detections of the copied frames;
on the Orin ``--backends dump:dumps/torch,trt:fp16`` compares its engines with them.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.filters import DEFAULT_SCORE_THRESHOLDS  # noqa: E402
from dog_perception.detection.boxes import score_threshold_array  # noqa: E402
from dog_perception.geometry import wrap_angle  # noqa: E402

GATES = dict(head_rel=1e-2, reported_match=0.99, score_p99=0.02, center_p99=0.1)


class DumpBackend:
    """Detections saved by ``--dump`` on another machine, looked up by frame file name (``current``)."""

    def __init__(self, path):
        self.path, self.current, self.last_raw, self.last_timing = path, None, None, {}

    def __call__(self, points_det, stamp=None):
        from dog_perception.detection.boxes import Detections
        d = np.load(os.path.join(self.path, self.current + ".npz"))
        get = lambda k: Detections(d[k + "_boxes"], d[k + "_velocities"], d[k + "_scores"], d[k + "_labels"], "det",
                                   stamp, d[k + "_subclasses"])
        self.last_raw = get("raw")
        return get("final")


def save_dump(path, raw, final):
    arrays = {}
    for k, d in (("raw", raw), ("final", final)):
        arrays.update({k + "_boxes": d.boxes, k + "_velocities": d.velocities, k + "_scores": d.scores,
                       k + "_labels": d.labels, k + "_subclasses": d.subclasses})
    np.savez_compressed(path, **arrays)


def make_backend(spec, model_dir, device=None):
    from dog_perception.detection.exported import ExportedPillarDetector, graph_file
    if spec.startswith("dump:"):
        return DumpBackend(spec[5:])
    if spec == "torch":
        from dog_perception.detection.centerpoint import CenterPointDetector
        from dog_perception.detection.model_meta import ModelMeta
        src = ModelMeta.load(os.path.join(model_dir, "model_meta.json")).source
        return CenterPointDetector(src["config"], src["checkpoint"], device=device, xy_range=src.get("xy_range", 40.8))
    if spec == "onnx":
        return ExportedPillarDetector(model_dir, "onnx", keep_stages=True)
    runtime, prec = spec.split(":")
    pfn = None
    if prec.startswith("int8") and not os.path.exists(graph_file(model_dir, "pfn", "trt", prec)):
        pfn = "pfn.fp16.engine"
    return ExportedPillarDetector(model_dir, runtime, prec, pfn=pfn, keep_stages=True)


def tensor_diff(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)) if len(b) else 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0
    return rel, cos


def match(ref, got, dist):
    """Greedy by reference score: (ref index, got index) pairs of the same class within ``dist``."""
    pairs, used = [], np.zeros(len(got), bool)
    for i in np.argsort(-ref.scores, kind="stable"):
        if len(got) == 0:
            break
        d = np.hypot(*(got.boxes[:, :2] - ref.boxes[i, :2]).T)
        d[used | (got.labels != ref.labels[i])] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist:
            pairs.append((int(i), j))
            used[j] = True
    return pairs


def det_stats(ref, got, dist, acc):
    pairs = match(ref, got, dist)
    acc["n_ref"] += len(ref)
    acc["n_got"] += len(got)
    acc["matched"] += len(pairs)
    for i, j in pairs:
        acc["score"].append(abs(ref.scores[i] - got.scores[j]))
        acc["center"].append(float(np.hypot(*(ref.boxes[i, :2] - got.boxes[j, :2]))))
        acc["yaw"].append(abs(float(wrap_angle(ref.boxes[i, 6] - got.boxes[j, 6]))))
        acc["size"].append(float(np.abs(ref.boxes[i, 3:6] - got.boxes[j, 3:6]).max()))
    rep_r = ref.scores >= score_threshold_array(ref.labels, DEFAULT_SCORE_THRESHOLDS)
    rep_g = got.scores >= score_threshold_array(got.labels, DEFAULT_SCORE_THRESHOLDS)
    m_r = {i for i, j in pairs if rep_g[j]}
    m_g = {j for i, j in pairs if rep_r[i]}
    acc["reported_ref"] += int(rep_r.sum())
    acc["reported_missed"] += int(sum(1 for i in np.nonzero(rep_r)[0] if i not in m_r))
    acc["reported_extra"] += int(sum(1 for j in np.nonzero(rep_g)[0] if j not in m_g))


def summarize_dets(acc):
    q = lambda k, p: float(np.percentile(acc[k], p)) if acc[k] else 0.0
    n = acc["reported_ref"]
    return dict(n_ref=acc["n_ref"], n_got=acc["n_got"], matched=acc["matched"],
                score_max=q("score", 100), score_p99=q("score", 99), center_max=q("center", 100),
                center_p99=q("center", 99), yaw_p99=q("yaw", 99), size_p99=q("size", 99),
                reported_ref=n, reported_missed=acc["reported_missed"], reported_extra=acc["reported_extra"],
                reported_match=1.0 - (acc["reported_missed"] + acc["reported_extra"]) / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="det-frame clouds (.npy, N x 5); globs allowed")
    ap.add_argument("--backends", default="onnx,trt:fp16", help="comma list; the first is the reference")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match-dist", type=float, default=0.5)
    ap.add_argument("--device", default=None, help="torch device")
    ap.add_argument("--out", default=None, help="JSON report")
    ap.add_argument("--dump", default=None, help="save every backend's detections to DIR/<backend>/<frame>.npz")
    args = ap.parse_args()

    files = sorted(f for p in args.frames for f in glob.glob(p))[::args.stride]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no frames match %s" % args.frames)
    specs, seen = [], {}
    for spec in args.backends.split(","):              # a backend given twice (a determinism check) gets #2
        seen[spec] = seen.get(spec, 0) + 1
        specs.append(spec if seen[spec] == 1 else "%s#%d" % (spec, seen[spec]))
    dets = {s: make_backend(s.split("#")[0], args.model_dir, args.device) for s in specs}
    ref = specs[0]
    stage = {s: defaultdict(list) for s in specs[1:]}
    accs = {s: {k: defaultdict(int, score=[], center=[], yaw=[], size=[]) for k in ("raw", "final")}
            for s in specs[1:]}
    timing = {s: defaultdict(list) for s in specs}
    pillars, skipped = [], 0
    for n, f in enumerate(files):
        frame = os.path.splitext(os.path.basename(f))[0]
        if any(isinstance(d, DumpBackend) and not os.path.exists(os.path.join(d.path, frame + ".npz"))
               for d in dets.values()):
            skipped += 1                              # not among the frames the other machine saved
            continue
        pts = np.load(f)
        out = {}
        for s in specs:
            if isinstance(dets[s], DumpBackend):
                dets[s].current = frame
            final = dets[s](pts, 0.0)
            if args.dump and not isinstance(dets[s], DumpBackend):
                d = os.path.join(args.dump, s.replace(":", "_").replace("#", "_"))
                os.makedirs(d, exist_ok=True)
                save_dump(os.path.join(d, frame + ".npz"), dets[s].last_raw, final)
            out[s] = (dets[s].last_raw, final, getattr(dets[s], "last_stages", None))
            for k, v in dets[s].last_timing.items():
                timing[s][k].append(v)
        if hasattr(dets[specs[-1]], "last_pillars"):
            pillars.append(dets[specs[-1]].last_pillars)
        for s in specs[1:]:
            det_stats(out[ref][0], out[s][0], args.match_dist, accs[s]["raw"])
            det_stats(out[ref][1], out[s][1], args.match_dist, accs[s]["final"])
            a, b = out[s][2], out[ref][2]
            if a is not None and b is not None:
                for key in ("pfn_out", "bev"):
                    rel, cos = tensor_diff(a[key], b[key])
                    stage[s][key + "_rel"].append(rel)
                    stage[s][key + "_cos"].append(cos)
                rel = [tensor_diff(x, y) for x, y in zip(a["heads"], b["heads"])]
                stage[s]["heads_rel"].append(max(r for r, _ in rel))
                stage[s]["heads_cos"].append(min(c for _, c in rel))
        if (n + 1) % 20 == 0:
            print("%d / %d frames" % (n + 1, len(files)), flush=True)

    if skipped:
        print("%d frames have no saved detections and were skipped" % skipped)
    report = dict(model_dir=args.model_dir, frames=len(files) - skipped, skipped=skipped, reference=ref, backends={})
    if pillars:
        used = np.array(pillars)
        report["pillars"] = dict(max=int(used[:, 1].max()), truncated_frames=int((used[:, 1] > used[:, 0]).sum()))
    for s in specs:
        r = dict(timing_ms={k: float(np.median(v)) for k, v in timing[s].items()})
        if s != ref:
            r["raw"] = summarize_dets(accs[s]["raw"])
            r["final"] = summarize_dets(accs[s]["final"])
            r["stages"] = {k: (float(np.max(v)) if k.endswith("_rel") else float(np.min(v))) for k, v in stage[s].items()}
            fin = r["final"]
            checks = dict(reported_match=fin["reported_match"] >= GATES["reported_match"],
                          score_p99=fin["score_p99"] <= GATES["score_p99"],
                          center_p99=fin["center_p99"] <= GATES["center_p99"])
            if "heads_rel" in r["stages"]:
                checks["head_rel"] = r["stages"]["heads_rel"] <= GATES["head_rel"]
            r["gates"] = dict(checks, passed=all(checks.values()))
        report["backends"][s] = r

    print("\nreference: %s, %d frames" % (ref, report["frames"]))
    print("| backend | reported match | missed / extra | score p99 | center p99 m | heads rel (max) | network ms | gates |")
    print("|---|---|---|---|---|---|---|---|")
    for s in specs[1:]:
        r = report["backends"][s]
        f = r["final"]
        print("| %s | %.4f | %d / %d | %.2e | %.2e | %s | %.1f | %s |" % (
            s, f["reported_match"], f["reported_missed"], f["reported_extra"], f["score_p99"], f["center_p99"],
            "%.2e" % r["stages"]["heads_rel"] if "heads_rel" in r["stages"] else "-",
            r["timing_ms"].get("network_ms", float("nan")), "pass" if r["gates"]["passed"] else "FAIL"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
