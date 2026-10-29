"""Compare detector backends on the same saved frames: PyTorch, ONNX Runtime, TensorRT FP32 / FP16 / INT8.

The first backend is the reference. Per frame, every other backend is compared with it

* stage by stage (exported backends only: ONNX / TensorRT keep their intermediate tensors): PFN
  output, BEV canvas, every head map; max |a - b| / max |ref| and cosine similarity, also per head
  kind (hm, reg, height, dim, rot, vel; the worst task). Not gated: over the whole map the small
  regression outputs of empty cells dominate (FP16 reaches 0.23 there with boxes within 5 mm); the
  boxes are what the gates look at;
* on the detections, before the geometric filters (``raw``) and after them (``final``): boxes
  matched by center (same class, within ``--match-dist``), score / center / yaw / size differences,
  and at the reporting thresholds (DEFAULT_SCORE_THRESHOLDS) the reference boxes the backend misses
  and the boxes it adds. Each miss / extra is one of
    near   the same box on the other side of the threshold, scores within ``--flip-tol``;
    filter (final only) the other side has the box before the filters (same class, within the
           match distance, reported or within ``--flip-tol``) and its filters dropped it: a filter
           measurement (bottom above the ground, points in the box) moved across its threshold;
    score  the same box, scores further apart;
    class  only a box of another class within the match distance;
    none   no box within the match distance.
  ``reported_match`` counts them all; ``reported_match_tol`` leaves out ``near`` and ``filter``:
  threshold flips of a box that moved by less than the gated score / center tolerances, not lost
  or new objects (T27: FP32 ONNX against PyTorch flips filters too). The gates are on
  ``reported_match_tol`` of both raw and final.

Stage differences need intermediate tensors: with a ``torch`` or ``dump:`` reference they are taken
against the first exported backend (``stage_reference`` in the report, normally ``onnx``).

    python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp \\
        --frames "data/runs/0921_164127_s000/frames/*.npy" --stride 3 \\
        --backends torch,onnx,trt:fp32,trt:fp16 --out data/reports/T26/compare_s000.json

Backends: ``torch`` (det3d + the checkpoint in model_meta.json), ``onnx``, ``trt:<precision>``
(``<graph>.<precision>.engine``; for ``int8*`` the PFN falls back to ``pfn.fp16.engine`` when there
is no INT8 PFN), ``dump:<dir>`` (detections another machine saved with ``--dump``). The gates of
docs/DEPLOY_ORIN.md (P1) are checked against the reference and printed.

Across machines (the Orin has neither det3d nor the full data): on the server
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

GATES = dict(reported_match_tol=0.99, score_p99=0.02, center_p99=0.1)
KINDS = ("near", "filter", "score", "class", "none")


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


def _reported(d):
    return d.scores >= score_threshold_array(d.labels, DEFAULT_SCORE_THRESHOLDS)


def _kind(a, i, b, j, dist, tol, b_raw=None):
    """Why box ``i`` of ``a`` has no reported partner in ``b``; ``j``: its same-class pair or None;
    ``b_raw``: ``b`` before the filters (final detections only)."""
    if j is not None:
        return "near" if abs(a.scores[i] - b.scores[j]) <= tol else "score"
    if b_raw is not None and len(b_raw):
        d = np.hypot(*(b_raw.boxes[:, :2] - a.boxes[i, :2]).T)
        d[b_raw.labels != a.labels[i]] = np.inf
        k = int(np.argmin(d))
        if d[k] <= dist and (_reported(b_raw)[k] or abs(a.scores[i] - b_raw.scores[k]) <= tol):
            return "filter"
    if len(b) and (np.hypot(*(b.boxes[:, :2] - a.boxes[i, :2]).T) <= dist).any():
        return "class"
    return "none"


def det_stats(ref, got, dist, acc, tol=0.02, ref_raw=None, got_raw=None):
    """``ref_raw`` / ``got_raw``: the detections before the filters, when ``ref`` / ``got`` are after them."""
    pairs = match(ref, got, dist)
    acc["n_ref"] += len(ref)
    acc["n_got"] += len(got)
    acc["matched"] += len(pairs)
    for i, j in pairs:
        acc["score"].append(abs(ref.scores[i] - got.scores[j]))
        acc["center"].append(float(np.hypot(*(ref.boxes[i, :2] - got.boxes[j, :2]))))
        acc["yaw"].append(abs(float(wrap_angle(ref.boxes[i, 6] - got.boxes[j, 6]))))
        acc["size"].append(float(np.abs(ref.boxes[i, 3:6] - got.boxes[j, 3:6]).max()))
    rep_r, rep_g = _reported(ref), _reported(got)
    of_ref, of_got = dict(pairs), {j: i for i, j in pairs}
    acc["reported_ref"] += int(rep_r.sum())
    for i in np.nonzero(rep_r)[0]:
        j = of_ref.get(int(i))
        if j is None or not rep_g[j]:
            acc["reported_missed"] += 1
            acc["missed_" + _kind(ref, i, got, j, dist, tol, got_raw)] += 1
    for j in np.nonzero(rep_g)[0]:
        i = of_got.get(int(j))
        if i is None or not rep_r[i]:
            acc["reported_extra"] += 1
            acc["extra_" + _kind(got, j, ref, i, dist, tol, ref_raw)] += 1


def summarize_dets(acc):
    q = lambda k, p: float(np.percentile(acc[k], p)) if acc[k] else 0.0
    n = acc["reported_ref"]
    wrong = acc["reported_missed"] + acc["reported_extra"]
    flips = sum(acc[w + "_" + k] for w in ("missed", "extra") for k in ("near", "filter"))
    return dict(n_ref=acc["n_ref"], n_got=acc["n_got"], matched=acc["matched"],
                score_max=q("score", 100), score_p99=q("score", 99), center_max=q("center", 100),
                center_p99=q("center", 99), yaw_p99=q("yaw", 99), size_p99=q("size", 99),
                reported_ref=n, reported_missed=acc["reported_missed"], reported_extra=acc["reported_extra"],
                missed_by={k: acc["missed_" + k] for k in KINDS}, extra_by={k: acc["extra_" + k] for k in KINDS},
                reported_match=1.0 - wrong / max(n, 1), reported_match_tol=1.0 - (wrong - flips) / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="det-frame clouds (.npy, N x 5); globs allowed")
    ap.add_argument("--backends", default="onnx,trt:fp16", help="comma list; the first is the reference")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match-dist", type=float, default=0.5)
    ap.add_argument("--flip-tol", type=float, default=GATES["score_p99"],
                    help="a reported box missing only because its score crossed the threshold by less is 'near'")
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
    # intermediate tensors: the first backend that keeps them (the reference unless it is torch / dump:)
    stage_ref = next((s for s in specs if getattr(dets[s], "keep_stages", False)), None)
    stage = {s: defaultdict(list) for s in specs}
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
            det_stats(out[ref][0], out[s][0], args.match_dist, accs[s]["raw"], args.flip_tol)
            det_stats(out[ref][1], out[s][1], args.match_dist, accs[s]["final"], args.flip_tol, out[ref][0], out[s][0])
        for s in specs:
            a, b = out[s][2], out[stage_ref][2] if stage_ref else None
            if s != stage_ref and a is not None and b is not None:
                for key in ("pfn_out", "bev"):
                    rel, cos = tensor_diff(a[key], b[key])
                    stage[s][key + "_rel"].append(rel)
                    stage[s][key + "_cos"].append(cos)
                rel = [tensor_diff(x, y) for x, y in zip(a["heads"], b["heads"])]
                stage[s]["heads_rel"].append(max(r for r, _ in rel))
                stage[s]["heads_cos"].append(min(c for _, c in rel))
                worst = {}
                for name, (r, c) in zip(dets[stage_ref].meta.rpn_outputs, rel):   # task<t>_<kind>
                    k = name.split("_", 1)[1]
                    rk, ck = worst.get(k, (0.0, 1.0))
                    worst[k] = (max(rk, r), min(ck, c))
                for k, (r, c) in worst.items():
                    stage[s]["head_%s_rel" % k].append(r)
                    stage[s]["head_%s_cos" % k].append(c)
        if (n + 1) % 20 == 0:
            print("%d / %d frames" % (n + 1, len(files)), flush=True)

    if skipped:
        print("%d frames have no saved detections and were skipped" % skipped)
    report = dict(model_dir=args.model_dir, frames=len(files) - skipped, skipped=skipped, reference=ref,
                  stage_reference=stage_ref, flip_tol=args.flip_tol, backends={})
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
            checks = dict(raw_match_tol=r["raw"]["reported_match_tol"] >= GATES["reported_match_tol"],
                          reported_match_tol=fin["reported_match_tol"] >= GATES["reported_match_tol"],
                          score_p99=fin["score_p99"] <= GATES["score_p99"],
                          center_p99=fin["center_p99"] <= GATES["center_p99"])
            r["gates"] = dict(checks, passed=all(checks.values()))
        report["backends"][s] = r

    print("\nreference: %s, %d frames; stages against %s; near = score within %.3f across the threshold"
          % (ref, report["frames"], stage_ref or "-", args.flip_tol))
    print("| backend | reported match | w/o flips raw | w/o flips | missed (%s) | extra (same) "
          "| score p99 | center p99 m | yaw p99 rad | worst head cos | network ms | gates |" % "/".join(KINDS))
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = lambda d: "/".join(str(d[k]) for k in KINDS)  # noqa: E731
    for s in specs[1:]:
        r = report["backends"][s]
        f, st = r["final"], r["stages"]
        cos = {k[5:-4]: v for k, v in st.items() if k.startswith("head_") and k.endswith("_cos")}
        head = "%.4f (%s)" % (cos[min(cos, key=cos.get)], min(cos, key=cos.get)) if cos else "-"
        print("| %s | %.4f | %.4f | %.4f | %d (%s) | %d (%s) | %.2e | %.2e | %.3f | %s | %.1f | %s |" % (
            s, f["reported_match"], r["raw"]["reported_match_tol"], f["reported_match_tol"], f["reported_missed"],
            by(f["missed_by"]), f["reported_extra"], by(f["extra_by"]), f["score_p99"], f["center_p99"], f["yaw_p99"],
            head, r["timing_ms"].get("network_ms", float("nan")), "pass" if r["gates"]["passed"] else "FAIL"))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
