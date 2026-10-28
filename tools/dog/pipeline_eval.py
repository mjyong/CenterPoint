"""Score post-processing, tracking, motion state and prediction on the dog's validation scenes, and tune
them (docs/PIPELINE_TUNING.md; metrics in dog_perception/pipeline_eval.py).

The back half of the pipeline is re-run on CPU from saved detections, so one set of parameters takes
about a minute for the 10 validation scenes and needs no GPU:

    python tools/dog/pipeline_eval.py --scenes $VAL \\
        --results "data/runs_veh_v2/{scene}/pillar/results.pkl" --labels "data/labels/{scene}/labels_sustech2.pkl" \\
        --dense-labels "data/labels/{scene}/labels_full2.pkl" --frames "data/runs/{scene}/frames" \\
        --out data/reports/T18/v2_pp

writes ``report.md`` / ``report.json`` (every metric with a 90 % interval from resampling the scenes),
``issues.csv`` (the worst ID switches, lost vehicles, false tracks, motion errors, predictions and
missed hazards, with the scene and the second to look at), a bird's-eye PNG of each scene's middle
keyframe and one Foxglove MCAP per scene (topics in dog_perception/clip_viz.py; open with the layout
``foxglove_layout.json`` copied next to them).

Detections (``--detections``): ``results`` as saved (filtered with the defaults); ``raw`` the ones
saved before the filters (``run_sequence.py --save-raw``), filtered here, so ``filter.*`` can be swept;
``labels`` the 10 Hz labels as a perfect detector (``--dense-labels``): what tracking and prediction
reach when detection is not the problem.

Parameters (``--set KEY=V1,V2``, repeatable; every combination is one row, the first one the reference):

    --set tracker.high_score=0.25,0.3,0.35        report / track-start threshold (vehicle)
    --set tracker.vehicle.gate=2,3,4               ClassParams of the vehicle class
    --set filter.min_points.vehicle=3,5,8          DetectionFilterConfig (with --detections raw)
    --set predict.merge_dist=0.25,0.5              IMMPredictor
    --set eval.hazard_radius=5,8                   EvalConfig
    --set eval.learned_model=,work_dirs/pred/gru.pt  IMM (empty) against a tier-2 model, paired

``--configs rows.json`` compares given rows instead (a list of {key: value}; e.g. the old defaults
against the new ones, ``[{"tracker.high_score": 0.35}, {}]``).

With more than one row it writes ``sweep.md`` / ``sweep.csv`` instead of a report: per row the
main metrics, and the difference to the first row with the share of resamples in which the row is
better (paired: the same scenes drawn for every row). No MCAP in a sweep.
"""
import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception import pipeline_eval as pe  # noqa: E402

LAYOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foxglove_pipeline_eval.json")
KEEP = ("stamp", "T_world_det", "T_world_body", "pose_bad", "detections", "raw_detections", "raw_measured")
METRIC_NAMES = {
    "det_recall_near": "detections: recall within near range", "det_precision_near": "detections: precision within near range",
    "det_f1_near": "detections: F1 within near range",
    "det_fp_per_frame": "detections: false per keyframe", "MOTA": "tracks: MOTA", "IDF1": "tracks: IDF1",
    "IDSW_per_min": "tracks: ID switches / min", "false_tracks_per_min": "tracks: false tracks / min",
    "confirm_p90": "tracks: confirm delay p90 [s]", "motion_relevant_missed": "motion: relevant moving stretches missed",
    "motion_false_episodes_per_h": "motion: parked reported moving / h", "pred_FDE3_moving": "prediction: FDE 3 s, moving [m]",
    "pred_MR3_moving": "prediction: miss rate 3 s, moving",
}
SHORT = {"det_recall_near": "det recall", "det_precision_near": "det precision", "det_f1_near": "det F1",
         "det_fp_per_frame": "det FP / frame",
         "MOTA": "MOTA", "IDF1": "IDF1", "IDSW_per_min": "IDSW / min", "false_tracks_per_min": "false tracks / min",
         "confirm_p90": "confirm p90 s", "motion_relevant_missed": "moving missed",
         "motion_false_episodes_per_h": "false moving / h", "pred_FDE3_moving": "FDE 3 s", "pred_MR3_moving": "MR 3 s"}

_W = {}


def _init(args):
    _W["args"] = args
    _W["scenes"] = {}


def _scene(name):
    if name not in _W["scenes"]:
        a = _W["args"]
        sub = lambda p: p.replace("{scene}", name)   # noqa: E731
        with open(sub(a.results), "rb") as f:
            rows = [{k: r[k] for k in KEEP if k in r} for r in pickle.load(f)]
        with open(sub(a.labels), "rb") as f:
            labels = pickle.load(f)
        dense = None
        if a.dense_labels and os.path.exists(sub(a.dense_labels)):
            with open(sub(a.dense_labels), "rb") as f:
                dense = pickle.load(f)
        _W["scenes"] = {name: pe.load_scene(name, rows, labels, dense)}      # keep one scene per worker
    return _W["scenes"][name]


def _cloud_of(name):
    a = _W["args"]
    if not a.frames:
        return None
    d = a.frames.replace("{scene}", name)

    def load(stamp):
        p = os.path.join(d, "%.6f.npy" % stamp)
        return np.load(p) if os.path.exists(p) else None
    return load


def _work(job):
    ci, overrides, name, draw = job
    a = _W["args"]
    cfg = pe.apply_overrides(pe.EvalConfig(), overrides)
    sd = _scene(name)
    viz = writer = None
    if draw:
        from dog_perception import clip_viz
        from dog_perception.foxglove import McapWriter
        writer = McapWriter(os.path.join(a.mcap_dir or a.out, "%s.mcap" % name))
        viz = clip_viz.ClipVisualizer(writer, frame_period=0.1, keyframe_period=0.5)
    logger = None
    if a.track_logs:
        from dog_perception.prediction import TrackLogger
        logger = TrackLogger()
    res = pe.run_scene(sd, cfg, a.detections, viz, _cloud_of(name), a.lidar_stride, logger)
    if logger is not None:
        logger.save(os.path.join(a.track_logs, "%s.npz" % name))
    if writer is not None:
        from dog_perception import foxglove as fx
        for i in sorted(res.issues, key=lambda i: i["stamp"]):      # the issues in the Log panel, at their time
            writer.write("/log", "foxglove.Log", i["stamp"], fx.log(i["stamp"], "%s: track %s, label %s, %.0f m %s %s" % (
                i["kind"], i["track"], i["gt"], i["range"], i["zone"], i["note"]), level=3, name="issues"))
        cams = os.path.join(a.seq_root, name, "cameras.mcap")
        if os.path.exists(cams) and sd.rows:
            writer.copy_from(cams, sd.rows[0]["stamp"] - 1.5, sd.rows[-1]["stamp"])
        writer.close()
        res.mcap = (writer.path, writer.messages, os.path.getsize(writer.path))
    return ci, name, res


# ------------------------------------------------------------------ formatting
def _f(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return ("%." + str(nd) + "f") % v if isinstance(v, (float, np.floating)) else str(v)


def _ival(boot, key, nd=3):
    v = boot.get(key)
    if v is None or not np.isfinite(v).any():
        return "-"
    return "%s – %s" % (_f(float(np.nanpercentile(v, 5)), nd), _f(float(np.nanpercentile(v, 95)), nd))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_issues(path, results, per_kind):
    rows = [i for r in results for i in r.issues]
    by_kind = {}
    for i in rows:
        by_kind.setdefault(i["kind"], []).append(i)
    cols = ["kind", "scene", "t", "duration", "track", "gt", "range", "zone", "value", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for kind in sorted(by_kind):
            for i in sorted(by_kind[kind], key=lambda i: -i["value"])[:per_kind]:
                w.writerow({k: ("%.1f" % v if isinstance(v, (float, np.floating)) else ("" if v is None else v))
                            for k, v in i.items()})
    return {k: len(v) for k, v in sorted(by_kind.items())}


def write_report(args, cfg, overrides, m, boot, results, issue_counts, figures):
    t = m["tables"]
    c, k = t["counts"], t["mot"]
    hs = pe.score_threshold_array([pe.VEHICLE], cfg.tracker.high_score)[0]
    L = ["# Pipeline evaluation", ""]
    L += ["- scenes (%d): %s" % (len(results), ", ".join(r.name for r in results)),
          "- %d frames, %d keyframes scored (%.1f min)" % (t["frames"], t["keyframes"], t["minutes"]),
          "- detections: `%s` from `%s`" % (args.detections, args.results),
          "- parameters: %s" % (", ".join("`%s=%s`" % kv for kv in overrides.items()) or "defaults"),
          "- scored: vehicles that are not ignore boxes, within %.0f m and with current-sweep points; a detection "
          "or track on any other labelled box is not counted. Label tracks stitched into one vehicle: %d."
          % (cfg.max_range, t["merges"]),
          "- intervals: 90 %% over %d resamples of the scenes" % args.bootstrap, ""]
    L += ["## Main metrics", "", "| metric | value | interval |", "|---|---|---|"]
    for key, _ in pe.MAIN_METRICS:
        L.append("| %s | **%s** | %s |" % (METRIC_NAMES.get(key, key), _f(m[key]), _ival(boot, key)))
    L += ["", "## Detections at the reporting threshold", "",
          "Vehicle threshold %.2f (`tracker.high_score`), 2 m match, keyframes. Near: within %.0f m." % (hs, cfg.near_range),
          "", "| range | precision | recall | TP | FP | FN |", "|---|---|---|---|---|---|"]
    for near, name in (("near", "0-%.0f m" % cfg.near_range), ("far", "%.0f-%.0f m" % (cfg.near_range, cfg.max_range))):
        L.append("| %s | %s | %s | %d | %d | %d |" % (name, _f(m["det_precision_" + near]), _f(m["det_recall_" + near]),
                                                     c["det_tp_" + near], c["det_fp_" + near], c["det_fn_" + near]))
    d = t["detection"]
    L += ["", "False detections per keyframe %s. AP over all scores: AP@2.0 %s, AP@4.0 %s, AP@2.0 0-20 m %s, 20-40 m %s."
          % (_f(m["det_fp_per_frame"]), _f(d.get("AP@2.0")), _f(d.get("AP@4.0")), _f(d.get("AP@2.0[0-20m]")),
             _f(d.get("AP@2.0[20-40m]"))), ""]
    L += ["## Tracking", "", "Reported tracks (confirmed, coasting included) against the scored vehicles, keyframes, 2 m.",
          "", "| MOTA | MOTP m | IDF1 | recall | precision | IDSW | IDSW / min | FRAG | MT | ML | vehicles | TP | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
          "| **%s** | %s | **%s** | %s | %s | %d | %s | %d | %d | %d | %d | %d | %d | %d |" % (
              _f(m["MOTA"]), _f(m["MOTP"], 2), _f(m["IDF1"]), _f(m["trk_recall"]), _f(m["trk_precision"]), k["IDSW"],
              _f(m["IDSW_per_min"], 2), k["FRAG"], k["MT"], k["ML"], k["num_gt_tracks"], k["TP"], k["FP"], k["FN"]), "",
          "- false tracks (a track false on consecutive keyframes counts once): %s per minute"
          % _f(m["false_tracks_per_min"], 2),
          "- a vehicle coming into %.0f m waits for its first matched track: median %s s, p90 %s s (%d vehicles; %d "
          "never tracked; 0.5 s resolution)" % (cfg.near_range, _f(m["confirm_p50"], 1), _f(m["confirm_p90"], 1),
                                               t["latency_n"], m["never_tracked"]), ""]
    L += ["### Where the errors are", "", "| false tracks: cause | count |", "|---|---|"]
    L += ["| %s | %d |" % (cause, c["fp_cause|" + cause]) for cause in pe.FP_CAUSES]
    L += ["", "| missed vehicles (FN) | count |", "|---|---|"]
    L += ["| %s | %d |" % (x, c["fn_cause|" + x]) for x in pe.FN_CAUSES]
    L += ["| %s | %d |" % (x, c["fn_range|" + x]) for _, x in pe.RANGE_BINS]
    L += ["| %s | %d |" % (x, c["fn_points|" + x]) for _, x in pe.POINT_BINS]
    L += ["", "Blind sector (%d-%d deg behind the robot) and the %d deg bands beside it:" % (
        cfg.sector[0], cfg.sector[1], cfg.edge), "",
        "| zone | scored vehicles | missed | missed share | tracks | false | false share |", "|---|---|---|---|---|---|---|"]
    for z in ("elsewhere", "edge", "blind"):
        g, fn, tr, fp = c["gt_zone|" + z], c["fn_zone|" + z], c["trk_zone|" + z], c["fp_zone|" + z]
        L.append("| %s | %d | %d | %s | %d | %d | %s |" % (z, g, fn, _f(fn / g if g else float("nan")), tr, fp,
                                                         _f(fp / tr if tr else float("nan"))))
    ms = t["motion"]
    L += ["", "## Motion state", "", "Online state against the labels' moving flags (tools/dog/motion_eval.py). Relevant: "
          ">= 1 m/s and within %.0f m." % cfg.near_range, "",
          "| moving recall | false moving | parked reported moving | false episodes (per h) | moving stretches missed "
          "| latency median / p90 s | relevant missed | relevant latency median / p90 s |", "|---|---|---|---|---|---|---|---|",
          "| %s | %s | %d / %d | %d (%s) | %d / %d | %s / %s | **%d / %d** | %s / %s |" % (
              _f(ms["moving_recall"]), _f(ms["false_moving"], 4), ms["static_flagged"], ms["static_tracks"],
              ms["false_episodes"], _f(ms["false_episodes_per_h"], 1), ms["missed"], ms["segments"],
              _f(ms["latency_median"], 2), _f(ms["latency_p90"], 2), ms["relevant_missed"], ms["relevant_segments"],
              _f(ms["relevant_latency_median"], 2), _f(ms["relevant_latency_p90"], 2)), ""]
    L += ["## Prediction", "", "Tracks matched to a vehicle with 3 s of labelled future (keyframes every 0.5 s). Moving: "
          ">= %.1f m in 3 s; turning: heading changes > %.0f deg. ADE1 / FDE1: the most likely mode; minADE / minFDE: "
          "the best; MR: no mode ends within 2 m. %s is what runs online (IMM: tier 1, learned: tier 2), CV keeps the "
          "track's velocity." % (cfg.moving_disp, cfg.turn_deg, results[0].predictor if results else "IMM"), ""]
    for subset, res in t["prediction"].items():
        n = next(iter(next(iter(res.values()), {}).values()), {}).get("num", 0)
        L += ["**%s** (%d samples)" % (subset, n), "", "| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |",
              "|---|---|---|---|---|---|---|"]
        for name, by_h in res.items():
            for h, r in by_h.items():
                L.append("| %s | %.0f s | %s | %s | %s | %s | %s |" % (name, h, _f(r["ADE1"], 2), _f(r["FDE1"], 2),
                                                                       _f(r["minADE"], 2), _f(r["minFDE"], 2), _f(r["MR"])))
        L.append("")
    hz = t["hazard"]
    lead = [h["lead"] for h in hz if h["reported"]]
    L += ["## Hazard proxy", "", "A moving vehicle that comes within %.0f m of the robot within %.0f s (labelled future, "
          "the robot's own path) should raise an alarm: a moving track whose most likely predicted path, or its position "
          "now, comes that close. Few such vehicles in 0921; the numbers are for the controlled scenes to come "
          "(DATA_COLLECTION_PLAN.md 3.2)." % (cfg.hazard_radius, cfg.hazard_horizon), "",
          "- events %d, reported %d, lead before the closest approach: median %s s, min %s s" % (
              len(hz), sum(h["reported"] for h in hz), _f(float(np.median(lead)) if lead else float("nan"), 1),
              _f(float(np.min(lead)) if lead else float("nan"), 1)),
          "- alarms without such a vehicle: %d (%s per hour)" % (sum(r.false_alarms for r in results),
                                                                  _f(m["false_alarms_per_h"], 1)), ""]
    L += ["## Timing (this machine, ms per frame: median / p90)", "", "| stage | median | p90 |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (key[:-3], _f(v[0], 2), _f(v[1], 2)) for key, v in t["timing"].items()]
    L += ["", "## Issues (`issues.csv`)", "", "| kind | count |", "|---|---|"]
    L += ["| %s | %d |" % kv for kv in issue_counts.items()]
    if figures:
        L += ["", "## Snapshots", "", "Middle keyframe of each scene, robot at the center heading up. Green / red outlines: "
              "scored vehicles tracked / missed; grey: not scored; filled: reported tracks; lines: the two most likely "
              "predicted modes; dashes: labelled 3 s future; shaded: the blind sector.", ""]
        L += ["![%s](%s)" % (f, f) for f in figures]
    mcaps = [r.mcap for r in results if getattr(r, "mcap", None)]
    if mcaps:
        L += ["", "## Foxglove", ""] + ["- `%s`: %d messages, %.0f MB" % (p, n, s / 1e6)
                                        for p, n, s in mcaps]
        L += ["", "Open a file in Foxglove, then Layout -> Import from file: `foxglove_layout.json` (3D following `ego`, "
              "Plot of `/metrics/frame`, Log of `/log`). Jump to the times in `issues.csv` (column t: seconds after the "
              "scene start)."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(text)
    flat = {key: v for key, v in m.items() if key != "tables"}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(_clean(dict(overrides=overrides, metrics=flat, intervals={key: [float(np.nanpercentile(v, 5)),
                                                                                 float(np.nanpercentile(v, 95))]
                                                                           for key, v in boot.items()
                                                                           if np.isfinite(v).any()},
                              counts=dict(c), mot=k, motion={kk: vv for kk, vv in ms.items() if kk != "details"},
                              prediction=t["prediction"], detection=d, hazard=hz, timing=t["timing"])), f, indent=1)
    return text


def write_sweep(args, grid, summaries, boot):
    names = list(summaries)
    keys = [k for k, _ in pe.MAIN_METRICS]
    L = ["# Parameter sweep", "", "- scenes: %s" % args.scenes_text, "- detections: `%s` from `%s`" % (
        args.detections, args.results), "- every row against row 1; ahead: share of %d paired resamples of the scenes in "
         "which the row is better" % args.bootstrap, "",
         "Columns: %s." % "; ".join("%s = %s" % (SHORT[k], METRIC_NAMES[k]) for k in keys), "",
         "| # | parameters | " + " | ".join(SHORT.get(k, k) for k in keys) + " |",
         "|---|---|" + "---|" * len(keys)]
    for i, name in enumerate(names):
        L.append("| %d | %s | %s |" % (i + 1, name, " | ".join(_f(summaries[name][k]) for k in keys)))
    L += ["", "Difference to row 1 (ahead %):", "", "| # | " + " | ".join(SHORT.get(k, k) for k in keys) + " |",
          "|---|" + "---|" * len(keys)]
    ref = names[0]
    for i, name in enumerate(names[1:], 2):
        cells = []
        for key, sign in pe.MAIN_METRICS:
            d = summaries[name][key] - summaries[ref][key]
            a, b = boot[name].get(key), boot[ref].get(key)
            ok = np.isfinite(a) & np.isfinite(b) if a is not None and b is not None else None
            ahead = float(np.mean(sign * (a[ok] - b[ok]) > 0)) if ok is not None and ok.any() else float("nan")
            cells.append("%s (%s)" % (_f(d), "-" if not np.isfinite(ahead) else "%.0f%%" % (100 * ahead)))
        L.append("| %d | %s |" % (i, " | ".join(cells)))
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "sweep.md"), "w") as f:
        f.write(text)
    flat_keys = [k for k in summaries[ref] if k != "tables"]
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        params = sorted({p for o in grid for p in o})
        w.writerow(["row"] + params + flat_keys + ["%s_p5" % k for k in keys] + ["%s_p95" % k for k in keys])
        for i, (name, o) in enumerate(zip(names, grid), 1):
            s = summaries[name]
            iv = [np.nanpercentile(boot[name][k], q) if np.isfinite(boot[name][k]).any() else "" for q in (5, 95)
                  for k in keys]
            w.writerow([i] + [o.get(p, "") for p in params] + [s[k] for k in flat_keys] + iv)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--results", required=True, help="results.pkl per scene, {scene} stands for the scene name")
    ap.add_argument("--labels", required=True, help="reviewed keyframe labels per scene (labels_sustech2.pkl)")
    ap.add_argument("--dense-labels", default=None,
                    help="10 Hz labels per scene (labels_train2.pkl / labels_full2.pkl): drawn between keyframes, "
                         "and the detections of --detections labels")
    ap.add_argument("--frames", default=None, help="saved detector inputs per scene (lidar in the MCAP and figures)")
    ap.add_argument("--seq-root", default="data/seq", help="<seq-root>/<scene>/cameras.mcap is copied into the MCAP")
    ap.add_argument("--detections", choices=("results", "raw", "labels"), default="results")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2", help="parameter values to sweep")
    ap.add_argument("--configs", default=None,
                    help="JSON list of {key: value} rows to compare instead of every combination; --set values apply "
                         "to every row unless the row sets the key itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mcap", action="store_true")
    ap.add_argument("--mcap-dir", default=None, help="where the MCAP files go (default --out); they are large")
    ap.add_argument("--track-logs", default=None, metavar="DIR",
                    help="also write each scene's tracks, robot poses and predictions to DIR/<scene>.npz: "
                         "the training data of tools/dog/train_predictor.py (one parameter set only)")
    ap.add_argument("--lidar-stride", type=int, default=2, help="keep every n-th point of /lidar")
    ap.add_argument("--bootstrap", type=int, default=500, help="resamples of the scenes for the intervals")
    ap.add_argument("--issues", type=int, default=30, help="issues listed per kind")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    args.scenes_text = ", ".join(args.scenes)
    os.makedirs(args.out, exist_ok=True)
    if args.mcap_dir:
        os.makedirs(args.mcap_dir, exist_ok=True)
    if args.track_logs:
        os.makedirs(args.track_logs, exist_ok=True)

    grid = pe.parse_grid(args.set)
    if args.configs:
        with open(args.configs) as f:
            rows = json.load(f)
        if len(grid) > 1:
            raise SystemExit("--configs takes single-valued --set only")
        grid = [dict(grid[0], **r) for r in rows]
    cfgs = [pe.apply_overrides(pe.EvalConfig(), o) for o in grid]       # fails early on a wrong key
    if args.detections == "labels" and not args.dense_labels:
        raise SystemExit("--detections labels needs --dense-labels")
    if args.track_logs and len(grid) > 1:
        raise SystemExit("--track-logs writes one set of logs: give one parameter set")
    if any(pe.needs_clouds(c) for c in cfgs) and not args.frames:
        raise SystemExit("box_margin / ground settings change what the filters measure: pass --frames")
    draw = len(grid) == 1 and not args.no_mcap
    jobs = [(ci, o, name, draw) for name in args.scenes for ci, o in enumerate(grid)]
    print("%d scenes x %d parameter sets, detections: %s" % (len(args.scenes), len(grid), args.detections), flush=True)
    if args.jobs > 1:
        with Pool(min(args.jobs, len(jobs)), initializer=_init, initargs=(args,)) as pool:
            out = pool.map(_work, jobs, chunksize=1)
    else:
        _init(args)
        out = [_work(j) for j in jobs]
    by_cfg = {ci: {} for ci in range(len(grid))}
    for ci, name, res in out:
        by_cfg[ci][name] = res
    label = [", ".join("%s=%s" % kv for kv in o.items()) or "defaults" for o in grid]
    results = {label[ci]: [by_cfg[ci][n] for n in args.scenes] for ci in range(len(grid))}
    boot = pe.bootstrap(results, args.bootstrap, step=cfgs[0].predictor["step"]) if args.bootstrap else \
        {n: {} for n in results}

    if len(grid) > 1:
        summaries = {n: pe.summarize(rs, cfgs[0].predictor["step"]) for n, rs in results.items()}
        print(write_sweep(args, grid, summaries, boot))
        return
    rs = results[label[0]]
    m = pe.summarize(rs, cfgs[0].predictor["step"])
    counts = write_issues(os.path.join(args.out, "issues.csv"), rs, args.issues)
    figures = []
    from dog_perception.clip_viz import bev_figure
    for r in rs:
        if r.snapshot is not None:
            path = os.path.join(args.out, "%s_bev.png" % r.name)
            bev_figure(r.snapshot, path, extent=cfgs[0].max_range, forward="y",
                       ego_outline=[[-0.25, -0.45], [0.25, -0.45], [0.25, 0.45], [-0.25, 0.45]],
                       sector=cfgs[0].sector)
            figures.append(os.path.basename(path))
    if draw and os.path.exists(LAYOUT):
        shutil.copy(LAYOUT, os.path.join(args.mcap_dir or args.out, "foxglove_layout.json"))
    print(write_report(args, cfgs[0], grid[0], m, boot[label[0]], rs, counts, figures))


if __name__ == "__main__":
    main()
