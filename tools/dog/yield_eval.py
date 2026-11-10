"""Offline evaluation of the yield gate (dog_perception/yield_gate.py) on recorded scenes with reviewed
labels: imagined crossings from where the robot stands, the gate fed by the tracker and IMM predictor re-run
on saved detections, the labels as the truth (dog_perception/yield_eval.py; docs/PLANNING_INTEGRATION.md 7).

    python tools/dog/yield_eval.py --scenes $(ls -d data/labels/1104_* | xargs -n1 basename) \
        --labels "data/labels/{scene}/labels_sustech2.pkl" --results "data/runs_online/{scene}/pillar/results.pkl" \
        --fan 8 --length 8 --details data/reports/T44/episodes.csv | tee data/reports/T44/yield_1104.md

Paths: ``--fan N`` straight paths every 360/N degrees from the robot (det frame of the scene's first frame,
named by the angle counter-clockwise from forward), or ``--path NAME X0 Y0 X1 Y1`` (det frame, metres).
"""
import argparse
import csv
import os
import pickle
import sys
from dataclasses import fields, replace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.yield_eval import (CAUSES, EvalConfig, det_to_world, evaluate_path, fan_paths,  # noqa: E402
                                       first_seen, gt_tracks, summarize, track_objects, traffic_stats)
from dog_perception.yield_gate import YieldConfig, required_sight_range  # noqa: E402

KMH = 3.6


def _fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%.*f" % (nd, x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs=2, action="append", default=[], metavar=("LABELS", "RESULTS"))
    ap.add_argument("--scenes", nargs="+", default=[], help="scene names, with --labels and --results patterns")
    ap.add_argument("--labels", default="data/labels/{scene}/labels_sustech2.pkl")
    ap.add_argument("--results", default="data/runs_online/{scene}/pillar/results.pkl")
    ap.add_argument("--fan", type=int, default=0)
    ap.add_argument("--length", type=float, default=8.0, help="[m] of the fan paths")
    ap.add_argument("--start", type=float, default=0.0, help="[m] fan paths start this far from the robot")
    ap.add_argument("--path", nargs=5, action="append", default=[], metavar=("NAME", "X0", "Y0", "X1", "Y1"))
    ap.add_argument("--pred-horizon", type=float, default=3.0, help="[s] IMM prediction (the robot's: 3.0)")
    for f in fields(YieldConfig):
        if f.type in (float, "float"):
            ap.add_argument("--" + f.name.replace("_", "-"), type=float, default=None, help="YieldConfig.%s" % f.name)
    for f in fields(EvalConfig):
        if f.type in (float, "float"):
            ap.add_argument("--eval-" + f.name.replace("_", "-"), type=float, default=None,
                            help="EvalConfig.%s" % f.name)
    ap.add_argument("--details", default=None, help="CSV with every episode")
    args = ap.parse_args(argv)

    cfg = replace(YieldConfig(), **{f.name: getattr(args, f.name) for f in fields(YieldConfig)
                                     if getattr(args, f.name, None) is not None})
    ev = replace(EvalConfig(), **{f.name: getattr(args, "eval_" + f.name) for f in fields(EvalConfig)
                                  if getattr(args, "eval_" + f.name, None) is not None})
    scenes = [(os.path.basename(os.path.dirname(os.path.abspath(lp))), lp, rp) for lp, rp in args.scene]
    scenes += [(s, args.labels.format(scene=s), args.results.format(scene=s)) for s in args.scenes]
    paths = fan_paths(args.fan, args.length, args.start) if args.fan else {}
    for name, x0, y0, x1, y1 in args.path:
        paths[name] = np.array([[float(x0), float(y0)], [float(x1), float(y1)]])
    if not scenes or not paths:
        ap.error("need scenes (--scene or --scenes) and paths (--fan or --path)")

    print("# Yield gate, offline\n")
    print("Gate: %s" % ", ".join("%s=%s" % (f.name, getattr(cfg, f.name)) for f in fields(YieldConfig)))
    print("Evaluation: %s, IMM horizon %.1f s\n" % (
        ", ".join("%s=%s" % (f.name, getattr(ev, f.name)) for f in fields(EvalConfig)), args.pred_horizon))

    eps_all, rows, parked, seen_all, speeds_all = [], [], {}, [], []
    for scene, lp, rp in scenes:
        if not (os.path.isfile(lp) and os.path.isfile(rp)):
            print("- %s: SKIPPED (missing %s)" % (scene, lp if not os.path.isfile(lp) else rp))
            continue
        labels = pickle.load(open(lp, "rb"))
        results = pickle.load(open(rp, "rb"))
        good = [r for r in results if not r.get("pose_bad")]
        if not good or not labels["frames"]:
            print("- %s: SKIPPED (no frames)" % scene)
            continue
        T0 = np.asarray(good[0]["T_world_det"])
        moved = max(float(np.linalg.norm(np.asarray(r["T_world_det"])[:2, 3] - T0[:2, 3])) for r in good)
        tracks = gt_tracks(labels, ev.classes)
        frames = track_objects(results, args.pred_horizon)
        gt_end = max(fr["stamp"] for fr in labels["frames"])
        dog_xy = T0[:2, 3]
        grid = np.arange(frames[0][0], gt_end, ev.grid_dt)
        top = traffic_stats(tracks, dog_xy, grid, ev) * KMH
        speeds_all += top.tolist()
        seen = first_seen(tracks, frames, dog_xy, ev)
        seen_all += seen
        print("- %s: %d frames, %.0f s labelled, robot moved up to %.2f m%s; %d labelled vehicles moving within 30 m"
              " (top speed median / max %s / %s km/h); from >= 25 m: %d, first tracked at %s m (median)"
              % (scene, len(frames), gt_end - frames[0][0], moved, " (MOVED: paths start off the robot)" if moved > 1.0
                 else "", len(top), _fmt(np.median(top) if len(top) else None), _fmt(top.max() if len(top) else None),
                 len(seen), _fmt(np.nanmedian([s for _, _, s in seen]) if any(np.isfinite(s) for _, _, s in seen)
                                 else None)))
        for name, p in paths.items():
            eps, info = evaluate_path(scene, name, det_to_world(p, T0), frames, tracks, gt_end, cfg, ev, T0)
            eps_all += eps
            parked.setdefault(name, []).append(info["parked_frac"])
            for e in eps:
                w = e.why or {}
                rows.append(dict(scene=scene, path=name, arrive=e.arrive - frames[0][0],
                                 go=None if e.go is None else e.go - e.arrive,
                                 oracle=None if e.oracle is None else e.oracle - e.arrive,
                                 known=int(e.known), unsafe=int(e.unsafe), risky=int(not e.oracle_ok_at_go),
                                 gap_true=e.gap_true, gap_track=e.gap_track, reason=e.reason,
                                 waited_for=" ".join(map(str, e.waited_for)), cause=w.get("cause", ""),
                                 first_dt=w.get("first_dt"), first_range=w.get("first_range"),
                                 first_bearing=w.get("first_bearing")))

    print("\n## Per path\n")
    print("crossing %.1f s (%.1f m + robot %.1f m at %.1f m/s); unsafe: labelled PET < %.1f s; risky: went where the "
          "oracle (labels, gate margins, pet %.1f s) would wait; waits in s; extra = gate - oracle\n"
          % ((args.length + 0.5 * cfg.dog_length) / cfg.dog_speed, args.length, cfg.dog_length, cfg.dog_speed,
             ev.tau_true, cfg.pet))
    print("| path | episodes | unknown | **unsafe** | risky | timeouts | wait median / p90 / max | oracle wait "
          "median / p90 | oracle waited | extra median / p90 / max | parked in path |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in list(paths) + ["all"]:
        sel = [e for e in eps_all if name == "all" or e.path == name]
        m = summarize(sel)
        waited = sum(1 for e in sel if e.known and e.oracle is not None and e.oracle > e.arrive + 1e-6)
        pk = np.mean(sum(parked.values(), [])) if name == "all" else np.mean(parked.get(name, [0.0]))
        print("| %s | %d | %d | **%d** | %d | %d | %s / %s / %s | %s / %s | %d | %s / %s / %s | %.0f%% |" % (
            name, m["episodes"], m["unknown"], m["unsafe"], m["risky"], m["timeouts"], _fmt(m["wait_med"]),
            _fmt(m["wait_p90"]), _fmt(m["wait_max"]), _fmt(m["oracle_wait_med"]), _fmt(m["oracle_wait_p90"]),
            waited, _fmt(m["extra_med"]), _fmt(m["extra_p90"]), _fmt(m["extra_max"]), 100 * pk))

    bad = [e for e in eps_all if e.unsafe]
    print("\n## Why unsafe\n")
    print("blind / far / near: the vehicle was first labelled less than %.1f s before the robot went, or after: "
          "out of the rear blind sector (%.0f-%.0f deg +- %.0f), from %.0f m or more, or closer (round a corner); "
          "occluded: labelled before but not when it went; detection: labelled, not tracked; prediction: tracked, "
          "no conflict seen\n" % (ev.recent, ev.blind_sector[0], ev.blind_sector[1], ev.blind_margin, ev.far))
    print("| path | unsafe | " + " | ".join(CAUSES) + " |")
    print("|---|---|" + "---|" * len(CAUSES))
    for name in list(paths) + ["all"]:
        m = summarize([e for e in eps_all if name == "all" or e.path == name])
        print("| %s | %d | %s |" % (name, m["unsafe"], " | ".join(str(m["causes"][c]) for c in CAUSES)))
    print("\n## Unsafe episodes (%d)\n" % len(bad))
    for e in bad[:50]:
        w = e.why or {}
        at_go = ("then %.0f m away at %.0f deg, %.0f km/h, %s" % (w["range"], w["bearing"], w["speed_kmh"],
                                                                 "tracked" if w["tracked"] else "NOT tracked")
                 if w.get("labelled") else "not labelled then")
        first = ("first labelled %+.1f s from the go at %.0f m, %.0f deg" % (w["first_dt"], w["first_range"],
                                                                           w["first_bearing"])
                 if "first_dt" in w else "")
        print("- %s %s: arrived %.1f s, went after %.1f s; labelled track %d came %.2f s after the robot left; "
              "%s; %s [%s]" % (e.scene, e.path, e.arrive - e.t0, e.go - e.arrive, e.gap_track, e.gap_true, at_go,
                               first, w.get("cause", "?")))

    print("\n## Traffic\n")
    sp = np.array(speeds_all)
    if len(sp):
        print("Labelled vehicles moving within 30 m: %d; top speed median %.1f, p90 %.1f, max %.1f km/h; "
              "above 15 km/h: %d, above 20 km/h: %d" % (len(sp), np.median(sp), np.percentile(sp, 90), sp.max(),
                                                       int(np.sum(sp > 15)), int(np.sum(sp > 20))))
    r = np.array([s for _, _, s in seen_all])
    if len(r):
        ok = r[np.isfinite(r)]
        print("\nVehicles coming from >= 25 m: %d; never tracked %d; first tracked at median %s m, p10 %s m, min %s m"
              % (len(r), int(np.sum(~np.isfinite(r))), _fmt(np.median(ok) if len(ok) else None),
                 _fmt(np.percentile(ok, 10) if len(ok) else None), _fmt(ok.min() if len(ok) else None)))

    print("\n## Sight range needed for a vehicle nobody has seen yet\n")
    print("crossing %.1f m at %.1f m/s, gap %.1f s, latency 1.1 s (tracker p90 1.0 s + node + planning)\n"
          % (args.length, cfg.dog_speed, cfg.pet))
    print("| vehicle speed | needed sight range |")
    print("|---|---|")
    for kmh in (10, 15, 20, 30):
        print("| %d km/h | %.0f m |" % (kmh, required_sight_range(args.length, kmh / KMH, 1.1, cfg)))

    if args.details:
        cols = ["scene", "path", "arrive", "go", "oracle", "known", "unsafe", "risky", "gap_true", "gap_track",
                "reason", "waited_for", "cause", "first_dt", "first_range", "first_bearing"]
        with open(args.details, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow({k: ("%.2f" % v if isinstance(v, float) else ("" if v is None else v))
                            for k, v in row.items()})
        print("\n%d episodes -> %s" % (len(rows), args.details))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
