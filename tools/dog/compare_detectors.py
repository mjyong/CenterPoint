"""Side-by-side comparison of detectors on the dog's own labelled data.

    python tools/dog/compare_detectors.py --labels data/rec_001/labels.pkl \
        --results pillar=work_dirs/rec_001/pillar/results.pkl voxel=work_dirs/rec_001/voxel/results.pkl

``--thresholds 0.1 0.15 ... 0.6`` also prints, per model and class, precision / recall / F1 at
each score threshold (2 m match, 0-40 m and 0-20 m) and two candidate reporting thresholds: the
best F1, and the highest threshold that still reaches ``--target-recall`` within 20 m.

``--bootstrap 1000`` tells whether a difference between two models is more than the luck of the
validation set: the recordings (one per labels file) are resampled with replacement, the same
draw for every model, and mAP and AP@2.0 within 20 m are recomputed. Prints each model's 90%
interval and, for every pair, the interval of the difference and how often the first model is ahead.

Several recordings at once: pass all their label files and repeat a name for
each of its results files (``voxel=rec_001/... voxel=rec_002/...``); frames
are matched by stamp.

Reports center-distance AP (overall and per range bin), translation /
velocity errors, network latency and tracking fragmentation. Labels are the
auto-labels (ideally hand-corrected); do not score the teacher against its
own labels.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import CLASSES  # noqa: E402
from dog_perception.detection.evaluate import (bootstrap_ap, evaluate_detections, format_table,  # noqa: E402
                                                threshold_sweep)
from dog_perception.label_review import train_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="labels.pkl (one per recording)")
    ap.add_argument("--results", nargs="+", required=True,
                    help="name=path/to/results.pkl; a name given several times scores all its files together")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--classes", nargs="+", default=list(CLASSES), choices=CLASSES,
                    help="classes to score (--classes vehicle for the vehicle-only models)")
    ap.add_argument("--review-policy", choices=("ignore", "keep"), default="ignore",
                    help="labels still flagged for review: not scored (as in training, default) or scored")
    ap.add_argument("--thresholds", type=float, nargs="*", default=None,
                    help="score thresholds for the precision / recall / F1 table")
    ap.add_argument("--target-recall", type=float, default=0.9,
                    help="recall within 20 m the suggested reporting threshold must keep")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="N resamples of the recordings: 90%% intervals of mAP, AP within 20 m and of the "
                         "differences between models")
    args = ap.parse_args()
    classes = list(args.classes)
    # which labels are uncertain: the same rule as export_dataset.py (vehicle-only: also a teacher
    # subclass that disagrees with the size)
    rule = "vehicle" if classes == ["vehicle"] else "coarse"
    sweeps = {}

    gt_by_stamp, rec_of = {}, {}
    for k, path in enumerate(args.labels):
        with open(path, "rb") as f:
            frames = pickle.load(f)["frames"]
        gt_by_stamp.update({round(fr["stamp"], 6): fr for fr in frames})
        rec_of.update({round(fr["stamp"], 6): k for fr in frames})
    runs = {}
    for spec in args.results:
        name, path = spec.split("=", 1)
        runs.setdefault(name, []).append(path)

    table, extra, scored = {}, {}, {}
    for name, paths in runs.items():
        res, timed, ids = [], [], {}
        for k, path in enumerate(paths):
            with open(path, "rb") as f:
                r = pickle.load(f)
            res += r
            timed += r[1:]                       # the first frame includes warm-up
            for fr in r:
                for t in fr["tracks"]:
                    ids[(k, t.track_id)] = ids.get((k, t.track_id), 0) + 1
        preds, gts, recs = [], [], []
        for r in res:
            fr = gt_by_stamp.get(round(r["stamp"], 6))
            if fr is None or fr.get("pose_bad"):
                continue
            recs.append(rec_of[round(r["stamp"], 6)])
            d = r["detections"]
            # class indices into ``classes``; other classes are not scored
            lut = np.array([classes.index(c) if c in classes else -1 for c in CLASSES], dtype=np.int64)
            keep = d.scores >= args.min_score
            preds.append(dict(boxes=d.boxes[keep], labels=lut[d.labels[keep]], scores=d.scores[keep],
                              velocities=d.velocities[keep]))
            # labels the cloud does not show at all are left out, as in export_dataset.py
            tl = [train_label(o, rule, args.review_policy) for o in fr["objects"]]
            keep_gt = [n is not None or o["label"] not in classes for o, (n, _) in zip(fr["objects"], tl)]
            objs = [o for o, k in zip(fr["objects"], keep_gt) if k]
            ign = [t[1] for t, k in zip(tl, keep_gt) if k]
            gts.append(dict(boxes=np.array([o["box"] for o in objs]).reshape(-1, 7),
                            labels=np.array([classes.index(o["label"]) if o["label"] in classes else -1
                                             for o in objs], dtype=np.int64),
                            velocities=np.array([o["velocity"] for o in objs]).reshape(-1, 2),
                            # uncertain labels are not scored (export_dataset.py: ignore boxes)
                            ignore=np.array(ign, dtype=bool)))
        table[name] = evaluate_detections(preds, gts, classes)
        scored[name] = (preds, gts, recs)
        if args.thresholds:
            sweeps[name] = {c: {rng: threshold_sweep(preds, gts, ci, sorted(args.thresholds), rng_bin=rng)
                                for rng in ((0, 40), (0, 20))} for ci, c in enumerate(classes)}
        net = [r["timings"].get("network_ms", np.nan) for r in timed]
        pre = [r["timings"].get("deskew_ms", 0) + r["timings"].get("accumulate_ms", 0) for r in timed]
        extra[name] = dict(frames=len(preds), mAP=table[name]["mAP"],
                           network_ms_median=float(np.nanmedian(net)) if net else float("nan"),
                           network_ms_p95=float(np.nanpercentile(net, 95)) if net else float("nan"),
                           preprocess_ms=float(np.mean(pre)) if pre else float("nan"),
                           tracks=len(ids), mean_track_len=float(np.mean(list(ids.values()))) if ids else 0.0)

    print(format_table(table, classes))
    print()
    print("| model | frames | mAP | net ms (median) | net ms (p95) | preprocess ms | #tracks | mean track len |")
    print("|---|---|---|---|---|---|---|---|")
    for name, e in extra.items():
        print("| %s | %d | %.3f | %.1f | %.1f | %.1f | %d | %.1f |" % (
            name, e["frames"], e["mAP"], e["network_ms_median"], e["network_ms_p95"], e["preprocess_ms"],
            e["tracks"], e["mean_track_len"]))
    for name, by_cls in sweeps.items():
        for c, by_rng in by_cls.items():
            print()
            print("%s / %s: reported detections at each score threshold (2 m match)" % (name, c))
            print("| threshold | precision 0-40 m | recall 0-40 m | F1 0-40 m | precision 0-20 m | recall 0-20 m | "
                  "F1 0-20 m | FP 0-40 m |")
            print("|---|---|---|---|---|---|---|---|")
            far, near = by_rng[(0, 40)], by_rng[(0, 20)]
            for a, b in zip(far, near):
                print("| %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d |" % (
                    a["threshold"], a["precision"], a["recall"], a["f1"], b["precision"], b["recall"], b["f1"],
                    a["fp"]))
            best = max(far, key=lambda r: r["f1"])
            ok = [r for r in near if r["recall"] >= args.target_recall]
            print("best F1 (0-40 m): threshold %.2f (F1 %.3f, recall %.3f); highest threshold with recall >= %.2f "
                  "within 20 m: %s" % (best["threshold"], best["f1"], best["recall"], args.target_recall,
                                       "%.2f (precision %.3f)" % (ok[-1]["threshold"], ok[-1]["precision"])
                                       if ok else "none"))

    if args.bootstrap:
        names = list(scored)
        for ci, c in enumerate(classes):
            boot, full = bootstrap_ap(scored, ci, args.bootstrap)
            key = [k for k in boot[names[0]] if k != "mAP"][0]
            q = lambda x: "%.3f – %.3f" % tuple(np.nanpercentile(x, [5, 95]))   # noqa: E731
            print()
            print("%s: bootstrap over the %d recordings (%d resamples, 90%% intervals)"
                  % (c, len(args.labels), args.bootstrap))
            print("| model | mAP | interval | AP@2.0 0-20 m | interval |")
            print("|---|---|---|---|---|")
            for name in names:
                print("| %s | %.3f | %s | %.3f | %s |" % (name, full[name]["mAP"], q(boot[name]["mAP"]),
                                                       full[name][key], q(boot[name][key])))
            print()
            print("| A − B | ΔmAP | interval | A ahead | ΔAP@2.0 0-20 m | interval | A ahead |")
            print("|---|---|---|---|---|---|---|")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    dm = boot[a]["mAP"] - boot[b]["mAP"]
                    dn = boot[a][key] - boot[b][key]
                    print("| %s − %s | %+.3f | %s | %.0f%% | %+.3f | %s | %.0f%% |" % (
                        a, b, full[a]["mAP"] - full[b]["mAP"], q(dm), 100 * np.nanmean(dm > 0),
                        full[a][key] - full[b][key], q(dn), 100 * np.nanmean(dn > 0)))


if __name__ == "__main__":
    main()
