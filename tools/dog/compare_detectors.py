"""Side-by-side comparison of detectors on the dog's own labelled data.

    python tools/dog/compare_detectors.py --labels data/rec_001/labels.pkl \
        --results pillar=work_dirs/rec_001/pillar/results.pkl voxel=work_dirs/rec_001/voxel/results.pkl

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
from dog_perception.detection.evaluate import evaluate_detections, format_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="name=path/to/results.pkl")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.labels, "rb") as f:
        labels = pickle.load(f)
    gt_by_stamp = {round(fr["stamp"], 6): fr for fr in labels["frames"]}

    table, extra = {}, {}
    for spec in args.results:
        name, path = spec.split("=", 1)
        with open(path, "rb") as f:
            res = pickle.load(f)
        preds, gts = [], []
        for r in res:
            fr = gt_by_stamp.get(round(r["stamp"], 6))
            if fr is None:
                continue
            d = r["detections"]
            keep = d.scores >= args.min_score
            preds.append(dict(boxes=d.boxes[keep], labels=d.labels[keep], scores=d.scores[keep],
                              velocities=d.velocities[keep]))
            objs = fr["objects"]
            gts.append(dict(boxes=np.array([o["box"] for o in objs]).reshape(-1, 7),
                            labels=np.array([CLASSES.index(o["label"]) for o in objs], dtype=np.int64),
                            velocities=np.array([o["velocity"] for o in objs]).reshape(-1, 2)))
        table[name] = evaluate_detections(preds, gts, CLASSES)
        net = [r["timings"].get("network_ms", np.nan) for r in res[1:]]
        pre = [r["timings"].get("deskew_ms", 0) + r["timings"].get("accumulate_ms", 0) for r in res[1:]]
        ids = {}
        for r in res:
            for t in r["tracks"]:
                ids.setdefault(t.track_id, 0)
                ids[t.track_id] += 1
        extra[name] = dict(frames=len(preds), mAP=table[name]["mAP"],
                           network_ms_median=float(np.nanmedian(net)) if net else float("nan"),
                           network_ms_p95=float(np.nanpercentile(net, 95)) if net else float("nan"),
                           preprocess_ms=float(np.mean(pre)) if pre else float("nan"),
                           tracks=len(ids), mean_track_len=float(np.mean(list(ids.values()))) if ids else 0.0)

    print(format_table(table, CLASSES))
    print()
    print("| model | frames | mAP | net ms (median) | net ms (p95) | preprocess ms | #tracks | mean track len |")
    print("|---|---|---|---|---|---|---|---|")
    for name, e in extra.items():
        print("| %s | %d | %.3f | %.1f | %.1f | %.1f | %d | %.1f |" % (
            name, e["frames"], e["mAP"], e["network_ms_median"], e["network_ms_p95"], e["preprocess_ms"],
            e["tracks"], e["mean_track_len"]))


if __name__ == "__main__":
    main()
