"""nuScenes-style center-distance detection metrics for the dog's own data.

AP is computed per class and per center-distance threshold (BEV), with
predictions globally sorted by score and greedily matched to the closest
unmatched ground truth of the same frame. Also reports per-range recall/AP,
which is where CenterPoint-Pillar and -Voxel differ most on a 32-beam lidar.
True positives at 2 m also give nuScenes' error metrics: ATE (center), ASE
(1 - IoU of the boxes aligned on center and heading), AOE (heading, rad) and
AVE (velocity).
"""
import numpy as np

from ..geometry import wrap_angle


def scale_error(a, b):
    """1 - IoU of two (l, w, h) boxes aligned on center and heading."""
    inter = np.prod(np.minimum(a, b))
    return float(1.0 - inter / (np.prod(a) + np.prod(b) - inter))


def _ap(tp, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / n_gt
    # precision envelope, all-point interpolation
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - r_prev) * prec))


def _match(frames_pred, frames_gt, cls, thr, rng_bin=None):
    preds = []
    n_gt = 0
    gt_used = []
    for f, (p, g) in enumerate(zip(frames_pred, frames_gt)):
        gm = g["labels"] == cls
        pm = p["labels"] == cls
        if rng_bin is not None:
            gr = np.hypot(g["boxes"][:, 0], g["boxes"][:, 1])
            pr = np.hypot(p["boxes"][:, 0], p["boxes"][:, 1])
            gm &= (gr >= rng_bin[0]) & (gr < rng_bin[1])
            pm &= (pr >= rng_bin[0]) & (pr < rng_bin[1])
        n_gt += int(gm.sum())
        gt_used.append(np.zeros(len(g["labels"]), dtype=bool) | ~gm)
        for i in np.nonzero(pm)[0]:
            preds.append((p["scores"][i], f, i))
    preds.sort(key=lambda x: -x[0])
    tp, errs = [], []
    for s, f, i in preds:
        g, p = frames_gt[f], frames_pred[f]
        free = np.nonzero(~gt_used[f])[0]
        if len(free) == 0:
            tp.append(0)
            continue
        d = np.linalg.norm(g["boxes"][free, :2] - p["boxes"][i, :2], axis=1)
        j = int(np.argmin(d))
        if d[j] <= thr:
            gt_used[f][free[j]] = True
            tp.append(1)
            e = {"trans": d[j]}
            gb, pb = g["boxes"][free[j]], p["boxes"][i]
            if len(gb) >= 7 and len(pb) >= 7:
                e["scale"] = scale_error(gb[3:6], pb[3:6])
                e["orient"] = float(abs(wrap_angle(pb[6] - gb[6])))
            if "velocities" in g and "velocities" in p:
                gv = g["velocities"][free[j]]
                if np.all(np.isfinite(gv)):
                    e["vel"] = float(np.linalg.norm(gv - p["velocities"][i]))
            errs.append(e)
        else:
            tp.append(0)
    return np.asarray(tp), n_gt, errs


def evaluate_detections(frames_pred, frames_gt, class_names, dist_thresholds=(0.5, 1.0, 2.0, 4.0),
                        range_bins=((0, 10), (10, 20), (20, 40))):
    """frames_*: lists of dicts with ``boxes`` (N, 7 standard), ``labels`` (N,) int,
    ``scores`` (pred only), optional ``velocities``. Returns a nested dict."""
    res = {}
    for c, name in enumerate(class_names):
        r = {}
        aps = []
        for thr in dist_thresholds:
            tp, n_gt, errs = _match(frames_pred, frames_gt, c, thr)
            ap = _ap(tp, n_gt)
            aps.append(ap)
            r["AP@%.1f" % thr] = ap
            if thr == 2.0 and errs:
                for key, metric in (("trans", "ATE"), ("scale", "ASE"), ("orient", "AOE"), ("vel", "AVE")):
                    v = [e[key] for e in errs if key in e]
                    if v:
                        r[metric] = float(np.mean(v))
        r["mAP"] = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan")
        r["num_gt"] = int(sum(int((g["labels"] == c).sum()) for g in frames_gt))
        for lo, hi in range_bins:
            tp, n_gt, _ = _match(frames_pred, frames_gt, c, 2.0, (lo, hi))
            r["AP@2.0[%d-%dm]" % (lo, hi)] = _ap(tp, n_gt)
        res[name] = r
    valid = [v["mAP"] for v in res.values() if not np.isnan(v["mAP"])]
    res["mAP"] = float(np.mean(valid)) if valid else float("nan")
    return res


def format_table(results_by_model, class_names):
    """results_by_model: {model_name: evaluate_detections(...)} -> markdown table."""
    keys = None
    lines = []
    for model, res in results_by_model.items():
        for name in class_names:
            r = res[name]
            if keys is None:
                keys = [k for k in r if k != "num_gt"]
                lines.append("| model | class | " + " | ".join(keys) + " | #gt |")
                lines.append("|" + "---|" * (len(keys) + 3))
            vals = ["%.3f" % r[k] if k in r and not np.isnan(r[k]) else "-" for k in keys]
            lines.append("| %s | %s | %s | %d |" % (model, name, " | ".join(vals), r["num_gt"]))
    return "\n".join(lines)


def match_frame(pred, gt, thr=2.0):
    """Per-frame greedy matching (by score, same class, center distance <= thr),
    for drawing: returns (pred_tp (P,) bool, gt_matched (G,) int index of the
    matched prediction or -1)."""
    order = np.argsort(-np.asarray(pred["scores"]), kind="stable")
    tp = np.zeros(len(order), dtype=bool)
    gt_m = np.full(len(gt["labels"]), -1, dtype=np.int64)
    for i in order:
        cand = np.nonzero((gt["labels"] == pred["labels"][i]) & (gt_m < 0))[0]
        if len(cand) == 0:
            continue
        d = np.linalg.norm(gt["boxes"][cand, :2] - pred["boxes"][i, :2], axis=1)
        j = int(np.argmin(d))
        if d[j] <= thr:
            gt_m[cand[j]] = i
            tp[i] = True
    return tp, gt_m
