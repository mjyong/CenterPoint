"""nuScenes-style center-distance detection metrics for the dog's own data.

AP is computed per class and per center-distance threshold (BEV), with
predictions globally sorted by score and greedily matched to the closest
unmatched ground truth of the same frame. Also reports per-range recall/AP,
which is where CenterPoint-Pillar and -Voxel differ most on a 32-beam lidar.
True positives at 2 m also give nuScenes' error metrics: ATE (center), ASE
(1 - IoU of the boxes aligned on center and heading), AOE (heading, rad) and
AVE (velocity).

Ground truth may carry an ``ignore`` mask (uncertain labels): those boxes are
not counted, and a prediction that matches no other box but lies on one
(within the distance threshold of its center, or inside its footprint
grown by 0.5 m) is dropped instead of counted as a false positive, like
KITTI's DontCare.
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


def _ignored(g):
    ign = g.get("ignore")
    return np.zeros(len(g["labels"]), dtype=bool) if ign is None else np.asarray(ign, dtype=bool)


def _on_ignore(xy, boxes, thr, margin=0.5):
    """Whether a point lies within ``thr`` of a box center or inside its BEV footprint grown by ``margin``."""
    if len(boxes) == 0:
        return False
    d = xy - boxes[:, :2]
    if np.any(np.hypot(d[:, 0], d[:, 1]) <= thr):
        return True
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return bool(np.any((np.abs(u) <= boxes[:, 3] / 2 + margin) & (np.abs(v) <= boxes[:, 4] / 2 + margin)))


def _match(frames_pred, frames_gt, cls, thr, rng_bin=None):
    preds = []
    n_gt = 0
    gt_used = []
    for f, (p, g) in enumerate(zip(frames_pred, frames_gt)):
        gm = (g["labels"] == cls) & ~_ignored(g)
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
    tp, errs, scores = [], [], []
    for s, f, i in preds:
        g, p = frames_gt[f], frames_pred[f]
        free = np.nonzero(~gt_used[f])[0]
        d = np.linalg.norm(g["boxes"][free, :2] - p["boxes"][i, :2], axis=1)
        j = int(np.argmin(d)) if len(free) else -1
        if j < 0 or d[j] > thr:
            if _on_ignore(p["boxes"][i, :2], g["boxes"][_ignored(g)], thr):
                continue        # on an uncertain label: neither TP nor FP
            tp.append(0)
            scores.append(s)
            continue
        gt_used[f][free[j]] = True
        tp.append(1)
        scores.append(s)
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
    return np.asarray(tp), n_gt, errs, np.asarray(scores, dtype=np.float64)


def evaluate_detections(frames_pred, frames_gt, class_names, dist_thresholds=(0.5, 1.0, 2.0, 4.0),
                        range_bins=((0, 10), (10, 20), (20, 40))):
    """frames_*: lists of dicts with ``boxes`` (N, 7 standard), ``labels`` (N,) int,
    ``scores`` (pred only), optional ``velocities``. Returns a nested dict."""
    res = {}
    for c, name in enumerate(class_names):
        r = {}
        aps = []
        for thr in dist_thresholds:
            tp, n_gt, errs, _ = _match(frames_pred, frames_gt, c, thr)
            ap = _ap(tp, n_gt)
            aps.append(ap)
            r["AP@%.1f" % thr] = ap
            if thr == 2.0 and errs:
                for key, metric in (("trans", "ATE"), ("scale", "ASE"), ("orient", "AOE"), ("vel", "AVE")):
                    v = [e[key] for e in errs if key in e]
                    if v:
                        r[metric] = float(np.mean(v))
        r["mAP"] = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan")
        r["num_gt"] = int(sum(int(((g["labels"] == c) & ~_ignored(g)).sum()) for g in frames_gt))
        for lo, hi in range_bins:
            tp, n_gt, _, _ = _match(frames_pred, frames_gt, c, 2.0, (lo, hi))
            r["AP@2.0[%d-%dm]" % (lo, hi)] = _ap(tp, n_gt)
        res[name] = r
    valid = [v["mAP"] for v in res.values() if not np.isnan(v["mAP"])]
    res["mAP"] = float(np.mean(valid)) if valid else float("nan")
    return res


def frame_matches(frames_pred, frames_gt, cls, thr, rng_bin=None):
    """(tp, n_gt, scores) of each frame matched on its own. Matching never crosses frames, so the AP
    of any multiset of frames (``ap_of_frames``, a bootstrap resample) follows without matching again."""
    out = []
    for p, g in zip(frames_pred, frames_gt):
        tp, n_gt, _, scores = _match([p], [g], cls, thr, rng_bin)
        out.append((tp.astype(np.float64), n_gt, scores))
    return out


def ap_of_frames(matches, idx):
    """AP over the frames ``idx`` (repeats allowed) of ``frame_matches``."""
    if len(idx) == 0:
        return float("nan")
    tp = np.concatenate([matches[i][0] for i in idx])
    scores = np.concatenate([matches[i][2] for i in idx])
    order = np.argsort(-scores, kind="stable")
    return _ap(tp[order], sum(matches[i][1] for i in idx))


def bootstrap_ap(models, cls, n=1000, seed=0, dist_thresholds=(0.5, 1.0, 2.0, 4.0), near=(0, 20)):
    """How much of a difference between detectors the validation set can tell apart.

    Frames of one recording are strongly correlated, so recordings are resampled (with replacement),
    the same draw for every model (paired). ``models``: {name: (frames_pred, frames_gt, recording of
    each frame)}. Returns {name: {"mAP": (n,), "AP@2.0[near]": (n,)}} and the values on all frames."""
    key = "AP@2.0[%d-%dm]" % near
    prep = {}
    for name, (preds, gts, rec) in models.items():
        m = [frame_matches(preds, gts, cls, t) for t in dist_thresholds] + [frame_matches(preds, gts, cls, 2.0, near)]
        rec = np.asarray(rec)
        prep[name] = (m, {r: np.nonzero(rec == r)[0] for r in np.unique(rec)})
    recs = sorted({r for _, by in prep.values() for r in by})
    rng = np.random.default_rng(seed)

    def metrics(name, draw):
        m, by = prep[name]
        idx = np.concatenate([by.get(r, np.zeros(0, np.int64)) for r in draw])
        aps = [ap_of_frames(x, idx) for x in m[:-1]]
        return (float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan"), ap_of_frames(m[-1], idx))

    full = {name: dict(zip(("mAP", key), metrics(name, recs))) for name in prep}
    out = {name: {"mAP": np.empty(n), key: np.empty(n)} for name in prep}
    for k in range(n):
        draw = rng.choice(recs, len(recs))
        for name in prep:
            out[name]["mAP"][k], out[name][key][k] = metrics(name, draw)
    return out, full


def threshold_sweep(frames_pred, frames_gt, cls, thresholds, dist=2.0, rng_bin=None):
    """Precision, recall and F1 of class ``cls`` when only detections scoring >= each threshold are
    reported (greedy matching by score within ``dist`` m, ignored ground truth not counted).
    Returns a list of dicts (threshold, tp, fp, n_gt, precision, recall, f1)."""
    tp, n_gt, _, scores = _match(frames_pred, frames_gt, cls, dist, rng_bin)
    out = []
    for t in thresholds:
        keep = scores >= t
        n_tp, n_fp = int(tp[keep].sum()), int((tp[keep] == 0).sum())
        prec = n_tp / max(n_tp + n_fp, 1)
        rec = n_tp / n_gt if n_gt else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_gt and prec + rec > 0 else 0.0
        out.append(dict(threshold=float(t), tp=n_tp, fp=n_fp, n_gt=int(n_gt), precision=prec, recall=rec, f1=f1))
    return out


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



def moving_velocity(frames_pred, frames_gt, cls, min_speed=1.0, thr=2.0, min_score=0.0, head_speed=0.15):
    """How a detector's velocity head reads moving objects: labels of class ``cls`` (not ignored) faster
    than ``min_speed``, matched per frame (``match_frame``, ``thr`` m, detections >= ``min_score``).
    Returns dict(labels, matched, speed_ratio = median |v_det| / |v_label|, read_moving = share of the
    matched with |v_det| > ``head_speed`` (what the auto-labeller needs to call a frame moving),
    error = median |v_det - v_label|)."""
    n, det_speed, gt_speed, err = 0, [], [], []
    for p, g in zip(frames_pred, frames_gt):
        if len(g["labels"]) == 0 or g.get("velocities") is None or p.get("velocities") is None:
            continue
        keep = np.asarray(p["scores"]) >= min_score
        pk = {k: np.asarray(p[k])[keep] for k in ("boxes", "labels", "scores", "velocities")}
        _, gt_m = match_frame(pk, g, thr)
        v_gt = np.asarray(g["velocities"], float).reshape(-1, 2)
        moving = (g["labels"] == cls) & ~_ignored(g) & (np.hypot(v_gt[:, 0], v_gt[:, 1]) > min_speed)
        n += int(moving.sum())
        for j in np.nonzero(moving & (gt_m >= 0))[0]:
            vd = np.asarray(pk["velocities"][gt_m[j]], float)
            det_speed.append(np.hypot(*vd))
            gt_speed.append(np.hypot(*v_gt[j]))
            err.append(np.hypot(*(vd - v_gt[j])))
    det_speed, gt_speed = np.asarray(det_speed), np.asarray(gt_speed)
    m = len(det_speed)
    return dict(labels=n, matched=m, speed_ratio=float(np.median(det_speed / gt_speed)) if m else float("nan"),
                read_moving=float(np.mean(det_speed > head_speed)) if m else float("nan"),
                error=float(np.median(err)) if m else float("nan"))
