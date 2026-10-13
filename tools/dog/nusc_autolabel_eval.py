"""Check auto-labelling against nuScenes ground truth.

Runs the detector as teacher on every frame of a few nuScenes scenes (cached:
the network runs once), auto-labels each scene exactly like
``auto_label.py`` and compares, on the annotated keyframes, the raw teacher
(boxes at the reporting thresholds) and the auto-labels with the ground
truth: matches within 2 m, center / velocity / heading errors, and how fast
static objects are labelled as moving. The ego vehicle drives, so parked
cars are seen from a changing viewpoint, as from a walking robot dog.

    python tools/dog/nusc_autolabel_eval.py --dataroot data/nuscenes_mini --scenes mini_val \
        --cache work_dirs/nusc_autolabel/teacher.pkl --out work_dirs/nusc_autolabel/report.md
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.autolabel import AutoLabelConfig, auto_label  # noqa: E402
from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS, build_detector  # noqa: E402
from dog_perception.detection.boxes import CLASSES, score_threshold_array  # noqa: E402
from dog_perception.detection.centerpoint import checkpoint_arch  # noqa: E402
from dog_perception.geometry import wrap_angle  # noqa: E402
from dog_perception.nuscenes_clip import EVAL_RANGE, NuscScene, NuscTables  # noqa: E402

STATIC_SPEED = 0.2      # [m/s] GT below this is a static object
MOVING_SPEED = 0.5      # [m/s] GT above this is a moving object
MATCH_DIST = 2.0


def teacher_frames(args):
    """{scene: [dict(stamp, dets (world), gt or None, ego_xy)]}, cached in ``args.cache``."""
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            return pickle.load(f)
    tables = NuscTables(args.dataroot, args.version)
    det = build_detector(args.config or checkpoint_arch(args.ckpt), checkpoint=args.ckpt, device=args.device,
                         xy_range=51.2)
    out = {}
    for name in tables.scene_names(args.scenes):
        scene = NuscScene(tables, name, hz=args.hz, cameras=())
        rows = []
        for f in scene.frames():
            rows.append(dict(stamp=f.stamp, dets=det(f.points, f.stamp).transform(f.T_global_D, "world"),
                             gt=f.gt, ego_xy=f.T_global_D[:2, 3].copy()))
        out[name] = rows
        print("%s: %d frames" % (name, len(rows)), flush=True)
    if args.cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump(out, f)
    return out


def _match(pred_xy, gt_xy):
    """Greedy nearest pairs within MATCH_DIST. Returns [(i_pred, j_gt)]."""
    if not len(pred_xy) or not len(gt_xy):
        return []
    d = np.hypot(*(pred_xy[:, None, :] - gt_xy[None, :, :]).transpose(2, 0, 1))
    pairs = []
    while True:
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > MATCH_DIST:
            return pairs
        pairs.append((i, j))
        d[i, :] = np.inf
        d[:, j] = np.inf


def evaluate(frames, boxes_of):
    """``boxes_of(row)`` -> (boxes (N,7) world, velocities (N,2), labels (N,)). Per-class error statistics."""
    acc = {c: dict(tp=0, fp=0, fn=0, ate=[], ave=[], aoe=[], static_speed=[], moving_ave=[], moving_aoe=[])
           for c in range(len(CLASSES))}
    for row in frames:
        gt = row["gt"]
        if gt is None:
            continue
        b, v, l = boxes_of(row)
        for c in range(len(CLASSES)):
            lim = EVAL_RANGE[CLASSES[c]]
            g = np.nonzero((gt.labels == c) & (gt.num_pts > 0) &
                           (np.hypot(*(gt.boxes[:, :2] - row["ego_xy"]).T) <= lim))[0]
            p = np.nonzero((l == c) & (np.hypot(*(b[:, :2] - row["ego_xy"]).T) <= lim))[0]
            pairs = _match(b[p, :2], gt.boxes[g, :2])
            a = acc[c]
            a["tp"] += len(pairs)
            a["fp"] += len(p) - len(pairs)
            a["fn"] += len(g) - len(pairs)
            for i, j in pairs:
                pi, gj = p[i], g[j]
                a["ate"].append(np.hypot(*(b[pi, :2] - gt.boxes[gj, :2])))
                gv = gt.velocities[gj]
                yaw_err = abs(wrap_angle(b[pi, 6] - gt.boxes[gj, 6]))
                a["aoe"].append(yaw_err)
                if not np.all(np.isfinite(gv)):
                    continue
                dv = np.hypot(*(v[pi] - gv))
                a["ave"].append(dv)
                gs = np.hypot(*gv)
                if gs < STATIC_SPEED:
                    a["static_speed"].append(np.hypot(*v[pi]))
                elif gs > MOVING_SPEED:
                    a["moving_ave"].append(dv)
                    a["moving_aoe"].append(yaw_err)
    return acc


def summarize(acc):
    rows = {}
    for c, a in acc.items():
        ss = np.array(a["static_speed"])
        rows[CLASSES[c]] = dict(
            TP=a["tp"], FP=a["fp"], FN=a["fn"],
            precision=a["tp"] / max(a["tp"] + a["fp"], 1), recall=a["tp"] / max(a["tp"] + a["fn"], 1),
            ATE=float(np.mean(a["ate"])) if a["ate"] else np.nan,
            AVE=float(np.mean(a["ave"])) if a["ave"] else np.nan,
            AOE=float(np.mean(a["aoe"])) if a["aoe"] else np.nan,
            static_n=len(ss), static_speed_median=float(np.median(ss)) if len(ss) else np.nan,
            static_over_05=float(np.mean(ss > 0.5)) if len(ss) else np.nan,
            moving_n=len(a["moving_ave"]),
            moving_AVE=float(np.mean(a["moving_ave"])) if a["moving_ave"] else np.nan,
            moving_AOE=float(np.mean(a["moving_aoe"])) if a["moving_aoe"] else np.nan)
    return rows


def teacher_boxes(row):
    d = row["dets"]
    keep = d.scores >= score_threshold_array(d.labels, DEFAULT_SCORE_THRESHOLDS)
    return d.boxes[keep], d.velocities[keep], d.labels[keep]


def label_boxes(labels):
    def f(row):
        objs = labels.get(round(float(row["stamp"]), 6), [])
        if not objs:
            return np.zeros((0, 7)), np.zeros((0, 2)), np.zeros(0, np.int64)
        return (np.stack([o["box"] for o in objs]), np.stack([o["velocity"] for o in objs]),
                np.array([o["label"] for o in objs]))
    return f


def write_report(path, results, meta):
    cols = [("TP", "%d"), ("FP", "%d"), ("FN", "%d"), ("precision", "%.3f"), ("recall", "%.3f"), ("ATE", "%.2f"),
            ("AVE", "%.2f"), ("AOE", "%.2f"), ("static_speed_median", "%.2f"), ("static_over_05", "%.2f"),
            ("moving_AVE", "%.2f"), ("moving_AOE", "%.2f")]
    lines = ["# Auto-labels vs nuScenes ground truth", "",
             "- scenes: %s (%d frames, %d keyframes)" % (", ".join(meta["scenes"]), meta["frames"], meta["keyframes"]),
             "- teacher: `%s`; teacher row = boxes at the reporting thresholds %s" % (meta["ckpt"],
                                                                                  DEFAULT_SCORE_THRESHOLDS),
             "- GT with lidar points within the class range; match <= %.0f m center distance; ATE / AVE / AOE are "
             "means over matches (m, m/s, rad); static = GT speed < %.1f m/s (label speed median and share "
             "> 0.5 m/s), moving = GT speed > %.1f m/s" % (MATCH_DIST, STATIC_SPEED, MOVING_SPEED), ""]
    for cls in CLASSES:
        lines += ["## %s" % cls, "", "| | " + " | ".join(k for k, _ in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for name, res in results.items():
            r = res[cls]
            lines.append("| %s | %s |" % (name, " | ".join(
                ("-" if (isinstance(r[k], float) and np.isnan(r[k])) else fmt % r[k]) for k, fmt in cols)))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", default="mini_val")
    ap.add_argument("--ckpt", default="work_dirs/pretrained/nusc_pp.pth")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--cache", default=None, help="teacher detections pickle (written on the first run)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="work_dirs/nusc_autolabel/report.md")
    args = ap.parse_args()

    scenes = teacher_frames(args)
    frames = [r for rows in scenes.values() for r in rows]
    labels = {}
    for rows in scenes.values():
        labels.update(auto_label([(r["stamp"], r["dets"]) for r in rows], AutoLabelConfig()))
    results = {"teacher": summarize(evaluate(frames, teacher_boxes)),
               "auto-label": summarize(evaluate(frames, label_boxes(labels)))}
    meta = dict(scenes=list(scenes), frames=len(frames), keyframes=sum(r["gt"] is not None for r in frames),
                ckpt=args.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_report(args.out, results, meta)
    print(open(args.out).read())


if __name__ == "__main__":
    main()
