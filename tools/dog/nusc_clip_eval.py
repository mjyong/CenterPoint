"""Run detection -> tracking -> prediction on nuScenes clips, compare with the
ground truth, write one Foxglove MCAP per scene and a metrics report.

    python tools/dog/nusc_clip_eval.py --dataroot /data/nuscenes --version v1.0-mini \
        --scenes mini_val --ckpt work_dirs/pretrained/nusc_pp.pth --out work_dirs/nusc_eval

``--scenes`` takes a split name (mini_val, val, ...) or comma-separated scene
names. The pretrained nuScenes weights were trained on the train split, so
evaluate on mini_val / val scenes; mini_train scenes were seen in training.

Everything runs on the dog pipeline's code: the same detector wrapper and
filters (fed nuScenes LIDAR_TOP through the inverse ModelFrame), the same
tracker in the nuScenes global frame, the same IMM predictor. nuScenes
classes are merged into vehicle / pedestrian / cyclist on both sides.

Outputs in ``--out``: ``<scene>.mcap`` (see dog_perception/clip_viz.py for the
topics), ``report.md``, ``report.json``.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.clip_viz import bev_figure  # noqa: E402
from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS, PRESET_CONFIGS, build_detector  # noqa: E402
from dog_perception.detection.boxes import CLASSES, score_threshold_array  # noqa: E402
from dog_perception.detection.centerpoint import checkpoint_arch  # noqa: E402
from dog_perception.detection.evaluate import evaluate_detections, match_frame  # noqa: E402
from dog_perception.geometry import inv_T  # noqa: E402
from dog_perception.nuscenes_clip import EVAL_RANGE, NuscScene, NuscTables, boxes_to_frame  # noqa: E402
from dog_perception.prediction.kinematic import IMMPredictor  # noqa: E402
from dog_perception.prediction.metrics import displacement_metrics  # noqa: E402
from dog_perception.tracking import MultiObjectTracker, TrackerConfig  # noqa: E402
from dog_perception.tracking.evaluate import MotAccumulator  # noqa: E402

RANGE_BINS = ((0, 20), (20, 30), (30, 50))
ID_STRIDE = 100000          # GT / track ids are made unique across scenes


def in_range(xy, origin, labels):
    lim = np.array([EVAL_RANGE[CLASSES[l]] if l >= 0 else -1.0 for l in labels])
    return np.hypot(xy[:, 0] - origin[0], xy[:, 1] - origin[1]) <= lim


def pad_modes(samples, key_modes="modes", key_probs="probs"):
    K = max(len(s[key_probs]) for s in samples)
    T = samples[0]["gt"].shape[0]
    M = np.zeros((len(samples), K, T, 2))
    P = np.zeros((len(samples), K))
    for b, s in enumerate(samples):
        k = len(s[key_probs])
        M[b, :k], P[b, :k] = s[key_modes], s[key_probs]
        M[b, k:] = s[key_modes][int(np.argmax(s[key_probs]))]
    return M, P


def prediction_metrics(samples):
    if not samples:
        return {}
    gt = np.stack([s["gt"] for s in samples])
    M, P = pad_modes(samples)
    out = {"IMM": displacement_metrics(M, P, gt)}
    cv = np.stack([s["cv"][None] for s in samples])
    out["CV"] = displacement_metrics(cv, np.ones((len(samples), 1)), gt)
    if "learned_modes" in samples[0]:
        M2, P2 = pad_modes(samples, "learned_modes", "learned_probs")
        out["learned"] = displacement_metrics(M2, P2, gt)
    return out


class Timer:
    def __init__(self):
        self.ms = defaultdict(list)

    def add(self, key, t0):
        self.ms[key].append(1e3 * (time.perf_counter() - t0))
        return time.perf_counter()

    def summary(self):
        return {k: float(np.median(v)) for k, v in self.ms.items()}


def run_scene(scene, si, det, tracker, predictor, acc, store, args, timer):
    from dog_perception import clip_viz

    tracker.reset()
    off = si * ID_STRIDE
    writer = viz = None
    if not args.no_mcap:
        from dog_perception.foxglove import McapWriter
        writer = McapWriter(os.path.join(args.out, "%s.mcap" % scene.name))
        viz = clip_viz.ClipVisualizer(writer, frame_period=1.0 / args.hz)
    ego_hist = deque(maxlen=64)
    T_ego_cams = {}
    last_ego = None
    cum = defaultdict(float)
    times = predictor.step * np.arange(1, int(round(predictor.horizon / predictor.step)) + 1) \
        if hasattr(predictor, "step") else 0.5 * np.arange(1, 7)
    n_frames = len(scene) if args.max_frames is None else min(args.max_frames, len(scene))
    keys = [i for i, s in enumerate(scene.selected[:n_frames]) if s["is_key_frame"]]
    snap_index = keys[len(keys) // 2] if keys else -1
    t0 = time.perf_counter()
    for f in scene.frames(0, n_frames):
        t0 = timer.add("load", t0)
        dets = det(f.points, f.stamp)
        timer.ms["det_network"].append(det.last_timing.get("network_ms", 0.0))
        t0 = timer.add("detect", t0)
        dets_w = dets.transform(f.T_global_D, "world")
        tracks = tracker.step(dets_w, f.stamp)
        t0 = timer.add("track", t0)
        ego_p = f.T_global_ego[:3, 3]
        ego_v = (ego_p - last_ego[1]) / (f.stamp - last_ego[0]) if last_ego is not None and f.stamp > last_ego[0] \
            else np.zeros(3)
        last_ego = (f.stamp, ego_p)
        ego_hist.append(np.array([f.stamp, ego_p[0], ego_p[1], ego_v[0], ego_v[1], 1.0]))
        preds = predictor(tracker, np.asarray(ego_hist)) if args.predictor_model else predictor(tracker)
        t0 = timer.add("predict", t0)

        lidar_xy = f.T_global_lidar[:2, 3]
        reported = dets.scores >= score_threshold_array(dets.labels, DEFAULT_SCORE_THRESHOLDS)
        trk_xy = np.array([s.position[:2] for s in tracks]).reshape(-1, 2)
        trk_lab = np.array([s.label for s in tracks], dtype=np.int64)
        trk_ok = in_range(trk_xy, lidar_xy, trk_lab) if len(tracks) else np.zeros(0, bool)
        pred_by_id = {p.track_id: p for p in preds}

        ev, gt_items, det_tp, det_gt = None, [], None, {}
        trk_info, fde_info = {}, {}
        if f.is_key:
            gt = f.gt
            ok = (gt.labels >= 0) & (gt.num_pts > 0) & in_range(gt.boxes[:, :2], lidar_xy, gt.labels)
            # detection AP: all detector output (score >= 0.1) vs evaluated GT, both in the D frame
            T_Dg = inv_T(f.T_global_D)
            gt_D = boxes_to_frame(gt.boxes[ok], T_Dg)
            gt_vD = gt.velocities[ok] @ T_Dg[:2, :2].T
            d_ok = in_range(dets.boxes[:, :2], (0.0, 0.0), dets.labels)
            store["det_pred"].append(dict(boxes=dets.boxes[d_ok], scores=dets.scores[d_ok], labels=dets.labels[d_ok],
                                          velocities=dets.velocities[d_ok]))
            store["det_gt"].append(dict(boxes=gt_D, labels=gt.labels[ok], velocities=gt_vD))
            # reported detections at the operating point (for drawing and precision / recall)
            rep_idx = np.nonzero(reported & d_ok)[0]
            tp, gt_m = match_frame(dict(boxes=dets.boxes[rep_idx], scores=dets.scores[rep_idx],
                                        labels=dets.labels[rep_idx]), dict(boxes=gt_D, labels=gt.labels[ok]))
            det_tp = np.zeros(len(dets), bool)
            det_tp[rep_idx[tp]] = True
            gt_ids_ok = gt.ids[ok]
            for j, i in enumerate(gt_m):
                if i >= 0:
                    det_gt[int(rep_idx[i])] = int(gt_ids_ok[j])
            for c in range(len(CLASSES)):
                cum["op_tp_%d" % c] += int(tp[dets.labels[rep_idx] == c].sum())
                cum["op_fp_%d" % c] += int((~tp[dets.labels[rep_idx] == c]).sum())
                cum["op_gt_%d" % c] += int((gt.labels[ok] == c).sum())
            # tracking
            ev = acc.update(gt.ids[ok] + off, gt.boxes[ok, :2], gt.labels[ok],
                            np.array([s.track_id for s in tracks], dtype=np.int64)[trk_ok] + off, trk_xy[trk_ok],
                            trk_lab[trk_ok])
            matched = {g - off: (tid - off, d) for g, tid, d, _ in ev["matches"]}
            for g, (tid, d) in matched.items():
                trk_info[tid] = {"eval": "TP", "gt_id": g, "center_err_m": "%.2f" % d}
            for tid in ev["fp"]:
                trk_info[tid - off] = {"eval": "FP"}
            _error_breakdown(store, ev, off, gt, ok, tracks, dets, reported, f.T_global_D, lidar_xy)
            # prediction vs GT future
            fdes = []
            for g, (tid, _) in matched.items():
                p = pred_by_id.get(tid)
                fut = scene.future(g, f.stamp, times)
                if p is None or fut is None:
                    continue
                s = next(s for s in tracks if s.track_id == tid)
                cur = gt.boxes[gt.ids == g][0, :2]
                sample = dict(scene=scene.name, t=f.stamp, cls=int(s.label), gt_id=g, trk_id=tid, gt=fut,
                              modes=p.modes, probs=p.probs, cv=s.position[:2] + np.outer(times, s.velocity[:2]),
                              moving=bool(np.linalg.norm(fut[-1] - cur) >= 1.5))
                store["pred"].append(sample)
                fde_k = np.linalg.norm(p.modes[:, -1] - fut[-1], axis=1)
                f1 = float(fde_k[int(np.argmax(p.probs))])
                fde_info[tid] = (fut[-1], f1, float(fde_k.min()))
                trk_info.setdefault(tid, {})["FDE_top_m"] = "%.2f" % f1
                fdes.append(f1)
            # GT drawing items
            status = {g - off: "tracked" for g, _, _, _ in ev["matches"]}
            status.update({g - off: "missed" for g in ev["fn"]})
            for j in range(len(gt.labels)):
                g = int(gt.ids[j])
                b = gt.boxes[j]
                gt_items.append(dict(
                    box=b, id=g, label=int(gt.labels[j]), status=status.get(g, "ignored"), vel=gt.velocities[j],
                    past=_path(scene.instance_path(g, f.stamp - 2.0, f.stamp), b),
                    future=_path(scene.instance_path(g, f.stamp, f.stamp + 3.0), b),
                    meta={"gt_id": g, "category": gt.category[j], "attribute": gt.attribute[j],
                          "visibility": gt.visibility[j], "lidar_pts": int(gt.num_pts[j]),
                          "status": status.get(g, "not evaluated"),
                          "track": matched.get(g, ("-", 0))[0],
                          "speed_mps": "%.2f" % np.hypot(*gt.velocities[j]) if np.all(np.isfinite(gt.velocities[j]))
                          else "-"}))
            for k in ("tp", "fp", "fn", "idsw"):
                cum["trk_" + k] += len(ev["matches"]) if k == "tp" else len(ev[k])
            cum["trk_gt"] += int(ok.sum())
            frame_metrics = dict(
                t=f.stamp, gt=int(ok.sum()), det_tp=int(tp.sum()), det_fp=int((~tp).sum()),
                det_fn=int(ok.sum() - tp.sum()), trk_tp=len(ev["matches"]), trk_fp=len(ev["fp"]),
                trk_fn=len(ev["fn"]), idsw=len(ev["idsw"]),
                mota_cum=1 - (cum["trk_fn"] + cum["trk_fp"] + cum["trk_idsw"]) / max(cum["trk_gt"], 1),
                motp_frame=float(np.mean([d for _, _, d, _ in ev["matches"]])) if ev["matches"] else float("nan"),
                fde_top_mean=float(np.mean(fdes)) if fdes else float("nan"),
                tracks=len(tracks), detections=int(reported.sum()))
        elif viz is not None:
            boxes, labels, ids = scene.interpolated(f.stamp)
            for b, l, g in zip(boxes, labels, ids):
                if l >= 0:
                    gt_items.append(dict(box=b, id=int(g), label=int(l), status="interp", vel=None,
                                         past=_path(scene.instance_path(g, f.stamp - 2.0, f.stamp), b),
                                         future=_path(scene.instance_path(g, f.stamp, f.stamp + 3.0), b)))

        if f.index == snap_index:
            cur = f.points[:f.n_current:3]
            store["snapshots"].append(dict(
                scene=scene.name, t=f.stamp, t0=1e-6 * scene.selected[0]["timestamp"], T_global_ego=f.T_global_ego,
                gt=gt_items, tracks=tracks, preds=preds,
                metrics=frame_metrics, lidar=cur[:, :3] @ f.T_global_D[:3, :3].T + f.T_global_D[:3, 3]))
        if viz is not None:
            items = [("map", "ego", f.T_global_ego), ("ego", "LIDAR_TOP", inv_T(f.T_global_ego) @ f.T_global_lidar)]
            for ch, cam in f.cameras.items():
                T_ego_cams[ch] = cam["T_ego_cam"]
            items += [("ego", ch, T) for ch, T in T_ego_cams.items()]
            viz.transforms(f.stamp, items)
            cur = f.points[:f.n_current]
            xyz = cur[:, :3] @ f.T_global_D[:3, :3].T + f.T_global_D[:3, 3]
            viz.lidar(f.stamp, xyz[::args.lidar_stride], cur[::args.lidar_stride, 3])
            viz.ego(f.stamp, f.T_global_ego, float(np.hypot(*ego_v[:2])))
            viz.ground_truth(f.stamp, gt_items, f.is_key)
            viz.detections(f.stamp, dets_w, reported, det_tp, det_gt)
            viz.tracks(f.stamp, tracks, trk_info)
            z_of = {}
            for s in tracks:
                z_of[s.track_id] = s.position[2] - s.size[2] / 2 + 0.15
                z_of[("xy", s.track_id)] = s.position[:2]
            viz.predictions(f.stamp, preds, z_of, fde_info if f.is_key else None)
            if f.is_key:
                gt_pos = {g + off: b[:3] for g, b in zip(f.gt.ids, f.gt.boxes)}
                trk_pos = {s.track_id + off: s.position for s in tracks}
                viz.events(f.stamp, ev, gt_pos, trk_pos)
                viz.metrics(f.stamp, frame_metrics)
                for ch, cam in f.cameras.items():
                    with open(cam["path"], "rb") as fh:
                        jpeg = fh.read()
                    from dog_perception.clip_viz import GT_RGB, track_rgb
                    boxes = [(it["box"], GT_RGB[it["status"]], "GT#%d" % it["id"]) for it in gt_items
                             if it["status"] in ("tracked", "missed")]
                    boxes += [(np.r_[s.position, s.size, s.yaw], track_rgb(s.track_id), "T%d" % s.track_id)
                              for s in tracks]
                    viz.camera(cam["stamp"], ch, cam, jpeg, boxes)
            t0 = timer.add("visualize", t0)
        else:
            t0 = time.perf_counter()
    for k, v in cum.items():
        store["op"][k] += v
    if writer is not None:
        writer.close()
        store["mcap"].append((writer.path, writer.messages, os.path.getsize(writer.path)))
    store["frames"] += n_frames
    store["keyframes"] += sum(1 for s in scene.selected[:n_frames] if s["is_key_frame"])


FP_CAUSES = ("second track on a tracked object (< 1 m from its track)", "next to a tracked object of its class",
             "other class on a GT object", "on a GT that is not evaluated", "coasting after the object was lost",
             "no object nearby")


def _error_breakdown(store, ev, off, gt, ok, tracks, dets, reported, T_global_D, lidar_xy, thr=2.0):
    """Why tracks are false and why GT objects are missed (keyframes)."""
    by_id = {s.track_id: s for s in tracks}
    track_of_gt = {g - off: t - off for g, t, _, _ in ev["matches"]}
    for tid in ev["fp"]:
        s = by_id[tid - off]
        d = np.hypot(*(gt.boxes[:, :2] - s.position[:2]).T) if len(gt.labels) else np.zeros(0)
        near = d <= thr
        same = np.nonzero(near & ok & (gt.labels == s.label))[0]
        twin = [track_of_gt.get(int(gt.ids[j])) for j in same]
        if any(t is not None and np.hypot(*(by_id[t].position[:2] - s.position[:2])) < 1.0 for t in twin):
            cause = FP_CAUSES[0]
        elif len(same):
            cause = FP_CAUSES[1]
        elif np.any(near & ok):
            cause = FP_CAUSES[2]
        elif np.any(near):
            cause = FP_CAUSES[3]
        elif s.coasting:
            cause = FP_CAUSES[4]
        else:
            cause = FP_CAUSES[5]
        store["fp_cause"][(int(s.label), cause)] += 1
    if not ev["fn"]:
        return
    rep_xy = (dets.boxes[reported, :2] @ T_global_D[:2, :2].T + T_global_D[:2, 3])
    rep_lab = dets.labels[reported]
    for g in ev["fn"]:
        j = int(np.nonzero(gt.ids == g - off)[0][0])
        lab, xy = int(gt.labels[j]), gt.boxes[j, :2]
        detected = bool(np.any((rep_lab == lab) & (np.hypot(*(rep_xy - xy).T) <= thr))) if len(rep_lab) else False
        rng = float(np.hypot(*(xy - lidar_xy)))
        band = "0-20 m" if rng < 20 else ("20-30 m" if rng < 30 else "30-50 m")
        pts = int(gt.num_pts[j])
        pts_band = "1-9 pts" if pts < 10 else ("10-49 pts" if pts < 50 else ">=50 pts")
        store["fn_cause"][(lab, "detected, not tracked" if detected else "not detected")] += 1
        store["fn_range"][(lab, band)] += 1
        store["fn_pts"][(lab, pts_band)] += 1


def _path(tpx, box):
    if len(tpx) == 0:
        return None
    z = box[2] - box[5] / 2 + 0.1
    return np.column_stack([tpx[:, 1], tpx[:, 2], np.full(len(tpx), z)])


# ------------------------------------------------------------------ report
def _f(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (("%." + str(nd) + "f") % v
                                                                            if isinstance(v, float) else str(v))


def write_report(path_md, path_json, meta, det, op, mot, pred, timing, errors=None):
    L = ["# nuScenes clip evaluation", ""]
    L += ["- data: %s, scenes %s (%d frames at %s Hz, %d keyframes evaluated)" % (
        meta["version"], ", ".join(meta["scenes"]), meta["frames"], meta["hz"], meta["keyframes"]),
        "- detector: `%s` (`%s`), classes merged to vehicle / pedestrian / cyclist" % (meta["ckpt"], meta["config"]),
        "- matching: center distance in BEV; GT counted when it has lidar points and lies within the class range "
        "(vehicle 50 m, pedestrian / cyclist 40 m), predictions and tracks filtered the same way", ""]
    L += ["## Detection", "", "AP over the detector output (score >= 0.1) at 0.5 / 1 / 2 / 4 m; error metrics on "
          "true positives at 2 m (ATE m, ASE 1-IoU, AOE rad, AVE m/s).", "",
          "| class | AP@0.5 | AP@1 | AP@2 | AP@4 | mAP | ATE | ASE | AOE | AVE | AP@2 0-20m | 20-30m | 30-50m | #GT |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        r = det[c]
        L.append("| %s | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %d |" % (
            c, _f(r.get("AP@0.5")), _f(r.get("AP@1.0")), _f(r.get("AP@2.0")), _f(r.get("AP@4.0")), _f(r["mAP"]),
            _f(r.get("ATE"), 2), _f(r.get("ASE"), 2), _f(r.get("AOE"), 2), _f(r.get("AVE"), 2),
            _f(r.get("AP@2.0[0-20m]")), _f(r.get("AP@2.0[20-30m]")), _f(r.get("AP@2.0[30-50m]")), r["num_gt"]))
    L += ["", "mAP over the three classes: **%s**" % _f(det["mAP"]), "",
          "At the reporting thresholds (vehicle 0.35, pedestrian 0.30, cyclist 0.40):", "",
          "| class | precision | recall | TP | FP | #GT |", "|---|---|---|---|---|---|"]
    for c, name in enumerate(CLASSES):
        tp, fp, n = op.get("op_tp_%d" % c, 0), op.get("op_fp_%d" % c, 0), op.get("op_gt_%d" % c, 0)
        L.append("| %s | %s | %s | %d | %d | %d |" % (name, _f(tp / (tp + fp) if tp + fp else float("nan")),
                                                     _f(tp / n if n else float("nan")), tp, fp, n))
    L += ["", "## Tracking", "", "Reported tracks (confirmed, coasting included) vs GT instances, 2 m.", "",
          "| class | MOTA | MOTP (m) | IDF1 | recall | precision | IDSW | FRAG | MT | ML | GT tracks | TP | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in list(CLASSES) + ["all"]:
        r = mot[c]
        L.append("| %s | **%s** | %s | %s | %s | %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            c, _f(r["MOTA"]), _f(r["MOTP"], 2), _f(r["IDF1"]), _f(r["recall"]), _f(r["precision"]), r["IDSW"],
            r["FRAG"], r["MT"], r["ML"], r["num_gt_tracks"], r["TP"], r["FP"], r["FN"]))
    if errors:
        L += ["", "### Where the tracking errors come from", "", "False tracks (FP) by cause:", "",
              "| cause | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for cause in FP_CAUSES:
            L.append("| %s | %s |" % (cause, " | ".join(str(errors["fp_cause"].get("%d|%s" % (c, cause), 0))
                                                         for c in range(len(CLASSES)))))
        L += ["", "Missed GT objects (FN): was there a reported detection within 2 m? How far, how many lidar points?",
              "", "| | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for key, cats in (("fn_cause", ("detected, not tracked", "not detected")),
                          ("fn_range", ("0-20 m", "20-30 m", "30-50 m")),
                          ("fn_pts", ("1-9 pts", "10-49 pts", ">=50 pts"))):
            for cat in cats:
                L.append("| %s | %s |" % (cat, " | ".join(str(errors[key].get("%d|%s" % (c, cat), 0))
                                                         for c in range(len(CLASSES)))))
    L += ["", "## Prediction", "", "Tracks matched to a GT object with 3 s of annotated future; 6 steps of 0.5 s. "
          "IMM = tier 1 (all modes with their probabilities), CV = constant velocity from the track. "
          "minADE / minFDE: best mode; ADE1 / FDE1: most likely mode; MR: no mode ends within 2 m.", ""]
    for subset, res in pred.items():
        if not res:
            continue
        L += ["**%s** (%d samples)" % (subset, res["IMM"]["num"]), "",
              "| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |", "|---|---|---|---|---|---|"]
        for name, r in res.items():
            L.append("| %s | %s | %s | %s | %s | %s |" % (name, _f(r["minADE"], 2), _f(r["minFDE"], 2),
                                                          _f(r["ADE1"], 2), _f(r["FDE1"], 2), _f(r["MR"])))
        L.append("")
    if meta.get("figures"):
        L += ["## Snapshots", "", "Bird's-eye view at the middle keyframe of each scene (ego at the center, heading up)."
              " Green / red outlines: GT tracked / missed; filled boxes: reported tracks (by class) with their id; "
              "colored lines: the two most likely predicted modes (opacity = probability); black dashes: GT 3 s "
              "future.", ""]
        L += ["![%s](%s)" % (f, f) for f in meta["figures"]] + [""]
    L += ["## Timing (median ms per frame, this machine)", "", "| stage | ms |", "|---|---|"]
    L += ["| %s | %.1f |" % (k, v) for k, v in timing.items()]
    L += ["", "## Output", ""] + ["- `%s`: %d messages, %.0f MB" % (p, n, s / 1e6) for p, n, s in meta["mcap"]]
    L += ["", "Open the MCAP in Foxglove: 3D panel (follow frame `ego`), Image panel on `/cam/CAM_FRONT/image` "
          "with `/cam/CAM_FRONT/annotations`, Plot panel on `/metrics/frame.mota_cum`, Log panel on `/log`."]
    with open(path_md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(path_json, "w") as fh:
        json.dump(_clean(dict(meta=meta, detection=det, detection_operating_point=op, tracking=mot,
                              tracking_errors=errors, prediction=pred, timing_ms=timing)), fh, indent=2)
    return "\n".join(L)


def _clean(o):
    """JSON-safe copy: NaN -> null, numpy scalars / arrays -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", default="mini_val", help="split name or comma-separated scene names")
    ap.add_argument("--ckpt", default="work_dirs/pretrained/nusc_pp.pth")
    ap.add_argument("--config", default=None, help="det3d config or preset; default: from --ckpt")
    ap.add_argument("--hz", type=float, default=10, help="10: every other sweep (tracker rate); 2: keyframes only")
    ap.add_argument("--cams", default="CAM_FRONT", help="comma-separated, 'all' or 'none'")
    ap.add_argument("--out", default="work_dirs/nusc_eval")
    ap.add_argument("--max-frames", type=int, default=None, help="per scene")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--lidar-stride", type=int, default=1, help="keep every n-th point of /lidar")
    ap.add_argument("--predictor-model", default=None, help="tier-2 model (.pt); default: IMM only")
    ap.add_argument("--no-mcap", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from dog_perception.nuscenes_clip import CAMERAS
    os.makedirs(args.out, exist_ok=True)
    tables = NuscTables(args.dataroot, args.version)
    names = tables.scene_names(args.scenes)[:args.max_scenes]
    if not names:
        raise SystemExit("no scenes match %r in %s" % (args.scenes, args.version))
    cams = () if args.cams == "none" else (CAMERAS if args.cams == "all" else tuple(args.cams.split(",")))
    config = args.config or checkpoint_arch(args.ckpt)
    det = build_detector(config, checkpoint=args.ckpt, device=args.device, xy_range=51.2)
    tcfg = TrackerConfig()
    if args.hz < 5:
        tcfg.tentative_max_age = 1.2 / args.hz       # survive one missed frame at the lower rate
    tracker = MultiObjectTracker(tcfg)
    predictor = IMMPredictor()
    if args.predictor_model:
        from dog_perception.prediction.learned import LearnedPredictor
        predictor = LearnedPredictor(args.predictor_model)
        predictor.horizon, predictor.step = predictor.fc.fut_len * predictor.fc.fut_dt, predictor.fc.fut_dt
    acc = MotAccumulator(CLASSES)
    store = dict(det_pred=[], det_gt=[], pred=[], op=defaultdict(float), mcap=[], frames=0, keyframes=0,
                 snapshots=[], fp_cause=defaultdict(int), fn_cause=defaultdict(int), fn_range=defaultdict(int),
                 fn_pts=defaultdict(int))
    timer = Timer()
    for si, name in enumerate(names):
        scene = NuscScene(tables, name, hz=args.hz, cameras=cams)
        print("scene %s: %d frames (%s)" % (name, len(scene), scene.scene["description"]), flush=True)
        run_scene(scene, si, det, tracker, predictor, acc, store, args, timer)

    det_res = evaluate_detections(store["det_pred"], store["det_gt"], CLASSES, range_bins=RANGE_BINS)
    mot = acc.summary()
    samples = store["pred"]
    pred = {"all objects": prediction_metrics(samples),
            "moving objects (>= 1.5 m in 3 s)": prediction_metrics([s for s in samples if s["moving"]])}
    for c, n in enumerate(CLASSES):
        pred["%s, moving" % n] = prediction_metrics([s for s in samples if s["moving"] and s["cls"] == c])
    figures = []
    for snap in store["snapshots"]:
        path = os.path.join(args.out, "%s_bev.png" % snap["scene"])
        bev_figure(snap, path)
        figures.append(os.path.basename(path))
    meta = dict(version=args.version, scenes=names, hz=args.hz, frames=store["frames"], keyframes=store["keyframes"],
                ckpt=args.ckpt, config=PRESET_CONFIGS.get(config, config), mcap=store["mcap"], figures=figures)
    errors = {k: {"%d|%s" % key: v for key, v in store[k].items()} for k in ("fp_cause", "fn_cause", "fn_range",
                                                                            "fn_pts")}
    text = write_report(os.path.join(args.out, "report.md"), os.path.join(args.out, "report.json"), meta, det_res,
                        dict(store["op"]), mot, pred, timer.summary(), errors)
    print(text)


if __name__ == "__main__":
    main()
