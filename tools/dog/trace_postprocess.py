"""Trace CenterPoint post-processing on one recorded frame, step by step.

From the 6 x 102 x 102 head maps to the detector output: per task the
candidates left after the score threshold, the range check, circle NMS and
the 83-box cap; then the 10 -> 3 class mapping, the merged-class NMS and the
three geometric filters. The re-implementation is checked box for box
against det3d's ``CenterHead.predict`` and the detector's own output. It also
follows the strongest candidate of one task through every step and prints a
few boxes removed by each filter. See docs/DETECTION_POSTPROCESS.md.

    python tools/dog/trace_postprocess.py --bag samples/rosbag2_2026_09_21-16_41_27 --frame 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="samples/rosbag2_2026_09_21-16_41_27")
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--ckpt", default="work_dirs/pretrained/nusc_pp.pth")
    ap.add_argument("--config", default="pillar")
    ap.add_argument("--task", type=int, default=0, help="task whose strongest box is followed (0 = car)")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    import torch
    from det3d.core.utils.circle_nms_jit import circle_nms as det3d_circle_nms

    from dog_perception.bag_replay import replay_kygi
    from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS, build_detector
    from dog_perception.detection.boxes import CLASSES, Detections, classwise_circle_nms, det3d_to_standard, \
        score_threshold_array
    from dog_perception.detection.filters import cross_class_suppress, local_ground_heights, points_in_box_counts

    det = build_detector(args.config, checkpoint=args.ckpt, device="cpu")
    fr = replay_kygi(args.bag, max_scans=args.frame + 1)[args.frame]
    pts_D = fr.points
    net, tc = det.net, det.net.test_cfg
    tasks = [t["class_names"] for t in det.cfg.model.bbox_head.tasks]
    names = [n for t in tasks for n in t]
    print("frame %d: %d points in the det frame (5 sweeps)" % (args.frame, len(pts_D)))
    print("score_threshold %.2f | post_center_limit_range %s | grid %d x %d cells of %.1f m (stride %d) | "
          "circle NMS min_radius (squared m) %s -> radius %s m | cap %d per task"
          % (tc.score_threshold, list(tc.post_center_limit_range), 102, 102, tc.out_size_factor * tc.voxel_size[0],
             tc.out_size_factor, list(tc.min_radius), [round(float(np.sqrt(r)), 2) for r in tc.min_radius],
             tc.nms.nms_post_max_size))

    # ---- network up to the head maps
    ex = det.build_example(det.frame.points_to_model(det._crop(pts_D)))
    with torch.no_grad():
        x = net.extract_feat(dict(features=ex["voxels"], num_voxels=ex["num_points"], coors=ex["coordinates"],
                                  batch_size=len(ex["num_voxels"]), input_shape=ex["shape"][0]))
        preds, _ = net.bbox_head(x)
        ref = net.bbox_head.predict(ex, [{k: v.clone() for k, v in p.items()} for p in preds], tc)[0]

    # ---- predict, re-implemented with counts
    lim = torch.tensor(tc.post_center_limit_range)
    boxes, scores, labels, offset, kept = [], [], [], 0, {}
    print("\n| task | classes | candidates | score > %.1f | in range | after circle NMS | after cap | per class |"
          % tc.score_threshold)
    print("|---|---|---|---|---|---|---|---|")
    for t, p in enumerate(preds):
        hm = torch.sigmoid(p["hm"][0]).permute(1, 2, 0)
        H, W, C = hm.shape
        reg = p["reg"][0].permute(1, 2, 0).reshape(-1, 2)
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        bx = (xs.reshape(-1, 1) + reg[:, :1]) * tc.out_size_factor * tc.voxel_size[0] + tc.pc_range[0]
        by = (ys.reshape(-1, 1) + reg[:, 1:]) * tc.out_size_factor * tc.voxel_size[1] + tc.pc_range[1]
        box = torch.cat([bx, by, p["height"][0].permute(1, 2, 0).reshape(-1, 1),
                         torch.exp(p["dim"][0]).permute(1, 2, 0).reshape(-1, 3),
                         p["vel"][0].permute(1, 2, 0).reshape(-1, 2),
                         torch.atan2(p["rot"][0, 0], p["rot"][0, 1]).reshape(-1, 1)], 1)
        sc, lb = hm.reshape(-1, C).max(1)
        m_score = sc > tc.score_threshold
        m_range = (box[:, :3] >= lim[:3]).all(1) & (box[:, :3] <= lim[3:]).all(1)
        idx = torch.nonzero(m_score & m_range)[:, 0]
        keep = np.array(det3d_circle_nms(torch.cat([box[idx, :2], sc[idx, None]], 1).numpy(), thresh=tc.min_radius[t]))
        sel = idx[torch.from_numpy(keep[:tc.nms.nms_post_max_size]).long()] if len(keep) else idx[:0]
        kept[t] = (sc.numpy(), idx.numpy(), sel.numpy(), box.numpy())
        print("| %d | %s | %d | %d | %d | %d | %d | %s |" % (
            t, " + ".join(tasks[t]), len(sc), int(m_score.sum()), len(idx), len(keep), len(sel),
            ", ".join("%s %d" % (tasks[t][c], int((lb[sel] == c).sum())) for c in range(C))))
        boxes.append(box[sel])
        scores.append(sc[sel])
        labels.append(lb[sel] + offset)
        offset += C
    box9, sc, lb = torch.cat(boxes).numpy(), torch.cat(scores).numpy(), torch.cat(labels).numpy()
    same = len(ref["scores"]) == len(sc) and np.allclose(np.sort(ref["scores"].numpy()), np.sort(sc), atol=1e-6)
    print("merged: box3d_lidar %s, scores %s, label_preds %s | identical to det3d CenterHead.predict: %s"
          % (box9.shape, sc.shape, lb.shape, same))

    # ---- follow the strongest box of one task
    s_all, idx, sel, box_all = kept[args.task]
    if len(sel):
        best = sel[np.argmax(s_all[sel])]
        r, c = divmod(int(best), 102)
        p = {k: v[0, :, r, c].numpy() for k, v in preds[args.task].items()}
        print("\nstrongest box of task %d (%s): pixel row %d, col %d" % (args.task, "+".join(tasks[args.task]), r, c))
        print("  hm logits %s -> sigmoid %s" % (np.round(p["hm"], 3), np.round(1 / (1 + np.exp(-p["hm"])), 3)))
        print("  reg %s, height %.3f, dim exp(%s) = %s, rot (sin, cos) %s -> r %.3f, vel %s"
              % (np.round(p["reg"], 3), p["height"][0], np.round(p["dim"], 3), np.round(np.exp(p["dim"]), 3),
                 np.round(p["rot"], 3), np.arctan2(p["rot"][0], p["rot"][1]), np.round(p["vel"], 3)))
        b = box_all[best]
        print("  x = (%d + %.3f) * %.1f %+.1f = %.2f m, y = (%d + %.3f) * %.1f %+.1f = %.2f m (M frame)"
              % (c, p["reg"][0], tc.out_size_factor * tc.voxel_size[0], tc.pc_range[0], b[0],
                 r, p["reg"][1], tc.out_size_factor * tc.voxel_size[1], tc.pc_range[1], b[1]))
        std, v = det3d_to_standard(b[None])
        sD, vD = det.frame.boxes_from_model(std, v)
        print("  box9 %s -> standard (M) %s -> D frame %s, vel %s" % (np.round(b, 3), np.round(std[0], 3),
                                                                    np.round(sD[0], 3), np.round(vD[0], 3)))
        sup = [(int(i) // 102 - r, int(i) % 102 - c, s_all[i], np.hypot(*(box_all[i, :2] - b[:2]))) for i in idx
               if i != best and np.hypot(*(box_all[i, :2] - b[:2])) <= np.sqrt(tc.min_radius[args.task])]
        print("  %d more candidates of this task within the NMS radius, suppressed:" % len(sup))
        for d in sorted(sup, key=lambda d: -d[2])[:args.examples]:
            print("    pixel offset (%+d, %+d) score %.3f, %.2f m away" % d)

    # ---- postprocess
    lab3 = det.label_map[lb]
    keep = lab3 >= 0
    std, v = det3d_to_standard(box9[keep])
    std, v = det.frame.boxes_from_model(std, v)
    pre = Detections(std, v, sc[keep].astype(float), lab3[keep], "det")
    post = classwise_circle_nms(pre, det.merged_nms)
    print("\npostprocess: %d -> %d after dropping %s; merged NMS:" % (len(lb), int(keep.sum()), ", ".join(
        "%s %d" % (names[i], int((lb == i).sum())) for i in range(len(names)) if det.label_map[i] < 0)))
    for c, n in enumerate(CLASSES):
        print("  %-10s %3d -> %3d (radius %s m)" % (n, (pre.labels == c).sum(), (post.labels == c).sum(),
                                                   det.merged_nms.get(n)))
    print("  -> %d boxes (D frame, standard [x, y, z, l, w, h, yaw])" % len(post))

    # ---- filters
    cfg = det.filter_cfg
    cur = pts_D[pts_D[:, -1] <= 1e-6, :3]
    g = local_ground_heights(cur, post.boxes, cfg.ground_cell, cfg.ground_ring, cfg.ground_percentile,
                             cfg.ground_min_cells)
    ground = np.where(np.isfinite(g), g, 0.0)
    bottom = post.boxes[:, 2] - post.boxes[:, 5] / 2 - ground
    ok_g = (bottom <= cfg.max_float) & (bottom >= -cfg.max_sink)
    need = np.array([cfg.min_points.get(CLASSES[l], 0) for l in post.labels])
    cnt = points_in_box_counts(cur, post.boxes, cfg.box_margin)
    ok_p = ok_g & (cnt >= need)
    d2 = post.select(np.nonzero(ok_p)[0])
    ok_x = cross_class_suppress(d2, cfg.cross_class_margin)
    final = d2.select(np.nonzero(ok_x)[0])
    print("\nfilter_detections: %d -> ground support %d -> min points %d -> cross-class %d"
          % (len(post), ok_g.sum(), ok_p.sum(), len(final)))
    for i in np.nonzero(~ok_g)[0][:args.examples]:
        bb = post.boxes[i]
        lg = "%.2f" % g[i] if np.isfinite(g[i]) else "n/a (no ground cells nearby, z = 0 used)"
        print("  ground  : %s %.2f at (%.1f, %.1f), bottom %.2f, local ground %s -> %.2f m off"
              % (CLASSES[post.labels[i]], post.scores[i], bb[0], bb[1], bb[2] - bb[5] / 2, lg, bottom[i]))
    for i in np.nonzero(ok_g & ~ok_p)[0][:args.examples]:
        bb = post.boxes[i]
        print("  points  : %s %.2f at (%.1f, %.1f), %.1f m away, %d points < %d"
              % (CLASSES[post.labels[i]], post.scores[i], bb[0], bb[1], np.hypot(bb[0], bb[1]), cnt[i], need[i]))
    for i in np.nonzero(~ok_x)[0][:args.examples]:
        print("  class   : %s %.2f at (%.1f, %.1f) inside a higher-scoring box of another class"
              % (CLASSES[d2.labels[i]], d2.scores[i], d2.boxes[i, 0], d2.boxes[i, 1]))
    out = det(pts_D, fr.stamp)
    thr = score_threshold_array(final.labels, DEFAULT_SCORE_THRESHOLDS)
    print("\ndetector output: %d boxes (same as the trace: %s); %d reach the report thresholds %s, %d stay for the "
          "tracker's low-score stage" % (len(out), len(out) == len(final), int((final.scores >= thr).sum()),
                                         DEFAULT_SCORE_THRESHOLDS, int((final.scores < thr).sum())))


if __name__ == "__main__":
    main()
