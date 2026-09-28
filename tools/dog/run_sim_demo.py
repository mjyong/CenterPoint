"""End-to-end four-stage demo on the built-in XT32 / legged-robot simulator.

Reports, per stage:
  1. preprocess : wall flatness with / without deskew, ground flatness in the
                  det frame vs. the pitching body frame, self-hit removal
  2. detection  : center-distance AP vs. simulator ground truth
  3. tracking   : MOTA-style counts, ID switches, position / velocity RMSE
  4. prediction : minADE / minFDE / MR against the *true* future of each object,
                  tier 1 (IMM) vs. tier 2 (learned, if --model is given)

Detectors: ``oracle`` (GT + noise, no weights needed), ``pillar`` / ``voxel``
(real det3d CenterPoint; pass --ckpt-*, otherwise random weights -> only the
plumbing and latency are meaningful).

    python tools/dog/run_sim_demo.py --frames 80 --out work_dirs/sim_demo
    python tools/dog/train_predictor.py --sim 12 --out work_dirs/pred/model.pt
    python tools/dog/run_sim_demo.py --model work_dirs/pred/model.pt --plot 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection import CLASSES, OracleDetector, OracleNoise  # noqa: E402
from dog_perception.detection.evaluate import evaluate_detections, format_table  # noqa: E402
from dog_perception.geometry import inv_T, transform_points  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.prediction.metrics import displacement_metrics  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig, det_frame_from_body  # noqa: E402
from dog_perception.preprocess.self_filter import points_in_boxes_aabb  # noqa: E402
from dog_perception.sim import GaitModel, make_scenario  # noqa: E402

WALL_Y = 13.85   # inner face of the simulated street walls


class SimPoseFeeder:
    """Streams 400 Hz gyro and 10 Hz odometry (arriving ``latency`` late) into the pipeline."""

    def __init__(self, sim, pipe, t_end, latency=0.03):
        imu_t, imu_w = sim.imu(0.0, t_end)
        ev = [(t, 0, w) for t, w in zip(imu_t, imu_w)]
        ev += [(to + latency, 1, to) for to in np.arange(0.0, t_end, 0.1)]
        ev.sort(key=lambda e: (e[0], e[1]))
        self.ev, self.i, self.sim, self.pipe = ev, 0, sim, pipe

    def advance_to(self, t):
        while self.i < len(self.ev) and self.ev[self.i][0] <= t + 1e-9:
            te, kind, val = self.ev[self.i]
            if kind == 0:
                self.pipe.on_imu(te, val)
            else:
                R, p, v = self.sim.odometry(val, pos_noise=0.01, rot_noise=0.002)
                self.pipe.on_odometry(val, R, p, v)
            self.i += 1


def gt_to_det(gt, T_det_world, min_points=5, max_range=40.0):
    R = T_det_world[:3, :3]
    dyaw = np.arctan2(R[1, 0], R[0, 0])
    boxes, labels, vels, ids = [], [], [], []
    for g in gt:
        if g["num_points"] < min_points:
            continue
        c = R @ g["center"] + T_det_world[:3, 3]
        if np.hypot(c[0], c[1]) > max_range:
            continue
        boxes.append([*c, *g["size"], g["yaw"] + dyaw])
        labels.append(CLASSES.index(g["label"]))
        vels.append(R[:2, :2] @ g["velocity"])
        ids.append(g["id"])
    return dict(boxes=np.asarray(boxes).reshape(-1, 7), labels=np.asarray(labels, dtype=np.int64),
                velocities=np.asarray(vels).reshape(-1, 2), ids=np.asarray(ids, dtype=np.int64))


def build_detector(kind, args):
    if kind == "oracle":
        return None
    from dog_perception.detection.centerpoint import CenterPointDetector
    cfg = {"pillar": args.config_pillar, "voxel": args.config_voxel}[kind]
    ckpt = {"pillar": args.ckpt_pillar, "voxel": args.ckpt_voxel}[kind]
    if ckpt is None:
        print("[%s] no checkpoint given: random weights, only latency / plumbing are meaningful" % kind)
    return CenterPointDetector(cfg, ckpt, device=args.device)


def preprocess_report(sim, pipe, out, scan, frame):
    """Deskew / gravity / self-filter diagnostics for one scan."""
    cfg = pipe.pre.cfg
    cur = frame.points[frame.points[:, 4] == 0, :3]
    world = transform_points(frame.T_world_det, cur.astype(np.float64))
    wall = (np.abs(np.abs(world[:, 1]) - WALL_Y) < 0.6) & (world[:, 2] > 0.3) & (world[:, 2] < 2.5)
    out["wall_rms_deskew"].append(np.sqrt(np.mean((np.abs(world[wall, 1]) - WALL_Y) ** 2)))

    # same scan without deskew: every point transformed with the scan-end pose
    keep = pipe.pre.self_filter(scan.xyz)
    raw_w = transform_points(frame.T_world_body @ cfg.T_body_lidar, scan.xyz[keep])
    wall_r = (np.abs(np.abs(raw_w[:, 1]) - WALL_Y) < 0.6) & (raw_w[:, 2] > 0.3) & (raw_w[:, 2] < 2.5)
    out["wall_rms_raw"].append(np.sqrt(np.mean((np.abs(raw_w[wall_r, 1]) - WALL_Y) ** 2)))

    # ground flatness: gravity-aligned det frame vs. body frame (pitching with the gait)
    r = np.hypot(cur[:, 0], cur[:, 1])
    g = (r > 3) & (r < 15) & (np.abs(cur[:, 2]) < 0.3)
    out["ground_z_std_det"].append(float(np.std(cur[g, 2])))
    body = transform_points(inv_T(frame.T_world_body), world)
    out["ground_z_std_body"].append(float(np.std(body[g, 2])))

    # self hits: points inside the payload box before / after filtering
    body_raw = transform_points(cfg.T_body_lidar, scan.xyz)
    in_box = points_in_boxes_aabb(body_raw, [sim.payload_box])
    out["self_hits_raw"].append(int(in_box.sum()))
    out["self_hits_kept"].append(int((in_box & keep).sum()))


def track_metrics(tracks, gt_w, acc, last_assign, match_dist=1.0):
    live = [s for s in tracks if not s.coasting]
    acc["gt"] += len(gt_w["ids"])
    if len(live) == 0 or len(gt_w["ids"]) == 0:
        acc["fn"] += len(gt_w["ids"])
        acc["fp"] += len(live)
        return {}
    tp = np.array([s.position[:2] for s in live])
    cost = np.linalg.norm(gt_w["boxes"][:, None, :2] - tp[None], axis=2)
    same = gt_w["labels"][:, None] == np.array([s.label for s in live])[None]
    cost = np.where(same, cost, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] <= match_dist
    r, c = r[ok], c[ok]
    acc["tp"] += len(r)
    acc["fn"] += len(gt_w["ids"]) - len(r)
    acc["fp"] += len(live) - len(r)
    pairs = {}
    for i, j in zip(r, c):
        gid, tid = int(gt_w["ids"][i]), live[j].track_id
        if gid in last_assign and last_assign[gid] != tid:
            acc["idsw"] += 1
        last_assign[gid] = tid
        acc["pos_se"].append(cost[i, j] ** 2)
        acc["vel_se"].append(float(np.sum((live[j].velocity - gt_w["velocities"][i]) ** 2)))
        pairs[tid] = gid
    return pairs


def run(kind, args, sim, learned=None):
    pcfg = PreprocessConfig(T_body_lidar=sim.T_bl, det_frame=DetFrameConfig(base_height=sim.robot.g.base_height))
    pipe = PerceptionPipeline(pcfg, build_detector(kind, args))
    oracle = OracleDetector(OracleNoise(), seed=args.seed + 7)
    feeder = SimPoseFeeder(sim, pipe, args.frames * 0.1 + 0.5)
    objs = {o.id: o for o in sim.objects if not o.static}

    pre_rep = {k: [] for k in ("wall_rms_deskew", "wall_rms_raw", "ground_z_std_det", "ground_z_std_body",
                               "self_hits_raw", "self_hits_kept")}
    det_pred, det_gt = [], []
    acc = dict(gt=0, tp=0, fn=0, fp=0, idsw=0, pos_se=[], vel_se=[])
    last_assign = {}
    pred_acc = {"imm": ([], [], [])}
    if learned is not None:
        pred_acc["learned"] = ([], [], [])
    timings = []

    for k in range(args.frames):
        scan, info = sim.scan(k)
        feeder.advance_to(scan.stamp)
        dets = None
        if kind == "oracle":
            T_wd = det_frame_from_body(pipe.poses.pose_at(scan.stamp), sim.robot.g.base_height)
            dets = oracle(info["gt"], inv_T(T_wd), scan.stamp)
        out = pipe.on_scan(scan, detections=dets)
        if out is None:
            continue
        if k >= 1:
            timings.append(out.timings)
        if k % 5 == 0:
            preprocess_report(sim, pipe, pre_rep, scan, out.frame)

        gt_d = gt_to_det(info["gt"], inv_T(out.frame.T_world_det))
        d = out.detections
        det_pred.append(dict(boxes=d.boxes, labels=d.labels, scores=d.scores, velocities=d.velocities))
        det_gt.append(gt_d)

        gt_w = gt_to_det(info["gt"], np.eye(4))
        pairs = track_metrics(out.tracks, gt_w, acc, last_assign)

        preds = {"imm": out.predictions}
        if learned is not None:
            preds["learned"] = learned(pipe.tracker, np.asarray(pipe._ego))
        for name, plist in preds.items():
            for p in plist:
                if p.track_id not in pairs:
                    continue
                o = objs[pairs[p.track_id]]
                fut, _, _ = o.state(out.stamp + p.times)
                K = 6
                m = np.repeat(p.modes[np.argmax(p.probs)][None], K, 0)
                pr = np.zeros(K)
                n = min(K, len(p.modes))
                order = np.argsort(-p.probs)[:n]
                m[:n], pr[:n] = p.modes[order], p.probs[order]
                pred_acc[name][0].append(m)
                pred_acc[name][1].append(pr)
                pred_acc[name][2].append(fut[:, :2])

        if args.plot and k in set(np.linspace(args.frames // 3, args.frames - 1, args.plot).astype(int)):
            plot_bev(out, info["gt"], os.path.join(args.out, "bev_%s_%03d.png" % (kind, k)))

    res = {"preprocess": {k: float(np.mean(v)) for k, v in pre_rep.items()}}
    res["detection"] = evaluate_detections(det_pred, det_gt, CLASSES)
    res["tracking"] = dict(
        MOTA=1 - (acc["fn"] + acc["fp"] + acc["idsw"]) / max(acc["gt"], 1),
        recall=acc["tp"] / max(acc["gt"], 1), FP=acc["fp"], FN=acc["fn"], IDSW=acc["idsw"], GT=acc["gt"],
        pos_rmse=float(np.sqrt(np.mean(acc["pos_se"]))) if acc["pos_se"] else float("nan"),
        vel_rmse=float(np.sqrt(np.mean(acc["vel_se"]))) if acc["vel_se"] else float("nan"))
    res["prediction"] = {}
    for name, (m, pr, g) in pred_acc.items():
        if m:
            res["prediction"][name] = displacement_metrics(np.stack(m), np.stack(pr), np.stack(g))
    keys = timings[0].keys() if timings else []
    res["latency_ms"] = {k: float(np.mean([t[k] for t in timings])) for k in keys}
    return res


def plot_bev(out, gt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    T_dw = inv_T(out.frame.T_world_det)
    pts = out.frame.points
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c=pts[:, 4], cmap="Greys_r", vmin=-0.2, vmax=0.6)

    def box_poly(x, y, l, w, yaw, **kw):
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2
        xy = corners @ np.array([[c, s], [-s, c]]) + [x, y]
        ax.add_patch(Polygon(xy, closed=True, fill=False, **kw))

    R, dyaw = T_dw[:3, :3], np.arctan2(T_dw[1, 0], T_dw[0, 0])
    for g in gt:
        c = R @ g["center"] + T_dw[:3, 3]
        box_poly(c[0], c[1], g["size"][0], g["size"][1], g["yaw"] + dyaw, ec="lime", lw=1.2)
    local_preds = [p.transform(T_dw, "det") for p in out.predictions]
    for s in out.tracks:
        p = R @ s.position + T_dw[:3, 3]
        box_poly(p[0], p[1], s.size[0], s.size[1], s.yaw + dyaw, ec="tab:blue", lw=1.5,
                 ls="--" if s.coasting else "-")
        ax.text(p[0], p[1] + 0.8, str(s.track_id), color="tab:blue", fontsize=7)
    for pr in local_preds:
        for m, w in zip(pr.modes, pr.probs):
            ax.plot(m[:, 0], m[:, 1], "-", color="tab:red", alpha=float(0.25 + 0.75 * w), lw=1.2)
    ax.plot(0, 0, "k^", ms=10)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.set_title("t=%.1fs  green: GT  blue: tracks (dashed=coasting)  red: 3 s predictions" % out.stamp)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detectors", default="oracle", help="comma list of oracle,pillar,voxel")
    ap.add_argument("--config-pillar", default="configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py")
    ap.add_argument("--config-voxel", default="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py")
    ap.add_argument("--ckpt-pillar", default=None)
    ap.add_argument("--ckpt-voxel", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None, help="tier-2 predictor checkpoint (train_predictor.py)")
    ap.add_argument("--plot", type=int, default=0, help="number of BEV snapshots to save")
    ap.add_argument("--out", default="work_dirs/sim_demo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gait = GaitModel(speed=1.2, yaw_rate=0.15, pitch_amp_deg=6.0, roll_amp_deg=3.0)
    learned = None
    if args.model:
        from dog_perception.prediction.learned import LearnedPredictor
        learned = LearnedPredictor(args.model)

    results = {}
    for kind in args.detectors.split(","):
        sim = make_scenario(seed=args.seed, duration=args.frames * 0.1 + 4.0, num_pedestrians=10,
                            num_cyclists=3, num_vehicles=3, gait=gait)
        t0 = time.time()
        results[kind] = run(kind, args, sim, learned)
        print("[%s] done in %.1fs" % (kind, time.time() - t0))

    for kind, r in results.items():
        p = r["preprocess"]
        print("\n=== %s ===" % kind)
        print("preprocess: wall RMS %.3f m (raw) -> %.3f m (deskewed); ground z std %.3f m (body) -> %.3f m (det); "
              "self hits %.0f -> %.0f per scan"
              % (p["wall_rms_raw"], p["wall_rms_deskew"], p["ground_z_std_body"], p["ground_z_std_det"],
                 p["self_hits_raw"], p["self_hits_kept"]))
        print(format_table({kind: r["detection"]}, CLASSES))
        t = r["tracking"]
        print("tracking: MOTA %.3f recall %.3f IDSW %d FP %d FN %d pos RMSE %.3f m vel RMSE %.3f m/s"
              % (t["MOTA"], t["recall"], t["IDSW"], t["FP"], t["FN"], t["pos_rmse"], t["vel_rmse"]))
        for name, m in r["prediction"].items():
            print("prediction[%s]: minADE %.3f minFDE %.3f ADE1 %.3f FDE1 %.3f MR@2m %.3f (n=%d)"
                  % (name, m["minADE"], m["minFDE"], m["ADE1"], m["FDE1"], m["MR"], m["num"]))
        print("latency (ms): " + ", ".join("%s %.1f" % kv for kv in r["latency_ms"].items()))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nsaved", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
