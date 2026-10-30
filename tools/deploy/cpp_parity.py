"""Python vs C++ on a recorded sequence: the whole pipeline, and the C++ results scored like the Python ones.

    # 1. the Python reference: the pipeline as run_sequence.py runs it, plus the detections it used and the
    #    C++ configuration of exactly this run (export_cpp_config.py)
    python tools/deploy/cpp_parity.py python --seq data/seq/<scene> --exported data/deploy/nusc_pp:trt:fp16 \\
        --out <dir>/py.npz --dets <dir>/py_dets.npz --config <dir>/config.yaml
    # 2. C++: tracking and prediction on the same detections, then the whole chain with its own detector
    dogperc_replay --seq data/seq/<scene> --config <dir>/config.yaml --detections <dir>/py_dets.npz \
        --out <dir>/cpp_dets.npz
    dogperc_replay --seq data/seq/<scene> --config <dir>/config.yaml --model-dir data/deploy/nusc_pp --gpu \
        --out <dir>/cpp.npz
    # 3. compare (gates of docs/DEPLOY_ORIN.md section 6), and turn the C++ run into a results.pkl for pipeline_eval.py
    python tools/deploy/cpp_parity.py compare --ref <dir>/py.npz --cpp <dir>/cpp_dets.npz --same-detections \
        --out <dir>/parity_dets.json
    python tools/deploy/cpp_parity.py compare --ref <dir>/py.npz --cpp <dir>/cpp.npz --out <dir>/parity.json
    python tools/deploy/cpp_parity.py to-results --cpp <dir>/cpp.npz --seq data/seq/<scene> \
        --out data/deploy/T28_runs/<scene>/cpp/results.pkl

``--same-detections``: C++ got the Python detections, so the tracks must be the same: identical IDs in every
frame, states within 1e-6, predictions within 1e-4. Without it each side ran its own detector: the
detections are compared like compare_backends.py does, and the tracks by ID.

The online node's path (docs/DEPLOY_ORIN.md section 8): ``python --bag`` replays a rosbag2 in arrival order
through dog_perception/online.py (the Python node's message handling) with the ``--robot`` preset, and
``--messages`` saves those messages as plain arrays for the C++ node path without ROS:

    python tools/deploy/cpp_parity.py python --bag <bag> --robot kygi --exported data/deploy/nusc_pp_dyn \
        --out <dir>/py_bag.npz --dets <dir>/py_bag_dets.npz --config <dir>/kygi.yaml --messages <dir>/msgs.npz
    dogperc_replay --messages <dir>/msgs.npz --config <dir>/kygi.yaml --detections <dir>/py_bag_dets.npz \
        --out <dir>/cpp_msgs.npz        # then compare --same-detections as above
    dogperc_bag_replay --bag <bag> ...  # (deploy/ros2) the same through the ROS messages: identical to cpp_msgs.npz
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dog_perception.detection.boxes import Detections  # noqa: E402

MOTION_NAMES = ("static", "moving", "stopped")


def frame_dets(z, prefix, f):
    p = prefix + "_" if prefix else ""
    a, b = z[p + "offsets"][f], z[p + "offsets"][f + 1]
    return Detections(z[p + "boxes"][a:b], z[p + "velocities"][a:b], z[p + "scores"][a:b], z[p + "labels"][a:b], "det",
                      None, z[p + "subclasses"][a:b])


# ------------------------------------------------------------------ python
def seq_scans(args, pipe):
    """The scans of a recorded sequence, the poses fed to ``pipe`` in time order (run_sequence.py)."""
    from dog_perception.io import SequenceReader

    for ev in SequenceReader(args.seq).events(odom_latency=args.odom_latency):
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            yield ev[1]


def bag_scans(args, inputs, log=None):
    """The scans of a bag in arrival order through the online node's message handling."""
    from message_log import bag_messages

    clouds = 0
    for kind, msg in bag_messages(args.bag, inputs.params):
        if kind == "cloud" and args.max_clouds and clouds >= args.max_clouds:
            break
        if log is not None:
            log.add(kind, msg)
        if kind == "cloud":
            clouds += 1
            yield inputs.scan(msg)
        else:
            getattr(inputs, "on_" + kind)(msg)


def run_python(args):
    from dump_golden import dets_arrays, pipeline_arrays
    from export_cpp_config import config_dict, node_section, to_yaml
    from dog_perception.pipeline import PerceptionPipeline

    if bool(args.seq) == bool(args.bag):
        raise SystemExit("--seq or --bag")
    if args.bag:
        from dog_perception.node_config import node_parameters, preprocess_config
        params = node_parameters(args.robot)
        pcfg = preprocess_config(params)
        source = dict(robot=args.robot, bag=os.path.abspath(args.bag))
    else:
        from dog_perception.io import SequenceReader
        reader = SequenceReader(args.seq)
        pcfg = reader.preprocess_config(reader.meta.get("base_height", 0.45))
        source = dict(seq=os.path.abspath(args.seq))
    det, given = None, None
    if args.exported:
        from dog_perception.detection.exported import ExportedPillarDetector, graph_file
        path, runtime, prec = (args.exported.split(":") + ["trt", "fp16"])[:3]
        pfn = "pfn.fp16.engine" if runtime == "trt" and prec.startswith("int8") and not os.path.exists(
            graph_file(path, "pfn", "trt", prec)) else None
        det = ExportedPillarDetector(path, runtime, prec, pfn=pfn)
    elif args.detections:
        given = np.load(args.detections)
    elif not args.no_detector:
        raise SystemExit("--exported, --detections or --no-detector")
    pipe = PerceptionPipeline(pcfg, det, predictor="imm")
    log = None
    if args.bag:
        from message_log import MessageLog
        from dog_perception.online import OnlineInputs
        inputs = OnlineInputs(params, pipe)
        log = MessageLog() if args.messages else None
        scans, on_scan = bag_scans(args, inputs, log), inputs.on_scan
    else:
        scans, on_scan = seq_scans(args, pipe), pipe.on_scan
    from dog_perception.detection.filters import DetectionFilterConfig, filter_measurements
    rec = dict(stamp=[], T_world_det=[], T_world_body=[], num_raw=[], num_kept=[], points=[], ground_offset=[],
               ground_tilt=[], dets=[], raw=[], tracks=[], preds=[])
    bottoms, counts = [], []
    k = 0
    for scan in scans:
        dets, match = None, False
        if given is not None:          # the other run's frames in order, as dogperc_replay --detections reads them
            match = k < len(given["stamp"]) and abs(given["stamp"][k] - scan.stamp) < 1e-9
            dets = frame_dets(given, "", k) if match else Detections()
        elif det is None:              # --no-detector
            dets = Detections()
        out = on_scan(scan, detections=dets)
        if out is None:
            continue
        k += match
        g = pipe.pre.ground
        for key, val in (("stamp", out.stamp), ("T_world_det", out.frame.T_world_det),
                         ("T_world_body", out.frame.T_world_body),
                         ("num_raw", out.frame.num_raw), ("num_kept", out.frame.num_kept), ("points", out.frame.points),
                         ("ground_offset", g.offset if g is not None else 0.0),
                         ("ground_tilt", np.asarray(g.tilt if g is not None else (0.0, 0.0), np.float64)),
                         ("dets", out.detections), ("tracks", out.tracks), ("preds", out.predictions)):
            rec[key].append(val)
        raw = getattr(det, "last_raw", None) or Detections()
        rec["raw"].append(raw)
        if len(raw):              # what the filters measured, for pipeline_eval.py --detections raw
            fcfg = getattr(det, "filter_cfg", None) or DetectionFilterConfig()
            b, c = filter_measurements(raw, out.frame.points, fcfg)
            bottoms.append(np.asarray(b, np.float64))
            counts.append(np.asarray(c, np.int64))
        if len(rec["stamp"]) % 50 == 0:
            print("%d frames" % len(rec["stamp"]), flush=True)
    a = pipeline_arrays(rec, save_points=args.save_points)
    a.update(dets_arrays("raw", rec["raw"]))
    a["raw_bottom"] = np.concatenate(bottoms) if bottoms else np.zeros(0)
    a["raw_counts"] = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    a["odom_latency"] = np.float64(args.odom_latency)
    np.savez(args.out, **a)
    if args.dets:
        d = dets_arrays("det", rec["dets"])
        np.savez(args.dets, stamp=np.asarray(rec["stamp"]), **{key[4:]: v for key, v in d.items()})
    if args.config:
        node = node_section(args.robot) if args.bag else None
        with open(args.config, "w") as f:
            f.write(to_yaml(config_dict(pcfg, source=source, node=node)) + "\n")
    if log is not None:
        log.save(args.messages)
        print("%d messages (%s) -> %s" % (len(log.order), ", ".join("%d %s" % (n, k) for k, n in log.counts.items()),
                                          args.messages))
    print("python: %d frames, %d tracks -> %s" % (len(rec["stamp"]), len(a["trk_id"]), args.out))


# ------------------------------------------------------------------ compare
def compare(args):
    from compare_backends import KINDS, det_stats, summarize_dets

    ref, cpp = np.load(args.ref), np.load(args.cpp)
    rs, cs = ref["stamp"], cpp["stamp"]
    common = sorted(set(np.round(rs, 6)) & set(np.round(cs, 6)))
    ri = {round(float(s), 6): i for i, s in enumerate(rs)}
    ci = {round(float(s), 6): i for i, s in enumerate(cs)}
    out = dict(frames_ref=int(len(rs)), frames_cpp=int(len(cs)), frames_common=len(common))
    pose = 0.0
    acc = {k: defaultdict(int, score=[], center=[], yaw=[], size=[]) for k in ("final", "raw")}
    ids_same, n_trk, worst = 0, 0, defaultdict(float)
    motion_same, motion_n = 0, 0
    pred_worst = 0.0
    for s in common:
        i, j = ri[s], ci[s]
        pose = max(pose, float(np.abs(ref["T_world_det"][i] - cpp["T_world_det"][j]).max()))
        raws = (frame_dets(ref, "raw", i), frame_dets(cpp, "raw", j)) if "raw_offsets" in ref and "raw_offsets" in cpp \
            else (None, None)
        det_stats(frame_dets(ref, "det", i), frame_dets(cpp, "det", j), 0.5, acc["final"], 0.02, *raws)
        if raws[0] is not None:
            det_stats(raws[0], raws[1], 0.5, acc["raw"])
        ra, rb = ref["trk_offsets"][i], ref["trk_offsets"][i + 1]
        ca, cb = cpp["trk_offsets"][j], cpp["trk_offsets"][j + 1]
        rid, cid = list(ref["trk_id"][ra:rb]), list(cpp["trk_id"][ca:cb])
        ids_same += rid == cid
        for t in set(rid) & set(cid):
            a, b = ra + rid.index(t), ca + cid.index(t)
            n_trk += 1
            for key in ("trk_position", "trk_velocity", "trk_size", "trk_score", "trk_covariance", "trk_age"):
                worst[key] = max(worst[key], float(np.abs(np.asarray(ref[key][a]) - np.asarray(cpp[key][b])).max()))
            dyaw = float((ref["trk_yaw"][a] - cpp["trk_yaw"][b] + np.pi) % (2 * np.pi) - np.pi)
            worst["trk_yaw"] = max(worst["trk_yaw"], abs(dyaw))
            motion_same += int(ref["trk_motion"][a] == cpp["trk_motion"][b])
            motion_n += 1
        pa, pb = ref["pred_offsets"][i], ref["pred_offsets"][i + 1]
        qa, qb = cpp["pred_offsets"][j], cpp["pred_offsets"][j + 1]
        pid, qid = list(ref["pred_track_id"][pa:pb]), list(cpp["pred_track_id"][qa:qb])
        for t in set(pid) & set(qid):
            a, b = pa + pid.index(t), qa + qid.index(t)
            if ref["pred_num_modes"][a] == cpp["pred_num_modes"][b]:
                k = int(ref["pred_num_modes"][a])
                pred_worst = max(pred_worst, float(np.abs(ref["pred_modes"][a, :k] - cpp["pred_modes"][b, :k]).max()),
                                 float(np.abs(ref["pred_probs"][a, :k] - cpp["pred_probs"][b, :k]).max()))
    out.update(pose_max=pose, detections=summarize_dets(acc["final"]),
               raw_detections=summarize_dets(acc["raw"]) if acc["raw"]["n_ref"] else None,
               track_ids_identical_frames=ids_same, track_pairs=n_trk, track_max_diff=dict(worst),
               motion_agreement=motion_same / max(motion_n, 1), prediction_max_diff=pred_worst)
    trk = max(worst.values()) if worst else 0.0
    # --subset: the C++ run covers the first frames only (the Orin copy of a recording)
    frames_ok = len(common) == len(cs) > 0 if args.subset else len(common) == len(rs) == len(cs)
    if args.same_detections:
        d = out["detections"]
        same = d["score_max"] == 0.0 and d["matched"] == d["n_ref"] == d["n_got"]
        gates = dict(frames=frames_ok, detections_identical=same,
                     track_ids=ids_same == len(common), track_states=trk <= 1e-6, predictions=pred_worst <= 1e-4)
    else:
        d = out["detections"]
        gates = dict(frames=frames_ok, reported_match_tol=d["reported_match_tol"] >= 0.99,
                     score_p99=d["score_p99"] <= 0.02, center_p99=d["center_p99"] <= 0.1)
        if out["raw_detections"]:
            gates["raw_match_tol"] = out["raw_detections"]["reported_match_tol"] >= 0.99
    out["gates"] = dict(gates, passed=all(gates.values()))
    d = out["detections"]
    print("| frames ref / cpp / common | pose max | det match w/o flips | missed (%s) | extra | track IDs same | "
          "track max diff | motion agree | pred max diff | gates |" % "/".join(KINDS))
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("| %d / %d / %d | %.1e | %.4f | %d (%s) | %d | %d / %d | %.1e | %.4f | %.1e | %s |" % (
        len(rs), len(cs), len(common), pose, d["reported_match_tol"], d["reported_missed"],
        "/".join(str(d["missed_by"][k]) for k in KINDS), d["reported_extra"], ids_same, len(common), trk,
        out["motion_agreement"], pred_worst, "pass" if out["gates"]["passed"] else "FAIL"))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


# ------------------------------------------------------------------ to-results
def to_results(args):
    from dog_perception.io import SequenceReader, pose_bad
    from dog_perception.prediction.kinematic import Prediction
    from dog_perception.tracking.tracker import TrackState

    z = np.load(args.cpp)
    meta = SequenceReader(args.seq).meta if args.seq else {}
    timing_keys = [k for k in z.files if k.startswith("timing_")]
    rows = []
    for f, stamp in enumerate(z["stamp"]):
        a, b = z["trk_offsets"][f], z["trk_offsets"][f + 1]
        tracks = []
        for r in range(a, b):
            meas = z["trk_measurement"][r]
            mp = z["trk_mode_probs"][r]
            tracks.append(TrackState(
                track_id=int(z["trk_id"][r]), label=int(z["trk_label"][r]), stamp=float(stamp),
                position=z["trk_position"][r].copy(), velocity=z["trk_velocity"][r].copy(), yaw=float(z["trk_yaw"][r]),
                size=z["trk_size"][r].copy(), score=float(z["trk_score"][r]), covariance=z["trk_covariance"][r].copy(),
                mode_probs=mp[np.isfinite(mp)], age=float(z["trk_age"][r]), hits=int(z["trk_hits"][r]),
                coasting=bool(z["trk_coasting"][r]), history=np.zeros((0, 6)),
                motion=MOTION_NAMES[int(z["trk_motion"][r])],
                time_since_moving=float(z["trk_time_since_moving"][r]),
                speed_displacement=float(z["trk_speed_displacement"][r]), speed_head=float(z["trk_speed_head"][r]),
                measurement=None if np.isnan(meas[0]) else meas.copy()))
        a, b = z["pred_offsets"][f], z["pred_offsets"][f + 1]
        preds = []
        for r in range(a, b):
            k = int(z["pred_num_modes"][r])
            preds.append(Prediction(int(z["pred_track_id"][r]), int(z["pred_label"][r]), float(stamp),
                                    z["pred_times"].copy(), z["pred_modes"][r, :k].copy(),
                                    z["pred_probs"][r, :k].copy(), z["pred_covs"][r, :k].copy(), "imm"))
        dets = frame_dets(z, "det", f)
        dets.stamp = float(stamp)
        row = dict(stamp=float(stamp), T_world_det=z["T_world_det"][f], T_world_body=z["T_world_body"][f],
                   pose_bad=pose_bad(meta, float(stamp)), detections=dets, tracks=tracks, predictions=preds,
                   timings={k[7:]: float(z[k][f]) for k in timing_keys})
        if "raw_offsets" in z.files and z["raw_offsets"][-1] > 0:
            raw = frame_dets(z, "raw", f)
            raw.stamp = float(stamp)
            a, b = z["raw_offsets"][f], z["raw_offsets"][f + 1]
            row.update(raw_detections=raw, raw_measured=(z["raw_bottom"][a:b].copy(), z["raw_counts"][a:b].copy()))
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(rows, f)
    print("%d frames -> %s" % (len(rows), args.out))


# ------------------------------------------------------------------ timings
def timings(args):
    """Per-stage latency of dogperc_replay runs (median, p90, p99, max over the frames, the first frames left
    out as warm-up); with several runs also ``all``, their frames pooled."""
    print("| run | stage | median ms | p90 | p99 | max |")
    print("|---|---|---|---|---|---|")
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.runs]
    if len(set(names)) < len(names):          # data/deploy/T28/<scene>/cpp.npz: name them by their directory
        names = [os.path.join(os.path.basename(os.path.dirname(os.path.abspath(p))), n)
                 for p, n in zip(args.runs, names)]
    out, pooled = {}, defaultdict(list)

    def row(name, k, v):
        q = [float(np.median(v)), float(np.percentile(v, 90)), float(np.percentile(v, 99)), float(v.max())]
        out.setdefault(name, {})[k] = q
        print("| %s | %s | %.2f | %.2f | %.2f | %.2f |" % (name, k, *q))

    for path, name in zip(args.runs, names):
        z = np.load(path)
        for k in sorted(f for f in z.files if f.startswith("timing_")):
            v = z[k][args.skip:]
            v = v[np.isfinite(v)]
            if len(v):
                row(name, k[7:], v)
                pooled[k[7:]].append(v)
    if len(args.runs) > 1:
        for k in sorted(pooled):
            row("all", k, np.concatenate(pooled[k]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("python", help="the Python reference run")
    p.add_argument("--seq", default=None, help="a recorded sequence (data/seq/<scene>)")
    p.add_argument("--bag", default=None, help="instead: a rosbag2, replayed in arrival order like the online node")
    p.add_argument("--robot", default="kygi", help="--bag: the node_config.py preset")
    p.add_argument("--messages", default=None, help="--bag: also save the messages for dogperc_replay --messages")
    p.add_argument("--max-clouds", type=int, default=0, help="--bag: stop after this many clouds")
    p.add_argument("--no-detector", action="store_true", help="no detections (preprocessing and poses only)")
    p.add_argument("--exported", default=None, help="dir[:runtime[:precision]] (exported.py), default runtime trt fp16")
    p.add_argument("--detections", default=None, help="instead of a detector: the detections of another run (dets npz)")
    p.add_argument("--odom-latency", type=float, default=0.0)
    p.add_argument("--save-points", action="store_true", help="also the det-frame clouds (large)")
    p.add_argument("--out", required=True)
    p.add_argument("--dets", default=None, help="its detections, for dogperc_replay --detections")
    p.add_argument("--config", default=None, help="the C++ configuration of this run")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cpp", required=True)
    c.add_argument("--same-detections", action="store_true", help="C++ ran on the Python detections")
    c.add_argument("--subset", action="store_true", help="the C++ run has only the first frames of the reference")
    c.add_argument("--out", default=None)
    r = sub.add_parser("to-results", help="dogperc_replay output -> results.pkl for pipeline_eval.py")
    r.add_argument("--cpp", required=True)
    r.add_argument("--seq", default=None, help="for pose_bad")
    r.add_argument("--out", required=True)
    t = sub.add_parser("timings", help="per-stage latency of dogperc_replay outputs")
    t.add_argument("runs", nargs="+")
    t.add_argument("--skip", type=int, default=5, help="warm-up frames left out")
    t.add_argument("--out", default=None)
    args = ap.parse_args()
    {"python": run_python, "compare": compare, "to-results": to_results, "timings": timings}[args.cmd](args)


if __name__ == "__main__":
    main()
