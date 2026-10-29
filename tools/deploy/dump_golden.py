"""Golden data for the C++ parity tests (deploy/cpp/tests): every stage run in Python on the same inputs.

    python tools/deploy/dump_golden.py --out deploy/cpp/build/golden      # CTest runs this first

Needs numpy and scipy only (no det3d, no PyTorch, no model): the detector stages run on a synthetic
model_meta.json with the deployed nuScenes Pillar's head layout and synthetic head maps (peaks planted on
the simulated objects); the network itself is checked by compare_backends.py and the TensorRT parity test.

Writes into --out:
    seq/                a simulated recording in the sequence format (io.py), auto ground / level on
    config.yaml         its parameters (export_cpp_config.py), all three classes tracked
    default.yaml        the dataclass defaults (the C++ defaults must equal them)
    model/model_meta.json
    pipeline.npz        PerceptionPipeline on seq/ (odometry 50 ms late): per frame the det-frame cloud,
                        poses, ground estimate, detections fed in, tracks and predictions
                        and the planner-frame output (PerceptionPipeline.to_local, local_*)
    dets.npz            those detections per frame, the C++ replay's input (dogperc_replay --detections)
    geometry.npz, imm.npz, lsa.npz, pose_buffer.npz, preprocess.npz, pillars.npz, decode.npz, ins.npz
    clouds.npz          PointCloud2 byte layouts (XT32, Livox, integer fields, big-endian) and what
                        ros_utils.scan_from_msg reads from them
    kygi.yaml           the kygi node's configuration (export_cpp_config.py --robot kygi, with node:)
    online_msgs.npz, online.npz, online_dets.npz, online_config.yaml
                        the moving sample bag's first clouds through the online node's message handling
                        (cpp_parity.py python --bag); only with rosbags installed and samples/ checked out
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.optimize import linear_sum_assignment  # noqa: E402

from export_cpp_config import config_dict, node_section, to_yaml  # noqa: E402
from dog_perception.detection.boxes import CLASSES, Detections  # noqa: E402
from dog_perception.detection.decode import HEAD_ORDER, decode_centerpoint  # noqa: E402
from dog_perception.detection.filters import (  # noqa: E402
    DetectionFilterConfig, filter_detections, filter_measurements, filter_reasons)
from dog_perception.detection.model_meta import ModelMeta, postprocess  # noqa: E402
from dog_perception.detection.oracle import OracleDetector, OracleNoise  # noqa: E402
from dog_perception.detection.pillar_host import decorate_pillars, pad_pillars, scatter_pillars  # noqa: E402
from dog_perception.detection.voxelize import voxelize  # noqa: E402
from dog_perception.geometry import inv_T, quat_to_R, rot_zyx, so3_exp, so3_log, wrap_angle  # noqa: E402
from dog_perception.inspvax import EnuOrigin, pose_from_inspvax  # noqa: E402
from dog_perception.io import SequenceReader, odom_row, write_sequence  # noqa: E402
from dog_perception.node_config import node_parameters, preprocess_config  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.pose_buffer import PoseBuffer  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig, Preprocessor  # noqa: E402
from dog_perception.preprocess.frames import det_frame_from_body  # noqa: E402
from dog_perception.preprocess.pipeline import dedup_returns  # noqa: E402
from dog_perception.ros_utils import scan_from_msg  # noqa: E402
from dog_perception.preprocess.self_filter import SelfFilter, SelfFilterConfig, neighbor_counts  # noqa: E402
from dog_perception.sim import GaitModel, make_scenario  # noqa: E402
from dog_perception.tracking import TrackerConfig  # noqa: E402
from dog_perception.tracking.imm import CTModel, CVModel, IMM  # noqa: E402

MOTION = {"static": 0, "moving": 1, "stopped": 2}
ODOM_LATENCY = 0.05


def offsets(counts):
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def dets_arrays(prefix, dets_list):
    d = {prefix + "_offsets": offsets([len(x) for x in dets_list])}
    cat = Detections.concat(dets_list) if any(len(x) for x in dets_list) else Detections()
    d[prefix + "_boxes"] = np.asarray(cat.boxes, np.float64).reshape(-1, 7)
    d[prefix + "_velocities"] = np.asarray(cat.velocities, np.float64).reshape(-1, 2)
    d[prefix + "_scores"] = np.asarray(cat.scores, np.float64)
    d[prefix + "_labels"] = np.asarray(cat.labels, np.int64)
    d[prefix + "_subclasses"] = np.asarray(cat.subclasses, np.int64)
    return d


# ------------------------------------------------------------------ the simulated recording
def write_sim_sequence(seq_dir, duration=2.6, seed=3):
    gait = GaitModel(speed=1.2, yaw_rate=0.25, pitch_amp_deg=6.0, base_height=0.45)
    sim = make_scenario(seed=seed, duration=duration + 1.0, num_pedestrians=6, num_cyclists=2, num_vehicles=3,
                        gait=gait)
    n_scans = int(round(duration / sim.lidar.period))
    scans, infos = [], {}
    for k in range(n_scans):
        scan, info = sim.scan(k, t0=1.0)              # the poses start before the first scan
        scans.append(scan)
        infos[float(scan.stamp)] = info
    ts, gyro = sim.imu(0.0, scans[-1].stamp + 0.05, rate=400.0)
    imu = np.column_stack([ts, gyro])
    odom = []
    for t in np.arange(0.0, scans[-1].stamp + 0.05, 0.05):    # 20 Hz odometry without a twist
        R, p, _ = sim.odometry(t, pos_noise=0.002, rot_noise=0.0005)
        odom.append(odom_row(t, R, p))
    meta = dict(T_body_lidar=sim.T_bl, base_height=gait.base_height, det_frame=dict(auto_ground=True, auto_level=True))
    write_sequence(seq_dir, scans, imu, np.asarray(odom), meta)
    return sim, infos, gait


def run_pipeline(seq_dir, sim, infos, base_height, tracker_cfg):
    reader = SequenceReader(seq_dir)
    pipe = PerceptionPipeline(reader.preprocess_config(), detector=None, tracker_cfg=tracker_cfg)
    oracle = OracleDetector(OracleNoise(), seed=11)
    rec = dict(stamp=[], T_world_det=[], T_world_body=[], num_raw=[], num_kept=[], points=[], ground_offset=[],
               ground_tilt=[], dets=[], tracks=[], preds=[], local=[])
    for ev in reader.events(odom_latency=ODOM_LATENCY):
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
            continue
        if ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
            continue
        scan = ev[1]
        try:
            T_wd = det_frame_from_body(pipe.poses.pose_at(scan.stamp), base_height)
        except Exception:
            continue
        dets = oracle(infos[float(scan.stamp)]["gt"], inv_T(T_wd), scan.stamp)
        out = pipe.on_scan(scan, detections=dets)
        if out is None:
            continue
        rec["stamp"].append(out.stamp)
        rec["T_world_det"].append(out.frame.T_world_det)
        rec["T_world_body"].append(out.frame.T_world_body)
        rec["num_raw"].append(out.frame.num_raw)
        rec["num_kept"].append(out.frame.num_kept)
        rec["points"].append(out.frame.points)
        g = pipe.pre.ground
        rec["ground_offset"].append(g.offset)
        rec["ground_tilt"].append(np.asarray(g.tilt, np.float64))
        rec["dets"].append(out.detections)
        rec["tracks"].append(out.tracks)
        rec["preds"].append(out.predictions)
        rec["local"].append(pipe.to_local(out))
    return rec


def local_arrays(rec):
    """PerceptionPipeline.to_local per frame, rows in the order of trk_* / pred_*."""
    tr = [t for ts, _ in rec["local"] for t in ts]
    pr = [p for _, ps in rec["local"] for p in ps]
    T = len(pr[0].times) if pr else 0
    modes, covs = np.full((len(pr), 2, T, 2), np.nan), np.full((len(pr), 2, T, 2, 2), np.nan)
    for i, p in enumerate(pr):
        modes[i, :len(p.probs)], covs[i, :len(p.probs)] = p.modes, p.covs
    return dict(local_id=np.asarray([t["id"] for t in tr], np.int64),
                local_position=np.asarray([t["position"] for t in tr], np.float64).reshape(-1, 3),
                local_velocity=np.asarray([t["velocity"] for t in tr], np.float64).reshape(-1, 2),
                local_yaw=np.asarray([t["yaw"] for t in tr], np.float64),
                local_covariance=np.asarray([t["covariance"] for t in tr], np.float64).reshape(-1, 4, 4),
                local_pred_modes=modes, local_pred_covs=covs)


def cloud_cases(rng):
    """PointCloud2 layouts and what scan_from_msg makes of them (the C++ parser must read the same)."""
    from types import SimpleNamespace as NS

    def field(name, offset, datatype):
        return NS(name=name, offset=offset, datatype=datatype, count=1)

    def points(n):
        xyz = rng.normal(0, 8, (n, 3)).astype(np.float32)
        xyz[::17] = 0.0                                   # no return
        xyz[5::23, 1] = np.nan
        xyz[9::31, 2] = np.inf
        return xyz

    cases, n = [], 300
    stamp = NS(sec=1789980190, nanosec=502000000)
    # Hesai XT32 (0921): float64 absolute seconds at the unaligned offset 18
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                   "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"], "offsets": [0, 4, 8, 12, 16, 18],
                   "itemsize": 26})
    a = np.zeros(n, dt)
    xyz = points(n)
    a["x"], a["y"], a["z"] = xyz.T
    a["intensity"], a["ring"] = rng.uniform(0, 255, n), np.arange(n) % 32
    a["timestamp"] = 1789980190.4 + np.sort(rng.uniform(0, 0.1, n))
    cases.append(("xt32", a, [field("x", 0, 7), field("y", 4, 7), field("z", 8, 7), field("intensity", 12, 7),
                              field("ring", 16, 4), field("timestamp", 18, 8)], 1, "timestamp", "absolute", 0, False))
    # Livox Mid-360 (livox_ros_driver2): nanoseconds as float64, noise tags
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "tag", "line", "timestamp"],
                   "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"], "offsets": [0, 4, 8, 12, 16, 17, 18],
                   "itemsize": 26})
    a = np.zeros(n, dt)
    xyz = points(n)
    a["x"], a["y"], a["z"] = xyz.T
    a["intensity"], a["tag"], a["line"] = rng.uniform(0, 100, n), rng.integers(0, 16, n), np.arange(n) % 4
    a["timestamp"] = 1789980190.4e9 + np.sort(rng.uniform(0, 1e8, n))
    cases.append(("livox", a, [field("x", 0, 7), field("y", 4, 7), field("z", 8, 7), field("intensity", 12, 7),
                               field("tag", 16, 2), field("line", 17, 2), field("timestamp", 18, 8)], 1,
                  "timestamp", "absolute_ns", 3, False))
    # integer coordinates (as some drivers pack them), uint8 intensity, uint32 nanoseconds after the stamp, 2 rows
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "t"], "formats": ["<i2", "<i2", "<i4", "u1", "<u4"],
                   "offsets": [0, 2, 4, 8, 12], "itemsize": 16})
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = rng.integers(-3000, 3000, n), rng.integers(-3000, 3000, n), rng.integers(-500, 500, n)
    a["x"][::13] = a["y"][::13] = a["z"][::13] = 0
    a["intensity"], a["t"] = rng.integers(0, 256, n), np.sort(rng.integers(0, 100_000_000, n))
    cases.append(("ints", a, [field("x", 0, 3), field("y", 2, 3), field("z", 4, 5), field("intensity", 8, 2),
                              field("t", 12, 6)], 2, "t", "relative_ns", 0, False))
    # big-endian, no intensity, no per-point time: the header stamp is the scan's
    dt = np.dtype({"names": ["x", "y", "z", "rel"], "formats": [">f4", ">f4", ">f4", ">f8"], "offsets": [0, 4, 8, 12],
                   "itemsize": 20})
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = points(n).T
    a["rel"] = rng.uniform(0, 0.1, n)
    cases.append(("bigendian", a, [field("x", 0, 7), field("y", 4, 7), field("z", 8, 7), field("rel", 12, 8)], 1,
                  "time", "relative", 0, True))

    def text(t):                                          # the C++ npz reader takes numbers only
        return np.frombuffer(t.encode(), np.uint8)

    out = dict(cases=np.asarray(len(cases)))
    for k, (name, arr, fields, height, tfield, tmode, mask, big) in enumerate(cases):
        msg = NS(header=NS(stamp=stamp), fields=fields, width=len(arr) // height, height=height,
                 point_step=arr.dtype.itemsize, is_bigendian=big, data=arr.tobytes())
        scan = scan_from_msg(msg, tfield, tmode, noise_tag_mask=mask)
        p = "c%d_" % k
        out.update({p + "data": np.frombuffer(arr.tobytes(), np.uint8), p + "width": np.asarray(msg.width),
                    p + "height": np.asarray(height), p + "point_step": np.asarray(arr.dtype.itemsize),
                    p + "bigendian": np.asarray(int(big)), p + "stamp": np.asarray(stamp.sec + stamp.nanosec * 1e-9),
                    p + "field_names": text(",".join(f.name for f in fields)), p + "time_field": text(tfield),
                    p + "field_offsets": np.asarray([f.offset for f in fields], np.int64),
                    p + "field_types": np.asarray([f.datatype for f in fields], np.int64),
                    p + "time_mode": text(tmode), p + "mask": np.asarray(mask),
                    p + "xyz": scan.xyz, p + "intensity": scan.intensity, p + "scan_stamp": np.asarray(scan.stamp),
                    p + "times": np.zeros(0) if scan.point_times is None else scan.point_times})
    return out


def online_golden(out, max_clouds=8):
    """The sample bag's first clouds through the online node's message handling (cpp_parity.py python --bag)."""
    bag = os.path.join(ROOT, "samples", "rosbag2_2026_09_21-16_41_27_moving")
    try:
        import rosbags  # noqa: F401
    except ImportError:
        print("golden: no rosbags, the online message replay is left out")
        return
    if not os.path.isdir(bag):
        print("golden: %s missing, the online message replay is left out" % bag)
        return
    from types import SimpleNamespace as NS
    from cpp_parity import run_python
    run_python(NS(seq=None, bag=bag, robot="kygi", messages=os.path.join(out, "online_msgs.npz"),
                  max_clouds=max_clouds, no_detector=True, exported=None, detections=None, odom_latency=0.0,
                  save_points=True, out=os.path.join(out, "online.npz"), dets=os.path.join(out, "online_dets.npz"),
                  config=os.path.join(out, "online_config.yaml")))


def pipeline_arrays(rec, save_points=True, points_every=1):
    """Per-frame results as flat arrays with offsets (the layout dogperc_replay writes). The clouds of every
    ``points_every``-th frame only (point_offsets: no points for the others)."""
    F = len(rec["stamp"])
    keep_pts = [save_points and f % points_every == 0 for f in range(F)]
    a = dict(stamp=np.asarray(rec["stamp"]), T_world_det=np.asarray(rec["T_world_det"]).reshape(F, 4, 4),
             T_world_body=np.asarray(rec["T_world_body"]).reshape(F, 4, 4),
             num_raw=np.asarray(rec["num_raw"], np.int64),
             num_kept=np.asarray(rec["num_kept"], np.int64), ground_offset=np.asarray(rec["ground_offset"]),
             ground_tilt=np.asarray(rec["ground_tilt"]).reshape(F, 2),
             point_offsets=offsets([len(p) if k else 0 for p, k in zip(rec["points"], keep_pts)]),
             num_points=np.asarray([len(p) for p in rec["points"]], np.int64),
             points=(np.concatenate([p for p, k in zip(rec["points"], keep_pts) if k]) if any(keep_pts)
                     else np.zeros((0, 5))).astype(np.float32))
    a.update(dets_arrays("det", rec["dets"]))
    tr = [t for ts in rec["tracks"] for t in ts]
    a["trk_offsets"] = offsets([len(ts) for ts in rec["tracks"]])
    a["trk_id"] = np.asarray([t.track_id for t in tr], np.int64)
    a["trk_label"] = np.asarray([t.label for t in tr], np.int64)
    a["trk_position"] = np.asarray([t.position for t in tr], np.float64).reshape(-1, 3)
    a["trk_velocity"] = np.asarray([t.velocity for t in tr], np.float64).reshape(-1, 2)
    a["trk_yaw"] = np.asarray([t.yaw for t in tr], np.float64)
    a["trk_size"] = np.asarray([t.size for t in tr], np.float64).reshape(-1, 3)
    a["trk_score"] = np.asarray([t.score for t in tr], np.float64)
    a["trk_covariance"] = np.asarray([t.covariance for t in tr], np.float64).reshape(-1, 4, 4)
    a["trk_mode_probs"] = np.asarray([t.mode_probs for t in tr], np.float64).reshape(-1, 2)
    a["trk_age"] = np.asarray([t.age for t in tr], np.float64)
    a["trk_hits"] = np.asarray([t.hits for t in tr], np.int64)
    a["trk_coasting"] = np.asarray([t.coasting for t in tr], np.int64)
    a["trk_motion"] = np.asarray([MOTION[t.motion] for t in tr], np.int64)
    a["trk_time_since_moving"] = np.asarray([t.time_since_moving for t in tr], np.float64)
    a["trk_speed_displacement"] = np.asarray([t.speed_displacement for t in tr], np.float64)
    a["trk_speed_head"] = np.asarray([t.speed_head for t in tr], np.float64)
    a["trk_measurement"] = np.asarray([np.full(6, np.nan) if t.measurement is None else t.measurement for t in tr],
                                      np.float64).reshape(-1, 6)
    a["trk_history_len"] = np.asarray([len(t.history) for t in tr], np.int64)
    pr = [p for ps in rec["preds"] for p in ps]
    a["pred_offsets"] = offsets([len(ps) for ps in rec["preds"]])
    T = len(pr[0].times) if pr else 0
    a["pred_times"] = np.asarray(pr[0].times if pr else [], np.float64)
    a["pred_track_id"] = np.asarray([p.track_id for p in pr], np.int64)
    a["pred_label"] = np.asarray([p.label for p in pr], np.int64)
    a["pred_num_modes"] = np.asarray([len(p.probs) for p in pr], np.int64)
    modes = np.full((len(pr), 2, T, 2), np.nan)
    covs = np.full((len(pr), 2, T, 2, 2), np.nan)
    probs = np.full((len(pr), 2), np.nan)
    for i, p in enumerate(pr):
        k = len(p.probs)
        modes[i, :k], covs[i, :k], probs[i, :k] = p.modes, p.covs, p.probs
    a.update(pred_modes=modes, pred_covs=covs, pred_probs=probs)
    return a


# ------------------------------------------------------------------ unit data
def geometry_data(rng):
    rv = rng.normal(0, 1.0, (60, 3))
    rv[:10] *= 1e-10                               # first-order branch
    rv[10:20] *= 1e-4
    R = so3_exp(rv)
    q = rng.normal(0, 1, (40, 4))
    eul = rng.uniform(-np.pi, np.pi, (40, 3))
    ang = np.concatenate([rng.uniform(-20, 20, 100), [np.pi, -np.pi, 3 * np.pi, 0.0, -0.0]])
    return dict(rotvec=rv, so3_exp=R, so3_log=so3_log(R), quat=q, quat_R=np.stack([quat_to_R(x) for x in q]),
                euler=eul, rot_zyx=np.stack([rot_zyx(*e) for e in eul]), angles=ang, wrapped=wrap_angle(ang))


def imm_data(rng):
    imm = IMM([CVModel(1.5, 1e-3), CTModel(1.5, 0.25)], 0.95)
    imm.initialize([2.0, -1.0, 3.0, 0.5, 0.0], np.diag([0.09, 0.09, 0.64, 0.64, 0.09]))
    ops, dts, zs, xs, Ps, mus, lls = [], [], [], [], [], [], []
    x_true = np.array([2.0, -1.0, 3.0, 0.5])
    for k in range(40):
        kind = [0, 1, 0, 2][k % 4]
        dt = float(rng.uniform(0.05, 0.15)) if kind == 0 else 0.0
        z = np.zeros(4)
        ll = np.nan
        if kind == 0:
            imm.predict(dt)
            x_true[:2] += dt * x_true[2:]
            x_true[2:] = (rot_zyx(0.3 * dt, 0, 0)[:2, :2] @ x_true[2:])
        elif kind == 1:
            z = np.concatenate([x_true[:2] + rng.normal(0, 0.2, 2), x_true[2:] + rng.normal(0, 0.5, 2)])
            H = np.zeros((4, 5))
            H[:4, :4] = np.eye(4)
            ll = imm.update(z, H, np.diag([0.09, 0.09, 0.64, 0.64]))
        else:
            z[:2] = x_true[:2] + rng.normal(0, 0.2, 2)
            H = np.zeros((2, 5))
            H[:2, :2] = np.eye(2)
            ll = imm.update(z[:2], H, np.eye(2) * 0.09)
        ops.append(kind)
        dts.append(dt)
        zs.append(z)
        xs.append(imm.x.copy())
        Ps.append(imm.P.copy())
        mus.append(imm.mu.copy())
        lls.append(ll)
    r = imm.rollout(3.0, 0.5, mixed=False)
    return dict(ops=np.asarray(ops, np.int64), dts=np.asarray(dts), zs=np.asarray(zs), x=np.asarray(xs),
                P=np.asarray(Ps), mu=np.asarray(mus), loglik=np.asarray(lls), roll_means=r["mode_means"],
                roll_covs=r["mode_covs"], roll_times=r["times"])


def lsa_data(rng):
    out = {}
    for k, (n, m) in enumerate([(5, 5), (4, 7), (8, 3), (1, 6), (12, 12)]):
        c = rng.uniform(0, 10, (n, m))
        r, col = linear_sum_assignment(c)
        out["cost%d" % k], out["rows%d" % k], out["cols%d" % k] = c, r.astype(np.int64), col.astype(np.int64)
    return out


def pose_buffer_data(sim, rng):
    ts = np.sort(np.concatenate([np.arange(0.5, 1.5, 1 / 400.0), [1.5 - 1 / 400.0 + 3e-4]]))   # a 0.3 ms end segment
    R, p = sim.robot.pose(ts)
    buf = PoseBuffer()
    for i in range(len(ts)):
        buf.add(ts[i], R[i], p[i])
    q = np.concatenate([rng.uniform(ts[0], ts[-1], 50), ts[-1] + rng.uniform(0, 0.05, 10),
                        ts[0] - rng.uniform(0, 0.05, 10), ts[:5]])
    Rq, pq = buf.interpolate(q)
    return dict(t=ts, R=R, p=p, query=q, R_query=Rq, p_query=pq)


def preprocess_data(seq_dir, sim):
    reader = SequenceReader(seq_dir)
    from dog_perception.io import load_scan
    scan = load_scan(reader.scan_files[3])
    dd = dedup_returns(scan)
    sf = SelfFilter(SelfFilterConfig(), sim.T_bl)
    fin = np.isfinite(dd.xyz).all(axis=1)
    keep = sf(dd.xyz[fin])
    xyz = dd.xyz[fin]
    near = keep & (np.linalg.norm(xyz, axis=1) < 3.0)
    cnt = neighbor_counts(xyz[near], 0.2)
    ts = np.arange(0.5, scan.stamp + 0.02, 1 / 400.0)
    R, p = sim.robot.pose(ts)
    buf = PoseBuffer()
    for i in range(len(ts)):
        buf.add(ts[i], R[i], p[i])
    pre = Preprocessor(PreprocessConfig(T_body_lidar=sim.T_bl, det_frame=DetFrameConfig(base_height=0.45)), buf)
    xyz_b, inten = pre.sweep_to_body(scan)
    return dict(xyz=scan.xyz, intensity=scan.intensity, t=scan.point_times, stamp=np.float64(scan.stamp),
                dedup_xyz=dd.xyz, dedup_t=dd.point_times, filter_xyz=xyz, keep=keep.astype(np.int64),
                near_xyz=xyz[near],
                near_counts=cnt.astype(np.int64), buf_t=ts, buf_R=R, buf_p=p, body_xyz=xyz_b, body_intensity=inten)


NUSC_TASKS = [["car"], ["truck", "construction_vehicle"], ["bus", "trailer"], ["barrier"], ["motorcycle", "bicycle"],
              ["pedestrian", "traffic_cone"]]


def synthetic_meta(xy_range=20.4, max_pillars=12000):
    """The deployed nuScenes Pillar's layout on a smaller grid (golden files stay small)."""
    from dog_perception.detection.boxes import NUSC_TO_DOG, NUSC_TO_SUBCLASS, class_index, subclass_index
    names = [n for t in NUSC_TASKS for n in t]
    radius = {"car": 4.0, "truck": 12.0, "construction_vehicle": 12.0, "bus": 10.0, "trailer": 10.0, "barrier": 1.0,
              "motorcycle": 0.85, "bicycle": 0.85, "pedestrian": 0.175, "traffic_cone": 0.175}
    pcr = [-xy_range, -xy_range, -5.0, xy_range, xy_range, 3.0]
    n = int(round(2 * xy_range / 0.2))
    meta = ModelMeta(
        voxel_size=[0.2, 0.2, 8.0], pc_range=pcr, grid_size=[n, n, 1], max_points=20, max_voxels=60000,
        max_pillars=max_pillars, num_point_features=5, pfn_channels=10, bev_channels=64, z_range=[-1.0, 5.0],
        frame=dict(nusc_axes=True, z_offset=-1.84019), tasks=[{"class_names": t} for t in NUSC_TASKS],
        head_keys=list(HEAD_ORDER),
        test_cfg=dict(pc_range=pcr[:2], voxel_size=[0.2, 0.2], out_size_factor=4, score_threshold=0.1,
                      post_center_limit_range=[-xy_range, -xy_range, -10.0, xy_range, xy_range, 10.0],
                      min_radius=[max(radius[c] for c in t) for t in NUSC_TASKS], nms={"nms_post_max_size": 83},
                      pre_max_size=1000),
        label_map=[class_index(NUSC_TO_DOG.get(c, "")) for c in names],
        subclass_map=[subclass_index(NUSC_TO_SUBCLASS[c]) if c in NUSC_TO_SUBCLASS else -1 for c in names],
        merged_nms={"vehicle": 2.0, "cyclist": 0.9, "pedestrian": [0.6, 1.2, 0.6]})
    meta.onnx["rpn"]["outputs"] = meta.rpn_outputs
    return meta


def pillar_data(meta, points_det, rng):
    m = meta
    frame = m.model_frame
    z = points_det[:, 2]
    crop = points_det[(z >= m.z_range[0]) & (z <= m.z_range[1])]
    pts = frame.points_to_model(crop)[:, :m.num_point_features].astype(np.float32)
    v, c, n = voxelize(pts, m.voxel_size, m.pc_range, m.max_points, m.max_voxels)
    feats = decorate_pillars(v, n, c, m.voxel_size, m.pc_range)
    x, cpad, nv = pad_pillars(feats, c, m.max_pillars)
    pf = rng.normal(0, 1, (m.bev_channels, m.max_pillars)).astype(np.float32)
    bev = scatter_pillars(pf, cpad, nv, m.grid_size)
    return dict(points_det=points_det, model_points=pts, voxels=v, coords=c, num_points=n, feats=feats, pfn_in=x,
                pfn_out=pf, bev=bev, num_valid=np.int64(nv))


def head_maps(meta, gt, T_det_world, rng):
    """Synthetic RPN outputs: low logits everywhere, a few random peaks, and peaks on the simulated objects."""
    m = meta
    H = W = m.grid_size[0] // m.test_cfg["out_size_factor"]
    cell = m.test_cfg["voxel_size"][0] * m.test_cfg["out_size_factor"]
    lo = m.test_cfg["pc_range"]
    maps = []
    for t, names in enumerate(NUSC_TASKS):
        C = len(names)
        d = dict(reg=rng.uniform(0, 1, (2, H, W)), height=rng.normal(-1.0, 0.3, (1, H, W)),
                 dim=np.log(np.stack([np.full((H, W), 1.8), np.full((H, W), 4.2), np.full((H, W), 1.6)])) +
                 rng.normal(0, 0.1, (3, H, W)), rot=rng.normal(0, 1, (2, H, W)), vel=rng.normal(0, 1, (2, H, W)),
                 hm=rng.normal(-7, 1, (C, H, W)))
        for _ in range(25):
            i, j, c = rng.integers(H), rng.integers(W), rng.integers(C)
            d["hm"][c, i, j] = rng.uniform(-2.5, 3.0)
        maps.append(d)
    where = {"vehicle": (0, 0), "pedestrian": (5, 0), "cyclist": (4, 1)}
    R = T_det_world[:3, :3]
    for g in gt:
        if g["label"] not in where:
            continue
        cd = R @ g["center"] + T_det_world[:3, 3]
        xm, ym, zm = -cd[1], cd[0], cd[2] - 1.84019
        col, row = (xm - lo[0]) / cell, (ym - lo[1]) / cell
        if not (0 <= col < W and 0 <= row < H):
            continue
        t, c = where[g["label"]]
        i, j = int(row), int(col)
        d = maps[t]
        d["hm"][c, i, j] = rng.uniform(0.5, 4.0)
        d["reg"][:, i, j] = (col - j, row - i)
        d["height"][0, i, j] = zm
        L, Wd, Hh = g["size"]
        d["dim"][:, i, j] = np.log([Wd, L, Hh])
        yaw_m = g["yaw"] + np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2
        r = -yaw_m - np.pi / 2
        d["rot"][:, i, j] = (np.sin(r), np.cos(r))
    return [{k: v.astype(np.float32) for k, v in d.items()} for d in maps]


def decode_data(meta, maps, points_det):
    box9, scores, labels = decode_centerpoint(maps, meta.num_classes, meta.test_cfg)
    dets = postprocess(box9, scores, labels, meta.label_map, meta.subclass_map, meta.model_frame, meta.merged_nms, 1.5)
    cfg = DetectionFilterConfig()
    why = filter_reasons(dets, points_det, cfg)
    final = filter_detections(dets, points_det, cfg)
    bottom, counts = filter_measurements(dets, points_det, cfg)
    reasons = {"": 0, "floating": 1, "sunk": 2, "few points": 3, "duplicate": 4}
    out = {"head%d_%s" % (t, k): maps[t][k] for t in range(len(maps)) for k in HEAD_ORDER}
    out.update(box9=np.asarray(box9, np.float64).reshape(-1, 9), scores=np.asarray(scores, np.float64),
               labels=np.asarray(labels, np.int64), reasons=np.asarray([reasons[w] for w in why], np.int64),
               points_det=points_det, bottom=np.asarray(bottom, np.float64), counts=np.asarray(counts, np.int64))
    out.update(dets_arrays("raw", [dets]))
    out.update(dets_arrays("final", [final]))
    return out


def ins_data(rng):
    origin = EnuOrigin(29.56, 106.45, 260.0)
    lat = 29.56 + rng.normal(0, 1e-4, 20)
    lon = 106.45 + rng.normal(0, 1e-4, 20)
    h = 260.0 + rng.normal(0, 1, 20)
    az = rng.uniform(0, 360, 20)

    class Msg:
        pass

    Rs, ps = [], []
    for k in range(20):
        msg = Msg()
        msg.ins_status = Msg()
        msg.ins_status.data = 3
        msg.latitude, msg.longitude, msg.height, msg.azimuth = lat[k], lon[k], h[k], az[k]
        msg.east_velocity, msg.north_velocity, msg.up_velocity = 1.0, 0.5, 0.0
        msg.header = Msg()
        msg.header.stamp = Msg()
        msg.header.stamp.sec, msg.header.stamp.nanosec = 10 + k, 0
        _, R, p, _ = pose_from_inspvax(msg, origin, (-0.06, -0.54, 0.0), 180.0)
        Rs.append(R)
        ps.append(p)
    return dict(lat=lat, lon=lon, h=h, azimuth=az, enu=np.stack([origin.enu(a, b, c) for a, b, c in zip(lat, lon, h)]),
                R=np.stack(Rs), p=np.stack(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    seq = os.path.join(out, "seq")
    sim, infos, gait = write_sim_sequence(seq)
    tracker_cfg = TrackerConfig(classes=CLASSES)
    reader = SequenceReader(seq)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        f.write(to_yaml(config_dict(reader.preprocess_config(), tracker=tracker_cfg, source=dict(golden=seq))) + "\n")
    with open(os.path.join(out, "default.yaml"), "w") as f:
        f.write(to_yaml(config_dict(source=dict(defaults="dataclasses"))) + "\n")

    rec = run_pipeline(seq, sim, infos, gait.base_height, tracker_cfg)
    pa = pipeline_arrays(rec, points_every=5)                  # the clouds dominate the size: every 5th frame
    pa.update(local_arrays(rec))
    pa["odom_latency"] = np.float64(ODOM_LATENCY)
    np.savez_compressed(os.path.join(out, "pipeline.npz"), **pa)
    d = dets_arrays("det", rec["dets"])
    np.savez_compressed(os.path.join(out, "dets.npz"), stamp=np.asarray(rec["stamp"]),
                        **{k[4:]: v for k, v in d.items()})

    np.savez_compressed(os.path.join(out, "geometry.npz"), **geometry_data(rng))
    np.savez_compressed(os.path.join(out, "imm.npz"), **imm_data(rng))
    np.savez_compressed(os.path.join(out, "lsa.npz"), **lsa_data(rng))
    np.savez_compressed(os.path.join(out, "pose_buffer.npz"), **pose_buffer_data(sim, rng))
    np.savez_compressed(os.path.join(out, "preprocess.npz"), **preprocess_data(seq, sim))
    np.savez_compressed(os.path.join(out, "ins.npz"), **ins_data(rng))
    np.savez_compressed(os.path.join(out, "clouds.npz"), **cloud_cases(rng))
    with open(os.path.join(out, "kygi.yaml"), "w") as f:
        f.write(to_yaml(config_dict(preprocess_config(node_parameters("kygi")), source=dict(robot="kygi"),
                                    node=node_section("kygi"))) + "\n")
    online_golden(out)

    meta = synthetic_meta()
    os.makedirs(os.path.join(out, "model"), exist_ok=True)
    meta.save(os.path.join(out, "model", "model_meta.json"))
    k = len(rec["stamp"]) // 2
    points = rec["points"][k]
    np.savez_compressed(os.path.join(out, "pillars.npz"), **pillar_data(meta, points, rng))
    stamp = rec["stamp"][k]
    maps = head_maps(meta, infos[float(stamp)]["gt"], inv_T(rec["T_world_det"][k]), rng)
    np.savez_compressed(os.path.join(out, "decode.npz"), **decode_data(meta, maps, points))
    summary = dict(frames=len(rec["stamp"]), points=int(sum(len(p) for p in rec["points"])),
                   tracks=int(len(pa["trk_id"])), predictions=int(len(pa["pred_track_id"])))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f)
    print("golden:", json.dumps(summary))


if __name__ == "__main__":
    main()
