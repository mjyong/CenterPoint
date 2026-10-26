"""Replay a recorded sequence through the four-stage pipeline.

Runs one or both CenterPoint variants on the *same* preprocessed frames and
saves, per detector, everything the other tools need:

    <out>/<name>/results.pkl    per frame: stamp, T_world_det, detections (det frame),
                                tracks, predictions, timings
    <out>/<name>/track_log.npz  tracker log for train_predictor.py
    <out>/frames/*.npy          (--save-frames) preprocessed clouds for auto_label / export
    <out>/<scene>_<first>.mcap  Foxglove view of the first detector (skip with --no-mcap), with the
                                sequence's camera streams when it has them (cameras.mcap)

    python tools/dog/run_sequence.py --seq data/rec_001 --detectors pillar,voxel \
        --ckpt-pillar pp.pth --ckpt-voxel voxel.pth --out work_dirs/rec_001

Evaluating models on scenes the teacher already ran (``--save-frames``) does not need the raw
scans: ``--replay data/runs/<scene>`` feeds the saved detector inputs with the poses of that run's
results.pkl, which skips reading and preprocessing the scans (about 150 of the 480 ms per frame
with two detectors on the server) and gives every model exactly the frames the training data was
exported from. ``--no-predict`` skips the trajectory prediction nobody reads in an evaluation.
``--save-raw`` adds, per frame, the detections before the geometric filters (``raw_detections``) and
what the filters measure on them (``raw_measured``: box bottom above the ground, current-sweep points),
for ``tools/dog/pipeline_eval.py --detections raw``.
Several scenes go in one call (``--out`` with ``{scene}``), ``--jobs N`` runs N of them at once,
each worker loading the detectors once:

    python tools/dog/run_sequence.py --replay data/runs/0921_164127_s00{0,1,2} --detectors pillar,voxel \
        --ckpt-pillar pp.pth --ckpt-voxel voxel.pth --no-predict --no-mcap --jobs 3 \
        --device cuda:0 --out data/runs_veh_v1/{scene}
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection import DEFAULT_SCORE_THRESHOLDS  # noqa: E402
from dog_perception.detection.filters import DetectionFilterConfig, filter_measurements  # noqa: E402
from dog_perception.geometry import transform_points  # noqa: E402
from dog_perception.io import SequenceReader, pose_bad  # noqa: E402
from dog_perception.mcap_viz import McapViz  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.prediction.dataset import TrackLogger  # noqa: E402
from dog_perception.preprocess.pipeline import PreprocessedFrame  # noqa: E402

PRESETS = {
    "pillar": "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py",
    "voxel": "configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py",
}


def scene_name(path):
    return os.path.basename(os.path.normpath(path))


def live_frames(reader, pipe, odom_latency=0.0):
    """(frame, pose_bad) of each scan of the recording, preprocessed by ``pipe`` (its pose buffer
    takes the IMU and odometry in between)."""
    for ev in reader.events(odom_latency=odom_latency):
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
            continue
        if ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
            continue
        frame = pipe.pre.process(ev[1])
        if frame is not None:
            yield frame, pose_bad(reader.meta, frame.stamp)


def replay_results(run_dir):
    """The results.pkl of an earlier run of this scene: its stamps and poses go with the saved frames."""
    for n in ("voxel", "pillar"):
        p = os.path.join(run_dir, n, "results.pkl")
        if os.path.exists(p):
            return p
    found = sorted(d for d in os.listdir(run_dir) if os.path.exists(os.path.join(run_dir, d, "results.pkl")))
    if not found:
        raise SystemExit("%s: no <detector>/results.pkl next to the frames" % run_dir)
    return os.path.join(run_dir, found[0], "results.pkl")


def replay_frames(run_dir):
    """(frame, pose_bad) of each frame saved by ``--save-frames`` in ``run_dir``, in time order, with
    the poses and pose_bad flags of that run's results.pkl."""
    with open(replay_results(run_dir), "rb") as f:
        rows = sorted(pickle.load(f), key=lambda r: r["stamp"])
    missing = 0
    for r in rows:
        path = os.path.join(run_dir, "frames", "%.6f.npy" % r["stamp"])
        if not os.path.exists(path):
            missing += 1
            continue
        pts = np.load(path)
        yield PreprocessedFrame(stamp=float(r["stamp"]), points=pts, T_world_det=np.asarray(r["T_world_det"]),
                                T_world_body=np.asarray(r["T_world_body"]), num_raw=len(pts), num_kept=len(pts),
                                timings={}), bool(r.get("pose_bad", False))
    if missing:
        print("%s: %d frames of the results have no saved frame" % (run_dir, missing))


def build_detectors(args):
    from dog_perception.detection.centerpoint import CenterPointDetector
    return {n: CenterPointDetector(getattr(args, "config_" + n), getattr(args, "ckpt_" + n),
                                   device=args.device, tta=args.tta) for n in args.detectors.split(",")}


def run_scene(source, out, dets, args):
    """One scene: ``source`` is a sequence directory, or with ``--replay`` an earlier run directory.
    Returns the summary lines."""
    names = list(dets)
    scene = scene_name(source)
    if args.replay:
        seq = os.path.join(args.seq_root, scene)
        reader = SequenceReader(seq) if os.path.exists(os.path.join(seq, "meta.json")) else None
        pcfg = None
    else:
        reader = SequenceReader(source)
        base_h = args.base_height if args.base_height is not None else reader.meta.get("base_height", 0.45)
        pcfg = reader.preprocess_config(base_h, num_sweeps=args.num_sweeps, deskew=not args.no_deskew)
    # one pipeline drives preprocessing; each detector gets its own tracker / predictor
    pipes = {n: PerceptionPipeline(pcfg, dets[n], predictor=None if args.no_predict else "imm") for n in names}
    frames = replay_frames(source) if args.replay else live_frames(reader, pipes[names[0]], args.odom_latency)
    results = {n: [] for n in names}
    logs = {n: TrackLogger() for n in names}
    if args.save_frames:
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    mcap_path = None if args.no_mcap else (args.mcap or os.path.join(out, "%s_%s.mcap" % (scene, names[0])))
    if mcap_path:
        os.makedirs(os.path.dirname(os.path.abspath(mcap_path)), exist_ok=True)
    viz = McapViz(mcap_path) if mcap_path else None
    lines = []

    for frame, bad in frames:
        if args.save_frames:
            np.save(os.path.join(out, "frames", "%.6f.npy" % frame.stamp), frame.points)
        shown = None
        for n in names:
            d = dets[n](frame.points, frame.stamp)
            res = run_back_half(pipes[n], frame, d)
            row = dict(stamp=frame.stamp, T_world_det=frame.T_world_det, pose_bad=bad,
                       T_world_body=frame.T_world_body, detections=d, tracks=res[0],
                       predictions=res[1], timings=dict(frame.timings, **dets[n].last_timing))
            raw = getattr(dets[n], "last_raw", None) if getattr(args, "save_raw", False) else None
            if raw is not None:
                row.update(raw_detections=raw, raw_measured=filter_measurements(
                    raw, frame.points, getattr(dets[n], "filter_cfg", None) or DetectionFilterConfig()))
            results[n].append(row)
            logs[n].add(frame.stamp, res[0], frame.T_world_body, res[1])
            if n == names[0]:
                shown = (d.above(DEFAULT_SCORE_THRESHOLDS), res[0])
        if viz is not None and shown is not None:
            cur = frame.points[frame.points[:, -1] == 0]
            xyz_w = transform_points(frame.T_world_det, cur[:, :3]) if len(cur) else np.zeros((0, 3))
            viz.add(frame.stamp, xyz_w, cur[:, 3] if len(cur) else np.zeros(0),
                    shown[0].transform(frame.T_world_det, "world"), frame.T_world_body, shown[1])
        if len(results[names[0]]) % 20 == 0 and args.jobs <= 1:
            print("frame %d  " % len(results[names[0]]) +
                  "  ".join("%s: %d dets %.1f ms" % (n, len(results[n][-1]["detections"]),
                                                     results[n][-1]["timings"].get("network_ms", 0)) for n in names))

    for n in names:
        d = os.path.join(out, n)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.pkl"), "wb") as f:
            pickle.dump(results[n], f)
        logs[n].save(os.path.join(d, "track_log.npz"))
        lat = [r["timings"].get("network_ms", np.nan) for r in results[n][1:]]
        lines.append("[%s] %d frames, network %.1f ms (median)" % (n, len(results[n]), np.nanmedian(lat) if lat else np.nan))
    if viz is not None:
        if reader is not None and reader.cameras and results[names[0]]:   # the recording's cameras next to the boxes
            n_cam = viz.copy_from(reader.cameras, results[names[0]][0]["stamp"] - 1.5, results[names[0]][-1]["stamp"])
            lines.append("mcap: %d camera messages from %s" % (n_cam, reader.cameras))
        viz.close()
        lines.append("mcap %s  %d frames" % (mcap_path, viz.frames))
    return lines


_WORKER = {}


def _init_worker(args, threads):
    import torch
    torch.set_num_threads(threads)
    _WORKER["args"], _WORKER["dets"] = args, build_detectors(args)


def _work(job):
    source, out = job
    return scene_name(source), run_scene(source, out, _WORKER["dets"], _WORKER["args"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", nargs="+", help="sequence directories")
    src.add_argument("--replay", nargs="+", metavar="RUN_DIR",
                     help="earlier run directories with frames/ (--save-frames) and <detector>/results.pkl: "
                          "replay the saved detector inputs instead of preprocessing the scans")
    ap.add_argument("--seq-root", default="data/seq", help="--replay: <seq-root>/<scene> for the camera streams")
    ap.add_argument("--detectors", default="pillar")
    ap.add_argument("--config-pillar", default=None, help="default: from --ckpt-pillar (nuScenes or dog model)")
    ap.add_argument("--config-voxel", default=None, help="default: from --ckpt-voxel")
    ap.add_argument("--ckpt-pillar", default=None)
    ap.add_argument("--ckpt-voxel", default=None)
    ap.add_argument("--tta", action="store_true", help="double-flip TTA (offline teacher)")
    ap.add_argument("--base-height", type=float, default=None)
    ap.add_argument("--num-sweeps", type=int, default=5)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--odom-latency", type=float, default=0.0)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--no-predict", action="store_true", help="skip the trajectory prediction (evaluation runs)")
    ap.add_argument("--save-raw", action="store_true",
                    help="also keep the detections before the geometric filters, with what the filters measure, "
                         "so tools/dog/pipeline_eval.py can sweep the filter thresholds offline")
    ap.add_argument("--mcap", default=None,
                    help="Foxglove MCAP of the first detector (default: <out>/<scene>_<first>.mcap): "
                         "cloud, reported detections, tracks, pose")
    ap.add_argument("--no-mcap", action="store_true", help="do not write the Foxglove MCAP")
    ap.add_argument("--device", default=None)
    ap.add_argument("--jobs", type=int, default=1,
                    help="scenes at once, each worker with its own copy of the detectors (GPU memory adds up)")
    ap.add_argument("--out", required=True, help="output directory; with several scenes it must contain {scene}")
    args = ap.parse_args()

    from dog_perception.detection import PRESET_CONFIGS
    from dog_perception.detection.centerpoint import checkpoint_arch

    for n in ("pillar", "voxel"):
        ckpt = getattr(args, "ckpt_" + n)
        if getattr(args, "config_" + n) is None:
            arch = checkpoint_arch(ckpt) if ckpt else None
            setattr(args, "config_" + n, PRESET_CONFIGS.get(arch, PRESETS[n]))

    sources = args.replay or args.seq
    if len(sources) > 1 and "{scene}" not in args.out:
        ap.error("several scenes: --out must contain {scene}")
    if len(sources) > 1 and args.mcap:
        ap.error("--mcap names one file: use the default <out>/<scene>_<detector>.mcap with several scenes")
    if args.replay and args.save_frames:
        ap.error("--replay reads saved frames; --save-frames would only copy them")
    jobs = [(s, args.out.replace("{scene}", scene_name(s))) for s in sources]
    n_jobs = max(1, min(args.jobs, len(jobs)))
    args.jobs = n_jobs

    if n_jobs == 1:
        dets = build_detectors(args)
        for source, out in jobs:
            if len(jobs) > 1:
                print("== %s" % scene_name(source))
            for line in run_scene(source, out, dets, args):
                print(line)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    threads = max(1, (os.cpu_count() or 1) // n_jobs)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(threads)                  # read by the spawned workers when they start
    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             initargs=(args, threads)) as ex:
        for scene, lines in ex.map(_work, jobs):
            print("== %s" % scene)
            for line in lines:
                print(line)


def run_back_half(pipe, frame, dets):
    """Tracking + prediction for an already preprocessed / detected frame."""
    dets_w = dets.transform(frame.T_world_det, "world")
    tracks = pipe.tracker.step(dets_w, frame.stamp)
    preds = pipe.predictor(pipe.tracker) if pipe.predictor is not None else []
    return tracks, preds


if __name__ == "__main__":
    main()
