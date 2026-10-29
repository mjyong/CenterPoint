"""Parameters of the C++ pipeline (deploy/cpp) from the Python dataclasses: their single source.

The C++ library hard-codes nothing it could read from here (``deploy/cpp/include/dogperc/config.hpp``
mirrors the defaults only so tests can check they agree). Every key below is required by the loader.

    # generic defaults (what PerceptionPipeline() runs with)
    python tools/deploy/export_cpp_config.py --out deploy/config/default.yaml
    # the kygi robot as the online node runs it (node_config.py preset, also its ROS parameters)
    python tools/deploy/export_cpp_config.py --robot kygi --out deploy/config/kygi.yaml
    # a recorded sequence as the offline tools run it (meta.json: mounting, base height, det frame, self filter)
    python tools/deploy/export_cpp_config.py --seq data/seq/0921_164127_s000 --out <dir>/config.yaml
"""
import argparse
import inspect
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import CLASSES  # noqa: E402
from dog_perception.detection.filters import DEFAULT_SCORE_THRESHOLDS, DetectionFilterConfig  # noqa: E402
from dog_perception.pose_buffer import ImuPropagator, PoseBuffer  # noqa: E402
from dog_perception.prediction.kinematic import IMMPredictor  # noqa: E402
from dog_perception.preprocess import PreprocessConfig  # noqa: E402
from dog_perception.tracking import TrackerConfig  # noqa: E402
from dog_perception.tracking.imm import build_model  # noqa: E402

VERSION = 1


def per_class(value):
    """float or {class: value} -> {class: value} for all three classes (missing ones read 0, as in Python)."""
    if isinstance(value, dict):
        return {c: float(value.get(c, 0.0)) for c in CLASSES}
    return {c: float(value) for c in CLASSES}


def defaults(cls):
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty}


def preprocess_section(cfg):
    sf, df = cfg.self_filter, cfg.det_frame
    return dict(
        T_body_lidar=[float(v) for v in np.asarray(cfg.T_body_lidar, np.float64).reshape(-1)],
        num_sweeps=int(cfg.num_sweeps), max_time_span=float(cfg.max_time_span), deskew=bool(cfg.deskew),
        time_resolution=float(cfg.time_resolution), dedup_returns=bool(cfg.dedup_returns),
        intensity_scale=float(cfg.intensity_scale),
        self_filter=dict(body_boxes=[[float(v) for v in b] for b in sf.body_boxes], min_range=float(sf.min_range),
                         max_range=float(sf.max_range), outlier_radius=float(sf.outlier_radius),
                         outlier_min_neighbors=int(sf.outlier_min_neighbors),
                         outlier_max_range=float(sf.outlier_max_range)),
        det_frame=dict(base_height=float(df.base_height), auto_ground=bool(df.auto_ground),
                       auto_level=bool(df.auto_level), ground_ring=[float(v) for v in df.ground_ring],
                       ground_band=float(df.ground_band), ground_min_points=int(df.ground_min_points),
                       ground_alpha=float(df.ground_alpha), ground_max_correction=float(df.ground_max_correction),
                       ground_max_tilt=float(df.ground_max_tilt)))


def tracker_section(t):
    cp = {}
    for c in CLASSES:
        p = t.class_params[c]
        models = []
        for spec in p.models:
            m = build_model(spec)                 # resolves the q_omega default of each model kind
            models.append([spec[0], float(m.q_acc), float(m.q_omega)])
        cp[c] = dict(gate=float(p.gate), max_age=float(p.max_age), min_hits=int(p.min_hits), pos_std=float(p.pos_std),
                     vel_std=float(p.vel_std), models=models, stay_prob=float(p.stay_prob),
                     init_omega_std=float(p.init_omega_std))
    return dict(class_params=cp, classes=list(t.classes), high_score=per_class(t.high_score),
                low_score=float(t.low_score), matching=t.matching, use_velocity=bool(t.use_velocity),
                tentative_max_age=float(t.tentative_max_age), output_coasting=bool(t.output_coasting),
                output_min_score=per_class(t.output_min_score), history=float(t.history),
                size_alpha=float(t.size_alpha), z_alpha=float(t.z_alpha), yaw_alpha=float(t.yaw_alpha),
                heading_from_velocity=float(t.heading_from_velocity), motion_on_speed=per_class(t.motion_on_speed),
                motion_off_speed=per_class(t.motion_off_speed), motion_window=int(t.motion_window),
                motion_traj_window=float(t.motion_traj_window), motion_on_frames=int(t.motion_on_frames),
                motion_off_time=float(t.motion_off_time), stop_memory=float(t.stop_memory))


def filter_section(f, enabled=True):
    return dict(enabled=bool(enabled), min_points=per_class(f.min_points), box_margin=float(f.box_margin),
                max_float=float(f.max_float), max_sink=float(f.max_sink), local_ground=bool(f.local_ground),
                ground_cell=float(f.ground_cell), ground_ring=float(f.ground_ring),
                ground_percentile=float(f.ground_percentile), ground_min_cells=int(f.ground_min_cells),
                cross_class_nms=bool(f.cross_class_nms), cross_class_margin=float(f.cross_class_margin))


def config_dict(preprocess=None, tracker=None, detection_filter=None, predictor=None, source=None, node=None):
    pb, ip = defaults(PoseBuffer), defaults(ImuPropagator)
    pred = predictor or IMMPredictor()
    out = dict(
        version=VERSION, source=source or {},
        preprocess=preprocess_section(preprocess or PreprocessConfig()),
        pose_buffer=dict(max_age=float(pb["max_age"]), max_extrapolation=float(pb["max_extrapolation"]),
                         extrapolation_window=float(pb["extrapolation_window"])),
        imu=dict(gyro_bias=[float(v) for v in ip["gyro_bias"]], imu_history=float(ip["imu_history"])),
        detection_filter=filter_section(detection_filter or DetectionFilterConfig()),
        score_thresholds=per_class(DEFAULT_SCORE_THRESHOLDS),
        tracker=tracker_section(tracker or TrackerConfig()),
        predictor=dict(enabled=True, horizon=float(pred.horizon), step=float(pred.step),
                       merge_dist=float(pred.merge_dist)))
    if node is not None:
        out["node"] = node
    return out


def _scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        return repr(v)                             # round-trips exactly through strtod
    s = str(v)
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def _flow(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    return _scalar(v)


def to_yaml(d, indent=0):
    """Block mappings, flow sequences: enough for these parameters, no PyYAML needed."""
    pad = "  " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append("%s%s:%s" % (pad, k, "" if v else " {}"))
            if v:
                lines.append(to_yaml(v, indent + 1))
        else:
            lines.append("%s%s: %s" % (pad, k, _flow(v)))
    return "\n".join(lines)


def node_section(robot):
    from dog_perception.node_config import ins_lever_arm, node_parameters
    params = node_parameters(robot)
    params["ins_lever_arm"] = [float(v) for v in ins_lever_arm(params)]
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--robot", default=None, help="node_config.py preset ('' = generic LIO node)")
    g.add_argument("--seq", default=None, help="recorded sequence: its meta.json gives the preprocessing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    node = None
    if args.seq:
        from dog_perception.io import SequenceReader
        pre = SequenceReader(args.seq).preprocess_config()
        source = dict(seq=os.path.abspath(args.seq))
    elif args.robot is not None:
        from dog_perception.node_config import node_parameters, preprocess_config
        pre = preprocess_config(node_parameters(args.robot))
        source = dict(robot=args.robot)
        node = node_section(args.robot)
    else:
        pre, source = PreprocessConfig(), dict(defaults="dog_perception dataclasses")
    text = "# written by tools/deploy/export_cpp_config.py; edit the Python dataclasses, not this file\n"
    text += to_yaml(config_dict(pre, source=source, node=node)) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
