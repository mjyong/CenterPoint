"""Online node vs. training data: replaying the moving sample bag into ros2_node.py (``robot:=kygi``,
ROS replaced by stubs) must give the same det-frame clouds as the converted sequence that
auto-labelling reads."""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dog_perception.node_config import NODE_DEFAULTS, ROBOT_PRESETS, node_parameters, preprocess_config

BAG = "samples/rosbag2_2026_09_21-16_41_27_moving"
TOPICS = dict(lidar="/robot/topic/rdap/top_point_cloud2", imu="/rte/rdap/rtk/imus/data_raw",
              ins="/rte/rdap/rtk/inspvax")


def test_presets_override_with_the_declared_types():
    for robot, preset in ROBOT_PRESETS.items():
        for k, v in preset.items():
            assert k in NODE_DEFAULTS and type(v) is type(NODE_DEFAULTS[k]), (robot, k)    # rclpy types by default
            if isinstance(v, list):
                assert all(isinstance(x, float) for x in v), (robot, k)
    assert node_parameters("")["base_height"] == 0.45 and not node_parameters("")["auto_level"]
    from dog_perception.inspvax import KYGI_IMU_IN_BODY, KYGI_INS_LEVER_ARM
    from dog_perception.node_config import ins_lever_arm
    kygi = node_parameters("kygi")
    assert np.allclose(ins_lever_arm(kygi), KYGI_INS_LEVER_ARM)                       # antennas swapped (0921)
    assert np.allclose(ins_lever_arm(dict(kygi, ins_azimuth_offset_deg=0.0)), KYGI_IMU_IN_BODY)   # fixed
    assert np.allclose(ins_lever_arm(dict(kygi, ins_lever_arm=[0.1, -0.2, 0.0])), [0.1, -0.2, 0.0])
    # the front Mid-360: its time format, noise tags, mounting and self filter, antennas fixed
    from dog_perception.lidars import LIDARS
    mid = node_parameters("kygi_mid360")
    assert mid["time_mode"] == "absolute_ns" and mid["noise_tag_mask"] == 3 and mid["ins_azimuth_offset_deg"] == 0.0
    cfg = preprocess_config(mid)
    assert cfg.self_filter.body_boxes == tuple(tuple(b) for b in LIDARS["mid360"].self_filter["body_boxes"])
    assert np.allclose(cfg.T_body_lidar[:3, :3] @ [0.304, -0.025, -0.952], [0, 0, 1], atol=1e-3)   # ground normal -> up
    assert preprocess_config(kygi).self_filter == preprocess_config(node_parameters("")).self_filter
    with pytest.raises(ValueError):
        node_parameters("spot")


def _stub_ros(monkeypatch, overrides):
    class Param:
        def __init__(self, value):
            self.value = value

    class Logger:
        def __getattr__(self, name):
            return lambda *a, **k: None

    class Node:
        def __init__(self, name):
            self.subs = {}

        def declare_parameter(self, name, default):
            return Param(overrides.get(name, default))

        def get_logger(self):
            return Logger()

        def create_subscription(self, msg_type, topic, cb, qos):
            self.subs[topic] = cb

        def create_publisher(self, *a):
            return types.SimpleNamespace(publish=lambda m: None)

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    dummy = type("Msg", (), {})
    module("rclpy", init=lambda: None, spin=lambda n: None, shutdown=lambda: None)
    module("rclpy.node", Node=Node)
    module("rclpy.qos", QoSProfile=lambda **k: None, ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=0))
    module("geometry_msgs.msg", Point=dummy)
    module("nav_msgs.msg", Odometry=dummy)
    module("sensor_msgs.msg", Imu=dummy, PointCloud2=dummy, PointField=dummy)
    module("visualization_msgs.msg", Marker=dummy, MarkerArray=dummy)
    module("kygi665_msgs.msg", INSPVAX=dummy)


def _online_frames(monkeypatch):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    import dog_perception.detection
    from dog_perception.inspvax import register_kygi

    _stub_ros(monkeypatch, dict(robot="kygi"))
    monkeypatch.setattr(dog_perception.detection, "build_detector", lambda *a, **k: None)
    spec = importlib.util.spec_from_file_location("ros2_node_under_test", "tools/dog/ros2_node.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    node = mod.DogPerceptionNode()
    assert set(node.subs) == set(TOPICS.values())

    frames = {}

    def on_scan(scan, detections=None):
        fr = node.pipe.pre.process(scan)
        if fr is not None:
            frames[round(fr.stamp, 6)] = fr
        return None

    node.pipe.on_scan = on_scan
    ts = register_kygi(get_typestore(Stores.ROS2_HUMBLE))
    with AnyReader([Path(BAG)], default_typestore=ts) as r:      # bag (arrival) order, as online
        for c, _, raw in r.messages(connections=[c for c in r.connections if c.topic in TOPICS.values()]):
            node.subs[c.topic](r.deserialize(raw, c.msgtype))
    return frames


def _offline_frames(seq_dir):
    from dog_perception.io import SequenceReader
    from dog_perception.pipeline import PerceptionPipeline
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig

    rd = SequenceReader(seq_dir)
    pcfg = PreprocessConfig(T_body_lidar=rd.T_body_lidar,                  # as tools/dog/auto_label.py
                            det_frame=DetFrameConfig(base_height=rd.meta["base_height"], **rd.meta["det_frame"]))
    pipe = PerceptionPipeline(pcfg, detector=None)
    frames = {}
    for ev in rd.events():
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            fr = pipe.pre.process(ev[1])
            if fr is not None:
                frames[round(fr.stamp, 6)] = fr
    return rd.meta, pcfg, frames


@pytest.mark.slow
def test_kygi_node_matches_the_training_sequences(monkeypatch, tmp_path):
    pytest.importorskip("rosbags")
    if not os.path.isdir(BAG):
        pytest.skip("sample bag not checked out")
    subprocess.run([sys.executable, "tools/dog/rosbag_to_sequence.py", "--bag", BAG, "--out", str(tmp_path / "seq"),
                    "--lidar-topic", TOPICS["lidar"], "--imu-topic", TOPICS["imu"], "--inspvax-topic", TOPICS["ins"]],
                   check=True, capture_output=True)
    meta, pcfg, offline = _offline_frames(str(tmp_path / "seq"))
    assert meta["base_height_sweeps"]["n"] == 3 and abs(meta["base_height"] - 0.72) < 0.03

    node_cfg = preprocess_config(node_parameters("kygi"))
    assert np.allclose(node_cfg.T_body_lidar, pcfg.T_body_lidar)
    assert (node_cfg.det_frame.auto_ground, node_cfg.det_frame.auto_level) == (True, True) == \
        (pcfg.det_frame.auto_ground, pcfg.det_frame.auto_level)

    online = _online_frames(monkeypatch)
    common = sorted(set(online) & set(offline))
    assert len(common) >= len(offline) - 1 >= 25
    err = []
    for t in common:
        a, b = online[t], offline[t]
        assert np.allclose(a.T_world_det[:3, :3], b.T_world_det[:3, :3], atol=1e-3)
        cur_a, cur_b = a.points[a.points[:, 4] == 0], b.points[b.points[:, 4] == 0]
        assert len(cur_a) == len(cur_b)
        err.append(np.median(np.hypot(*(cur_a[:, :2] - cur_b[:, :2]).T)))
        # base height 0.72 (node) vs the sequence median: the ground estimate pulls both onto the ground
        assert abs(np.median(cur_a[:, 2] - cur_b[:, 2])) < 0.01
    # Same processing; only the message order differs (bag arrival vs stamps). Where the INS
    # position steps back (every ~0.2 s while slowing down, up to 0.33 m in this bag) the node has
    # the next INS sample and interpolates, the sequence replay extrapolates: a few cm on those frames.
    err = np.array(err)
    assert np.mean(err < 0.005) >= 0.6 and err.max() < 0.1, err.round(4)
    print(json.dumps(dict(frames=len(common), base_height_sequence=meta["base_height"],
                          frames_within_5mm=float(np.mean(err < 0.005)))))


@pytest.mark.slow
def test_a_bag_played_again_starts_over():
    """The stamps jump back: the online inputs start over as a new node, and the second pass gives the
    first pass's frames and tracks again (otherwise the pose buffer drops the new poses as out of order).
    The first pass's last cloud arriving after the second pass's first message is dropped."""
    pytest.importorskip("rosbags")
    if not os.path.isdir(BAG):
        pytest.skip("sample bag not checked out")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "deploy"))
    from message_log import bag_messages
    from dog_perception.detection.boxes import Detections
    from dog_perception.online import OnlineInputs
    from dog_perception.pipeline import PerceptionPipeline

    params = node_parameters("kygi")
    msgs = list(bag_messages(BAG, params))
    pipe = PerceptionPipeline(preprocess_config(params), None, predictor="imm")
    warned = []
    inputs = OnlineInputs(params, pipe, lambda level, text, throttle=0.0: level == "warn" and warned.append(text))
    car = Detections(np.array([[8.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0]]), np.zeros((1, 2)), np.array([0.9]),
                     np.array([0]), frame="det")          # one car 8 m ahead in every frame: a track to carry over
    last_cloud = [m for k, m in msgs if k == "cloud"][-1]
    assert msgs[0][0] != "cloud"
    passes = []
    for p in range(2):
        frames = []
        for i, (kind, m) in enumerate(msgs):
            if p == 1 and i == 1:                                 # late, from before the jump
                assert inputs.on_cloud(last_cloud, detections=car) is None
            if kind == "cloud":
                out = inputs.on_cloud(m, detections=car)
                if out is not None:
                    frames.append((out.stamp, out.frame.T_world_det, out.frame.points,
                                   [(s.track_id, s.age, tuple(s.position)) for s in out.tracks]))
            else:
                getattr(inputs, "on_" + kind)(m)
        passes.append(frames)
    assert [w.split(" ")[1] for w in warned] == ["stamps", "message"], warned
    assert len(passes[0]) == len(passes[1]) == 27 and passes[0][-1][3]
    for a, b in zip(*passes):
        assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and a[3] == b[3]
