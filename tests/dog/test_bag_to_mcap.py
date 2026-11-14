"""Nav_D20/planning/tools/bag_to_mcap.py: a sqlite3 bag as recorded on dog 8 (its ros2 bag has no mcap plugin, T61)
becomes mcap with every topic, including the robot team's and our own message types."""
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make

TOOLS = os.path.join(ROOT, "Nav_D20", "planning", "tools")


@pytest.fixture
def tool():
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOLS)
    try:
        import bag_on_map
        import bag_to_mcap
        yield bag_to_mcap, bag_on_map
    finally:
        sys.path.remove(TOOLS)


def test_every_topic_goes_over(tmp_path, tool, capsys):
    bag_to_mcap, bag_on_map = tool
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import Writer
    ts = bag_on_map.typestore(all_packages=True, quiet=True)
    stamp = _make(ts, "builtin_interfaces/msg/Time", sec=1700000000, nanosec=0)
    header = _make(ts, "std_msgs/msg/Header", stamp=stamp, frame_id="x")
    msgs = {
        "/lightning/odom": ("nav_msgs/msg/Odometry", _make(ts, "nav_msgs/msg/Odometry", header=header)),
        "/perception/lidar_3d/objects": ("perception_interface/msg/LidarObjectArray",
                                         _make(ts, "perception_interface/msg/LidarObjectArray", header=header)),
        "/rte/rdap/rtk/inspvax": ("kygi665_msgs/msg/INSPVAX", _make(ts, "kygi665_msgs/msg/INSPVAX")),
        "/mc/motion_control": ("mc_interface/msg/Motion",
                               _make(ts, "mc_interface/msg/Motion", header=header, uuid="planning", mode=1, motion=2,
                                     para=np.array([0.4, 0.0, 0.1], dtype=np.float32))),
    }
    src = tmp_path / "rec"
    with Writer(src, version=8) as w:                       # sqlite3, as on the dog
        for k, (topic, (msgtype, m)) in enumerate(sorted(msgs.items())):
            c = w.add_connection(topic, msgtype, typestore=ts)
            for i in range(3):
                w.write(c, 1700000000 * 10 ** 9 + i * 10 ** 8 + k, ts.serialize_cdr(m, msgtype))
    assert bag_to_mcap.main([str(src), str(tmp_path / "out")]) == 0
    assert "4 topics, 12 messages" in capsys.readouterr().out
    out = tmp_path / "out"
    assert any(f.suffix == ".mcap" for f in out.iterdir())
    with AnyReader([out]) as r:                              # the mcap carries its own definitions
        topics = {c.topic: c.msgtype for c in r.connections}
        assert topics == {t: v[0] for t, v in msgs.items()}
        motion = [r.deserialize(raw, c.msgtype) for c, _, raw in r.messages() if c.topic == "/mc/motion_control"]
    assert len(motion) == 3 and motion[0].uuid == "planning" and abs(motion[0].para[0] - 0.4) < 1e-6
