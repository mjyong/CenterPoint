"""Nav_D20/planning/tools/objects_in_map.py: parked cars stay put in the map with the right extrinsic. A made-up
db3 bag: the dog walks and turns (odometry), perception sees three parked cars and one moving car in its own frame
(x right, y forward, under the lidar), placed with the D20 extrinsic the planner uses."""
import math
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make

TOOLS = os.path.join(ROOT, "Nav_D20", "planning", "tools")
LIDAR = (0.319, 0.043)


@pytest.fixture
def tool():
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOLS)
    try:
        import objects_in_map
        yield objects_in_map
    finally:
        sys.path.remove(TOOLS)


def _pose(t):
    """Walks east at 0.8 m/s for 10 s, turns left 90 deg on the spot over 5 s, walks north."""
    if t < 10:
        return 0.8 * t, 0.0, 0.0
    if t < 15:
        return 8.0, 0.0, math.radians(18.0 * (t - 10))
    return 8.0, 0.8 * (t - 15), math.pi / 2


def _det(pose, xy_map, det_yaw_deg):
    """Map point -> perception frame (the inverse of the planner's transform), for the true extrinsic."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy_map[0] - x, xy_map[1] - y
    bx, by = c * dx + s * dy - LIDAR[0], -s * dx + c * dy - LIDAR[1]
    cd, sd = math.cos(math.radians(det_yaw_deg)), math.sin(math.radians(det_yaw_deg))
    return cd * bx + sd * by, -sd * bx + cd * by


def _bag(path, ts, det_yaw_deg=-90.0, gap=None, parked_flag=0, pose=None):
    pose_of = pose or _pose
    from rosbags.rosbag2 import Writer
    parked = [(5.0, 6.0), (12.0, -4.0), (4.0, 12.0)]

    def stamp(t):
        return _make(ts, "builtin_interfaces/msg/Time", sec=int(t), nanosec=int(round((t % 1) * 1e9)))

    with Writer(path, version=8) as w:
        co = w.add_connection("/lightning/odom", "nav_msgs/msg/Odometry", typestore=ts)
        cp = w.add_connection("/perception/lidar_3d/objects", "perception_interface/msg/LidarObjectArray",
                              typestore=ts)
        t0 = 1.7e9
        for k in range(25 * 50):        # odometry 50 Hz, 25 s
            t = k * 0.02
            if gap and gap[0] <= t < gap[1]:
                continue
            x, y, yaw = pose_of(t)
            msg_pose = _make(ts, "geometry_msgs/msg/Pose",
                         position=_make(ts, "geometry_msgs/msg/Point", x=x, y=y, z=0.0),
                         orientation=_make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=math.sin(yaw / 2),
                                           w=math.cos(yaw / 2)))
            m = _make(ts, "nav_msgs/msg/Odometry",
                      header=_make(ts, "std_msgs/msg/Header", stamp=stamp(t0 + t), frame_id="map"),
                      pose=_make(ts, "geometry_msgs/msg/PoseWithCovariance", pose=msg_pose))
            w.write(co, int(round((t0 + t) * 1e9)), ts.serialize_cdr(m, "nav_msgs/msg/Odometry"))
        for k in range(25 * 10):        # perception 10 Hz
            t = k * 0.1
            pose = pose_of(t)
            objs = []
            for i, p in enumerate(parked + [(-10.0 + 3.0 * t, 3.0)]):
                px, py = _det(pose, p, det_yaw_deg)
                objs.append(_make(ts, "perception_interface/msg/LidarObject", id=i + 1, label=0,
                                  motion=1 if i == 3 else parked_flag,
                                  position=_make(ts, "geometry_msgs/msg/Point", x=px, y=py, z=0.0)))
            a = _make(ts, "perception_interface/msg/LidarObjectArray",
                      header=_make(ts, "std_msgs/msg/Header", stamp=stamp(t0 + t), frame_id="lidar_3d"), objects=objs)
            w.write(cp, int(round((t0 + t) * 1e9)), ts.serialize_cdr(a, "perception_interface/msg/LidarObjectArray"))


def test_the_d20_extrinsic_keeps_parked_cars_still(tmp_path, capsys, tool):
    _bag(tmp_path / "rec" / "rec_0", _ts())             # a replay directory: the recording inside it
    assert tool.main([str(tmp_path / "rec")]) == 0
    out = capsys.readouterr().out
    assert "3 static vehicle tracks" in out
    line = next(l for l in out.splitlines() if "(configured)" in l)
    assert " -90.0 deg" in line and "<- best" in line and "median 0.00 m" in line and "3 of 3 still" in line
    assert "extrinsic OK" in out


def test_a_wrong_extrinsic_is_reported(tmp_path, capsys, tool):
    _bag(tmp_path / "rec", _ts(), det_yaw_deg=90.0)           # perception's frame turned the other way
    assert tool.main([str(tmp_path / "rec")]) == 2
    out = capsys.readouterr().out
    assert "extrinsic WRONG" in out
    line = next(l for l in out.splitlines() if "<- best" in l)
    assert "  90.0 deg" in line


def test_messages_without_odometry_count_as_stale(tmp_path, tool):
    _bag(tmp_path / "rec", _ts(), gap=(3.0, 5.0))
    odom, frames = tool.read(str(tmp_path / "rec"), "/lightning/odom")
    r = tool.check(odom, frames)
    # odometry missing 3.0..5.0 s: the messages at 3.3..4.6 s are more than 0.3 s from any sample
    assert 13 <= r["stale"] <= 15
    assert r["wander"][-90.0] < 0.01


def _ts():
    sys.path.insert(0, TOOLS)
    from bag_on_map import typestore
    return typestore()


def test_parked_cars_flagged_moving_need_any_motion(tmp_path, capsys, tool):
    """Objects computed with a wrong INS heading: parked cars look moving while the dog walks (1112)."""
    _bag(tmp_path / "rec", _ts(), parked_flag=1)
    assert tool.main([str(tmp_path / "rec")]) == 1
    assert "nothing to check (try --any-motion)" in capsys.readouterr().out
    assert tool.main([str(tmp_path / "rec"), "--any-motion"]) == 0
    out = capsys.readouterr().out
    assert "4 vehicle tracks, any motion flag" in out       # the moving car too: still under no yaw
    line = next(l for l in out.splitlines() if "(configured)" in l)
    assert "3 of 4 still" in line and "<- best" in line
    assert "extrinsic OK" in out


def test_a_localization_jump_is_not_moving(tmp_path, capsys, tool):
    """The dog stands still; its odometry starts 140 m away for 0.2 s (the initialization, T54): nothing to check,
    not a verdict from tracks that cannot tell the yaws apart."""
    def standing(t):
        return (140.0, 0.0, 0.0) if t < 0.2 else (0.0, 0.0, 0.1)
    _bag(tmp_path / "rec", _ts(), pose=standing)
    assert tool.main([str(tmp_path / "rec")]) == 1
    assert "nothing to check" in capsys.readouterr().out
