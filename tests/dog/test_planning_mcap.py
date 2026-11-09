"""Nav_D20/planning/tools/check_planning_mcap.py on two made-up planning recordings: one as planning wrote /tf and
its debug topics before 2026-11-09 (each transform with its inverse, map -> ptz_visible next to base_link ->
ptz_visible, the base_link costmap stamped with the wall clock, markers living 0.05 s at 20 Hz), one as now."""
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT

TOOL = os.path.join(ROOT, "Nav_D20", "planning", "tools")
BAG_T0 = 1758444087.0          # 2026-09-21: the replayed bag's stamps
WALL_T0 = 1762650000.0         # 2026-11-09: the wall clock while replaying it


BASE, NAME, ARRAY = 1, 2, 3        # rosbags' Nodetype values (its import path differs between versions)


def _default(ts, node, desc):
    if node == BASE:
        t = desc[0]
        return "" if t == "string" else False if t == "bool" else 0.0 if t.startswith("float") else 0
    if node == NAME:
        return _make(ts, desc)
    (sub_node, sub_desc), n = desc
    if sub_node == BASE and sub_desc[0] not in ("string",):
        return np.zeros(n if node == ARRAY else 0, dtype=sub_desc[0] if sub_desc[0] != "bool" else bool)
    return [_default(ts, sub_node, sub_desc) for _ in range(n if node == ARRAY else 0)]


def _make(ts, name, **fields):
    """A rosbags message of type name, unset fields at their defaults."""
    values = {f: fields[f] if f in fields else _default(ts, *d) for f, d in ts.fielddefs[name][1]}
    return ts.types[name](**values)


def _write(path, old, wall_stamps=None, runaway=False):
    wall_stamps = old if wall_stamps is None else wall_stamps
    from mcap.writer import CompressionType, Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)

    def time(t):
        return _make(ts, "builtin_interfaces/msg/Time", sec=int(t), nanosec=int(round((t % 1) * 1e9)))

    def header(t, frame):
        return _make(ts, "std_msgs/msg/Header", stamp=time(t), frame_id=frame)

    def tf(t, parent, child, x):
        tr = _make(ts, "geometry_msgs/msg/Transform",
                   translation=_make(ts, "geometry_msgs/msg/Vector3", x=x, y=0.0, z=0.0),
                   rotation=_make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=0.0, w=1.0))
        return _make(ts, "geometry_msgs/msg/TransformStamped", header=header(t, parent), child_frame_id=child,
                     transform=tr)

    def marker(t, frame, action, life):
        return _make(ts, "visualization_msgs/msg/Marker", header=header(t, frame), action=action, type=4,
                     lifetime=_make(ts, "builtin_interfaces/msg/Duration", sec=int(life),
                                    nanosec=int(round((life % 1) * 1e9))))

    channels = {}
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start(profile="ros2", library="test")

        def put(topic, typename, msg, log_t):
            if topic not in channels:
                sid = w.register_schema(typename, "ros2msg", ts.generate_msgdef(typename, ros_version=2)[0].encode())
                channels[topic] = w.register_channel(topic, "cdr", sid)
            ns = int(log_t * 1e9)
            w.add_message(channels[topic], ns, bytes(ts.serialize_cdr(msg, typename)), ns)

        for i in range(60):                                       # 3 s at 20 Hz
            bag_t, wall_t = BAG_T0 + 0.05 * i, WALL_T0 + 0.05 * i
            edges = [tf(bag_t, "map", "base_link", 78.0 + 0.1 * i), tf(bag_t, "base_link", "ptz_visible", 0.27)]
            if old:
                edges += [tf(bag_t, "base_link", "map", -78.0 - 0.1 * i), tf(bag_t, "ptz_visible", "base_link", -0.27),
                          tf(bag_t, "map", "ptz_visible", 78.27 + 0.1 * i),
                          tf(bag_t, "ptz_visible", "map", -78.27 - 0.1 * i)]     # the six edges the old TfBuffer sent
            put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=edges), wall_t)
            # a dog walking 1.5 m/s along x; runaway: the T41 localization, falling at 8.8 m/s^2 between corrections
            # once a second and snapping back
            k = i % 20
            pos = _make(ts, "geometry_msgs/msg/Point", x=78.0 + 0.075 * i, y=119.0,
                        z=-0.5 * 8.8 * (0.05 * k) ** 2 if runaway else 0.0)
            pose = _make(ts, "geometry_msgs/msg/Pose", position=pos,
                         orientation=_make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=0.0, w=1.0))
            put("/lightning/odom", "nav_msgs/msg/Odometry",
                _make(ts, "nav_msgs/msg/Odometry", header=header(bag_t, "map"), child_frame_id="base_link",
                      pose=_make(ts, "geometry_msgs/msg/PoseWithCovariance", pose=pose)), wall_t)
            put("/planning/debug/cost_map_production/occmap", "nav_msgs/msg/OccupancyGrid",
                _make(ts, "nav_msgs/msg/OccupancyGrid", header=header(wall_t if wall_stamps else bag_t, "base_link"),
                      data=np.zeros(4, np.int8)), wall_t)
            markers = [] if old else [marker(0.0, "", 3, 0.0)]
            markers += [marker(0.0, "map", 0, 0.05 if old else 0.5) for _ in range(3)]
            put("/planning/debug/path_tracking", "visualization_msgs/msg/MarkerArray",
                _make(ts, "visualization_msgs/msg/MarkerArray", markers=markers), wall_t + 0.01 * (i % 3))
        w.finish()


@pytest.mark.parametrize("old", [True, False])
def test_check_planning_mcap_finds_what_foxglove_cannot_draw(tmp_path, capsys, old):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import check_planning_mcap

    bag = str(tmp_path / ("old.mcap" if old else "new.mcap"))
    _write(bag, old)
    layout_json = os.path.join(ROOT, "Nav_D20", "planning", "script", "planning_io.json")
    code = check_planning_mcap.main([bag, "--layout", layout_json])
    out = capsys.readouterr().out
    problems = [line for line in out.splitlines() if line.startswith("PROBLEM ")]
    tf = [p for p in problems if "TF:" in p]
    layout = [p for p in problems if "layout:" in p]
    rest = [p for p in problems if p not in tf and p not in layout]
    assert code == 1                                               # the layout's other topics are not in either bag
    assert {p.split("layout: ")[1].split(" ")[0] for p in layout} == {
        "/fused_point_cloud", "/map/occupancy_grid", "/planning/debug/goal_pose", "/goal_pose"}
    if old:
        assert any("base_link has 2 parents" in p for p in tf) and any("map has 2 parents" in p for p in tf)
        assert any("ptz_visible has 2 parents" in p for p in tf)
        assert any("occmap" in p and "no single TF path" in p for p in rest)
        assert any("path_tracking" in p and "lifetime 0.05" in p for p in rest)
    else:
        assert tf == [] and rest == [], out


def test_check_planning_mcap_finds_odometry_no_robot_can_have(tmp_path, capsys):
    """The localization of the T41 replay: between corrections the IMU-rate prediction fell at about 8.8 m/s^2
    (the kygi IMU's acceleration in g taken as m/s^2)."""
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import check_planning_mcap

    bag = str(tmp_path / "runaway.mcap")
    _write(bag, old=False, runaway=True)
    assert check_planning_mcap.main([bag]) == 1
    problems = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PROBLEM ")]
    assert len(problems) == 1 and "/lightning/odom: not a walking robot" in problems[0] and "z -4.0 .." \
        in problems[0], problems


def test_check_planning_mcap_finds_a_costmap_on_the_wall_clock(tmp_path, capsys):
    """The TF tree fixed alone is not enough: the base_link costmap stamped with the wall clock while the bag's
    odometry (and so map -> base_link) has the September stamps cannot be placed."""
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import check_planning_mcap

    bag = str(tmp_path / "wall.mcap")
    _write(bag, old=False, wall_stamps=True)
    assert check_planning_mcap.main([bag]) == 1
    problems = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PROBLEM ")]
    assert len(problems) == 1 and "occmap (base_link): 100% of the header stamps are outside the base_link TF" \
        in problems[0], problems
