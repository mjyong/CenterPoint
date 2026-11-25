"""Nav_D20/planning/tools/planning_debug_bag.py on a made-up bag recorded on the dog (perception's objects under the
old perception_interface type, the goal sent in base_link and re-stamped to the start as a clip does, the planner's
own ptz frames on /tf, lidar and INS that the planner does not read, the field planner's debug topics); and
planning_mcap_viz.py leaving out a goal marker drawn at the base_link numbers (planners before the T66 review)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make

TOOL = os.path.join(ROOT, "Nav_D20", "planning", "tools")
T0 = 1794744619.4
DOG = (107.0, 209.0, math.radians(156.0))
GOAL_BASE = (27.0, 4.03, -0.0657)


def _write(path, n=50):
    sys.path.insert(0, TOOL)
    from bag_on_map import typestore
    from mcap.writer import CompressionType, Writer
    ts = typestore(all_packages=True, quiet=True)

    def time(t):
        ns = int(round(t * 1e9))
        return _make(ts, "builtin_interfaces/msg/Time", sec=ns // 10 ** 9, nanosec=ns % 10 ** 9)

    def header(t, frame):
        return _make(ts, "std_msgs/msg/Header", stamp=time(t), frame_id=frame)

    def point(x, y, z=0.0):
        return _make(ts, "geometry_msgs/msg/Point", x=float(x), y=float(y), z=float(z))

    def vec(x=0.0, y=0.0, z=0.0):
        return _make(ts, "geometry_msgs/msg/Vector3", x=float(x), y=float(y), z=float(z))

    def quat(yaw):
        return _make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))

    def pose(x, y, yaw=0.0):
        return _make(ts, "geometry_msgs/msg/Pose", position=point(x, y), orientation=quat(yaw))

    def tr(t, parent, child, x, y, yaw):
        return _make(ts, "geometry_msgs/msg/TransformStamped", header=header(t, parent), child_frame_id=child,
                     transform=_make(ts, "geometry_msgs/msg/Transform", translation=vec(x, y), rotation=quat(yaw)))

    static_qos = ("- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 9223372036\n"
                  "    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n"
                  "  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n"
                  "  avoid_ros_namespace_conventions: false")      # as the clip's /tf_static
    channels = {}
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.ZSTD)
        w.start(profile="ros2", library="test")

        def put(topic, typename, msg, t, qos=""):
            if topic not in channels:
                sid = w.register_schema(typename, "ros2msg", ts.generate_msgdef(typename, ros_version=2)[0].encode())
                channels[topic] = w.register_channel(topic, "cdr", sid, metadata={"offered_qos_profiles": qos})
            ns = int(round(t * 1e9))
            w.add_message(channels[topic], ns, bytes(ts.serialize_cdr(msg, typename)), ns)

        put("/tf_static", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=[
            tr(T0, "base_link", "hesai_lidar", 0.319, 0.043, math.pi / 2)]), T0 - 0.5, qos=static_qos)
        # the goal as a clip carries it: sent in base_link, re-stamped to the start
        put("/goal_pose", "geometry_msgs/msg/PoseStamped",
            _make(ts, "geometry_msgs/msg/PoseStamped", header=header(T0, "base_link"),
                  pose=pose(GOAL_BASE[0], GOAL_BASE[1], GOAL_BASE[2])), T0)
        for i in range(n):
            t = T0 + 0.1 * i
            x, y, yaw = DOG[0] + 0.001 * i, DOG[1], DOG[2]
            put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=[
                tr(t, "map", "base_link", x, y, yaw)]), t + 0.002)
            ptz = [tr(t, "base_link", "ptz_visible", 0.27, 0.1, 0.0),
                   tr(t, "base_link", "ptz_infrared", 0.26, -0.1, 0.0)]
            put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=ptz), t + 0.05)
            put("/lightning/odom", "nav_msgs/msg/Odometry",
                _make(ts, "nav_msgs/msg/Odometry", header=header(t, "map"), child_frame_id="base_link",
                      pose=_make(ts, "geometry_msgs/msg/PoseWithCovariance", pose=pose(x, y, yaw))), t + 0.001)
            grid = _make(ts, "nav_msgs/msg/OccupancyGrid", header=header(t, "base_link"),
                         info=_make(ts, "nav_msgs/msg/MapMetaData", resolution=0.05, width=20, height=20,
                                    origin=pose(-0.5, -0.5)), data=np.zeros(400, np.int8))
            put("/map/occupancy_grid", "nav_msgs/msg/OccupancyGrid", grid, t + 0.02)
            path = _make(ts, "perception_interface/msg/LidarPredictedPath", probability=1.0,
                         positions=[point(1, 2)], covariance=np.zeros(4))
            car = _make(ts, "perception_interface/msg/LidarObject", id=1327 + i % 2, label=0, label_name="vehicle",
                        motion=1, score=0.9, coasting=False, age=0.1 * i, position=point(15.8, -8.6 + i, 0.8),
                        yaw=0.3, size=vec(12.1, 2.5, 3.5), velocity=vec(0.0, 8.5), covariance=np.arange(16.0),
                        prediction_times=np.array([1.0], np.float32), predictions=[path])
            put("/perception/lidar_3d/objects", "perception_interface/msg/LidarObjectArray",
                _make(ts, "perception_interface/msg/LidarObjectArray", header=header(t, "lidar_det"),
                      world_frame="camera_init", world_pose=pose(1, 2), objects=[car]), t + 0.04)
            put("/planning/debug/plotjuggler/string", "std_msgs/msg/String",
                _make(ts, "std_msgs/msg/String", data="crosswalk_mode: 6\n"), t + 0.05)
            gp = _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), ns="global_path_2d", id=1,
                       type=4, action=0, points=[point(106.77, 209.26), point(80.75, 216.45)])
            put("/planning/debug/path_tracking", "visualization_msgs/msg/MarkerArray",
                _make(ts, "visualization_msgs/msg/MarkerArray", markers=[gp]), t + 0.05)
            # the old planner's goal marker: the base_link numbers drawn in map
            put("/planning/debug/goal_pose", "visualization_msgs/msg/Marker",
                _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), ns="goal_pose", id=1, type=0,
                      action=0, pose=pose(GOAL_BASE[0], GOAL_BASE[1])), t + 0.05)
            put("/nav_cw/shadow/cmd_vel", "geometry_msgs/msg/Twist",
                _make(ts, "geometry_msgs/msg/Twist", linear=vec(0.0), angular=vec()), t + 0.03)
            put("/rte/rdap/rtk/imus/data_raw", "sensor_msgs/msg/Imu",
                _make(ts, "sensor_msgs/msg/Imu", header=header(t, "imu")), t)
            put("/robot/topic/rdap/top_point_cloud2", "sensor_msgs/msg/PointCloud2",
                _make(ts, "sensor_msgs/msg/PointCloud2", header=header(t, "hesai_lidar"), fields=[],
                      data=np.zeros(64, np.uint8)), t + 0.03)
        w.finish()
    return ts


def _goal_map():
    x, y, yaw = DOG
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * GOAL_BASE[0] - s * GOAL_BASE[1], y + s * GOAL_BASE[0] + c * GOAL_BASE[1], yaw + GOAL_BASE[2]


def _read(out, ts):
    from rosbags.highlevel import AnyReader
    rows = {}
    with AnyReader([Path(out)], default_typestore=ts) as r:
        types = {c.topic: c.msgtype for c in r.connections}
        for c, t, raw in r.messages():
            rows.setdefault(c.topic, []).append((t, raw, r.deserialize(raw, c.msgtype)))
    return types, rows


def test_planner_inputs_new_objects_type_field_outputs_and_the_goal_in_map(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import planning_debug_bag as pdb
    src = tmp_path / "x6_clip"
    src.mkdir()
    ts = _write(str(src / "x6_clip_0.mcap"))
    out = str(tmp_path / "planning_debug")
    assert pdb.main([str(src), "-o", out]) == 0
    text = capsys.readouterr().out
    gx, gy, gyaw = _goal_map()
    assert "goal in map: %.3f %.3f" % (gx, gy) in text and "with the pose at its stamp" in text, text
    assert "global path ends at 80.750 216.450" in text
    assert "--planning-only" in text and "ptz_infrared 50, ptz_visible 50" in text

    types, rows = _read(out, ts)          # a ros2 bag directory (metadata.yaml + mcap)
    assert types["/perception/lidar_3d/objects"] == "nav_interface/msg/LidarObjectArray"
    assert set(types) == {"/tf", "/tf_static", "/lightning/odom", "/map/occupancy_grid", "/perception/lidar_3d/objects",
                          "/goal_pose", "/field/planning/debug/plotjuggler/string",
                          "/field/planning/debug/path_tracking", "/field/planning/debug/goal_pose",
                          "/field/nav_cw/shadow/cmd_vel"}
    # the objects: the same bytes, readable as the new type, every field as it was
    from mcap.reader import make_reader
    with open(str(src / "x6_clip_0.mcap"), "rb") as f:
        old = [m.data for _, c, m in make_reader(f).iter_messages(topics=["/perception/lidar_3d/objects"])]
    new = rows["/perception/lidar_3d/objects"]
    assert [r[1] for r in new] == old
    o = new[3][2].objects[0]
    got = (o.id, o.label_name, o.size.x, o.velocity.y, o.predictions[0].positions[0].y)
    assert got == (1328, "vehicle", 12.1, 8.5, 2.0)
    assert list(o.covariance) == list(np.arange(16.0))
    # /tf without the planner's own frames; the static TF transient local as recorded
    assert all(t.child_frame_id == "base_link" for _, _, m in rows["/tf"] for t in m.transforms)
    assert len(rows["/tf"]) == 50
    meta = open(os.path.join(out, "metadata.yaml")).read()
    assert "durability: 1" in meta and "ros_distro: humble" in meta and "storage_identifier: mcap" in meta
    # one goal, in map, after the first pose
    (t_goal, _, g), = rows["/goal_pose"]
    assert g.header.frame_id == "map"
    assert abs(g.pose.position.x - gx) < 1e-6 and abs(g.pose.position.y - gy) < 1e-6
    yaw = 2 * math.atan2(g.pose.orientation.z, g.pose.orientation.w)
    assert abs((yaw - gyaw + math.pi) % (2 * math.pi) - math.pi) < 1e-6
    assert t_goal >= rows["/tf"][0][0] and t_goal >= rows["/lightning/odom"][0][0]
    assert rows["/field/planning/debug/plotjuggler/string"][0][2].data == "crosswalk_mode: 6\n"


def test_goal_given_start_end_and_sqlite(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import planning_debug_bag as pdb
    src = tmp_path / "bag"
    src.mkdir()
    ts = _write(str(src / "bag_0.mcap"))
    out = str(tmp_path / "cut")
    assert pdb.main([str(src / "bag_0.mcap"), "-o", out, "--goal", "80.72", "216.44", "152", "--start", "1.0",
                     "--end", "3.0"]) == 0
    types, rows = _read(out, ts)
    starts = [r[0] * 1e-9 for r in rows["/lightning/odom"]]
    assert 1.0 - 1e-3 <= starts[0] - T0 and starts[-1] - T0 <= 3.0 and len(starts) == 20
    (_, _, g), = rows["/goal_pose"]
    assert (round(g.pose.position.x, 2), round(g.pose.position.y, 2)) == (80.72, 216.44)
    assert len(rows["/tf_static"]) == 1 and rows["/tf_static"][0][0] * 1e-9 >= T0 + 1.0 - 1e-3  # brought to the start
    assert "from --goal" in capsys.readouterr().out
    db = tmp_path / "sqlite"
    db.mkdir()
    (db / "x_0.db3").write_bytes(b"")
    with pytest.raises(SystemExit, match="bag_to_mcap"):
        pdb.main([str(db), "-o", str(tmp_path / "o")])


def test_viz_leaves_out_a_goal_marker_at_the_base_link_numbers(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz
    src = tmp_path / "rec"
    src.mkdir()
    _write(str(src / "rec_0.mcap"))
    out = str(tmp_path / "viz.mcap")
    assert viz.main([str(src), "-o", out, "--map", "none", "--overview", "none", "--lidar-rate", "0"]) == 0
    text = capsys.readouterr().out
    assert "50 of 50 goal markers (/planning/debug/goal_pose) are more than 1 m from /goal_pose" in text, text
    import json

    from mcap.reader import make_reader
    with open(out, "rb") as f:
        dist = [json.loads(m.data)["goal_distance"] for _, _, m in make_reader(f).iter_messages(topics=["/viz/state"])]
    assert dist and all(abs(d - math.hypot(*GOAL_BASE[:2])) < 0.1 for d in dist)
