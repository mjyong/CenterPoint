"""Nav_D20/planning/tools/planning_mcap_viz.py (and check_planning_mcap.py's costmap polarity) on a made-up planning
recording: a dog walking 1 m/s on a diagonal road, logged on a wall clock 52 days after the bag's stamps, the
localization's first pose at the origin, planning's markers with stamp 0, its costmap published either way round,
perception's grid with one obstacle, a route-network map and a crosswalk."""
import json
import math
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make

TOOL = os.path.join(ROOT, "Nav_D20", "planning", "tools")
BAG_T0 = 1789980090.0          # 2026-09-21
OFFSET = 4504296.0             # the replay server's clock, 52 days later
YAW = 0.3                      # the road's heading
SPEED = 1.0                    # [m/s]
START = (78.0, 119.0)


def _pose(t):
    s = SPEED * (t - BAG_T0)
    return START[0] + s * math.cos(YAW), START[1] + s * math.sin(YAW)


def _write(path, inverted):
    from mcap.writer import CompressionType, Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)

    def time(t):
        ns = int(round(t * 1e9))
        return _make(ts, "builtin_interfaces/msg/Time", sec=ns // 10 ** 9, nanosec=ns % 10 ** 9)

    def header(t, frame):
        return _make(ts, "std_msgs/msg/Header", stamp=time(t), frame_id=frame)

    def point(x, y, z=0.0):
        return _make(ts, "geometry_msgs/msg/Point", x=float(x), y=float(y), z=float(z))

    def quat(yaw, roll=0.0):
        cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
        return _make(ts, "geometry_msgs/msg/Quaternion", x=sr * cy, y=sr * sy, z=cr * sy, w=cr * cy)   # yaw, then roll

    def pose(x, y, z=0.0, yaw=0.0, roll=0.0):
        return _make(ts, "geometry_msgs/msg/Pose", position=point(x, y, z), orientation=quat(yaw, roll))

    channels = {}
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start(profile="ros2", library="test")

        def put(topic, typename, msg, bag_t, lag=0.03):
            if topic not in channels:
                sid = w.register_schema(typename, "ros2msg", ts.generate_msgdef(typename, ros_version=2)[0].encode())
                channels[topic] = w.register_channel(topic, "cdr", sid)
            ns = int((bag_t + OFFSET + lag) * 1e9)
            w.add_message(channels[topic], ns, bytes(ts.serialize_cdr(msg, typename)), ns)

        def grid(t, frame, data, origin=(-2.0, -2.0)):
            info = _make(ts, "nav_msgs/msg/MapMetaData", resolution=0.1, width=data.shape[1], height=data.shape[0],
                         origin=pose(*origin))
            return _make(ts, "nav_msgs/msg/OccupancyGrid", header=header(t, frame), info=info,
                         data=data.astype(np.int8).ravel())

        # the localization's first pose before it initialises: at the origin
        tr0 = _make(ts, "geometry_msgs/msg/TransformStamped", header=header(BAG_T0 - 0.01, "map"),
                    child_frame_id="base_link",
                    transform=_make(ts, "geometry_msgs/msg/Transform",
                                    translation=_make(ts, "geometry_msgs/msg/Vector3", x=0.0, y=0.0, z=0.0),
                                    rotation=quat(0.0)))
        put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=[tr0]), BAG_T0 - 0.01)
        for i in range(80):                         # 4 s at 20 Hz
            t = BAG_T0 + 0.05 * i
            x, y = _pose(t)
            roll = 0.05 * math.sin(7 * t)            # the dog rocks while walking
            tr = _make(ts, "geometry_msgs/msg/TransformStamped", header=header(t, "map"), child_frame_id="base_link",
                       transform=_make(ts, "geometry_msgs/msg/Transform",
                                       translation=_make(ts, "geometry_msgs/msg/Vector3", x=x, y=y, z=-0.21),
                                       rotation=quat(YAW, roll)))
            put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=[tr]), t)
            twist = _make(ts, "geometry_msgs/msg/TwistWithCovariance",
                          twist=_make(ts, "geometry_msgs/msg/Twist",
                                      linear=_make(ts, "geometry_msgs/msg/Vector3", x=SPEED, y=0.0, z=0.0)))
            put("/lightning/odom", "nav_msgs/msg/Odometry",
                _make(ts, "nav_msgs/msg/Odometry", header=header(t, "map"), child_frame_id="base_link",
                      pose=_make(ts, "geometry_msgs/msg/PoseWithCovariance", pose=pose(x, y, -0.21, YAW)),
                      twist=twist), t)
            # perception: one obstacle 1.0-1.3 m ahead, 1.0-1.3 m to the left
            perc = np.zeros((40, 40))
            perc[30:33, 30:33] = 100
            if i % 2 == 0:
                put("/map/occupancy_grid", "nav_msgs/msg/OccupancyGrid", grid(t - 0.1, "base_link", perc), t - 0.1)
            # planning's costmap: the road (|y| < 0.85 m) free, the rest and perception's obstacle occupied
            cost = np.full((40, 60), 100)
            ys = -2.0 + (np.arange(40) + 0.5) * 0.1
            cost[np.abs(ys) < 0.85, :] = 0
            cost[30:33, 30:33] = 100
            if inverted:
                cost = 100 - cost
            put("/planning/debug/cost_map_production/occmap", "nav_msgs/msg/OccupancyGrid",
                grid(t, "base_link", cost), t)
            # planning's markers, stamp 0 as before the T47 review
            gx, gy = _pose(BAG_T0 + 6.0)
            path = _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, "map"), ns="global_path_2d", id=1,
                         type=4, action=0, points=[point(*_pose(BAG_T0)), point(gx, gy)],
                         scale=_make(ts, "geometry_msgs/msg/Vector3", x=0.1, y=0.0, z=0.0))
            body = _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, "map"), ns="robot_footprint", id=2,
                         type=4, action=0, points=[point(x, y), point(x + 0.5, y)])
            contact = _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, "base_link"), ns="contact", id=3,
                            type=2, action=0, pose=pose(0.5, 0.0))
            clear = _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, ""), action=3)
            ref = _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, "map"), ns="local_reference_2d", id=4,
                        type=4, action=0, points=[point(*_pose(t + k)) for k in (0.0, 1.0, 2.0)])
            put("/planning/debug/path_tracking", "visualization_msgs/msg/MarkerArray",
                _make(ts, "visualization_msgs/msg/MarkerArray", markers=[clear, path, body, contact, ref]), t)
            put("/planning/debug/goal_pose", "visualization_msgs/msg/Marker",
                _make(ts, "visualization_msgs/msg/Marker", header=header(0.0, "map"), ns="goal_pose", id=1, type=0,
                      action=0, pose=pose(gx, gy, 0.0, YAW)), t)
            status = 2 if i < 20 else 3                 # standby, then active after 1 s
            text = "planning: %.6f\nmission_type: 1.000000\nmission_status: %d.000000\ncmd_vx: 0.800000\n" \
                   "cmd_vy: 0.000000\ncmd_omega: 0.100000\nmission_input_timeout: 0.000000\ntimeout: \n" % (
                       t + OFFSET, status)
            put("/planning/debug/plotjuggler/string", "std_msgs/msg/String",
                _make(ts, "std_msgs/msg/String", data=text), t)
        w.finish()


def _map(tmp_path):
    """A 0.1 m route-network map around the run: the road white (free), the rest black, and one crosswalk."""
    folder = tmp_path / "site" / "2d" / "1780551612"
    folder.mkdir(parents=True)
    w, h, ox, oy = 400, 400, 60.0, 100.0
    img = np.zeros((h, w), np.uint8)
    xs = ox + (np.arange(w) + 0.5) * 0.1
    ys = oy + (h - np.arange(h) - 0.5) * 0.1          # row 0 is the top
    X, Y = np.meshgrid(xs, ys)
    d = np.abs(-(X - START[0]) * math.sin(YAW) + (Y - START[1]) * math.cos(YAW))
    img[d < 0.85] = 255
    with open(folder / "map.pgm", "wb") as f:
        f.write(b"P5\n# made up\n%d %d\n255\n" % (w, h) + img.tobytes())
    (folder / "map.yaml").write_text("image: map.pgm\nmode: trinary\nresolution: 0.1\norigin: [%.1f, %.1f, 0]\n"
                                     "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n" % (ox, oy))
    cx, cy = _pose(BAG_T0 + 2.0)
    corners = [(cx - 1, cy - 4), (cx + 1, cy - 4), (cx + 1, cy + 4), (cx - 1, cy + 4)]
    nodes = "".join('<node id="%d" lat="0" lon="0"><tag k="local_x" v="%.3f"/><tag k="local_y" v="%.3f"/></node>'
                    % (i + 1, x, y) for i, (x, y) in enumerate(corners))
    way = '<way id="10">' + "".join('<nd ref="%d"/>' % (i + 1) for i in range(4)) + '<nd ref="1"/></way>'
    rel = ('<relation id="20"><member type="way" ref="10" role="outer"/><tag k="type" v="multipolygon"/>'
           '<tag k="subtype" v="crosswalk"/></relation>')
    (folder / "semantic.osm").write_text('<?xml version="1.0"?><osm version="0.6">%s%s%s</osm>' % (nodes, way, rel))
    return str(tmp_path / "site")


def _read(path):
    from mcap.reader import make_reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    out = {}
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(log_time_order=True):
            if channel.message_encoding == "json":
                msg = json.loads(message.data)
            else:
                msg = ts.deserialize_cdr(message.data, schema.name)
            out.setdefault(channel.topic, []).append((message.log_time * 1e-9, msg))
    return out


def _stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


@pytest.mark.parametrize("inverted", [True, False])
def test_planning_mcap_viz_flattens_retimes_and_turns_the_costmap_over(tmp_path, capsys, inverted):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    bag = tmp_path / "rec"
    bag.mkdir()
    _write(str(bag / "rec_0.mcap"), inverted)
    out = str(tmp_path / "viz.mcap")
    assert viz.main([str(bag), "-o", out, "--map", _map(tmp_path), "--overview", "none"]) == 0
    text = capsys.readouterr().out
    assert ("inverted, free = 100: turned over" in text) == inverted, text
    msgs = _read(out)

    # one time base: the bag's stamps, not the server's clock
    for topic, rows in msgs.items():
        for log, _ in rows:
            assert BAG_T0 - 1 < log < BAG_T0 + 10, (topic, log)

    # /tf: one edge, flat, without the start-up pose at the origin
    tf = msgs["/tf"]
    assert len(tf) == 80
    for log, m in tf:
        (t,) = m.transforms
        assert (t.header.frame_id, t.child_frame_id) == ("map", "base_link")
        assert t.transform.translation.z == 0.0 and abs(_yaw(t.transform.rotation) - YAW) < 1e-6
        assert abs(_stamp(t.header) - log) < 1e-6
        assert t.transform.rotation.x == 0.0 and t.transform.rotation.y == 0.0

    # costmap: in map at its layer height, placed with the pose at its stamp, obstacles 100 either way round
    cost = msgs["/viz/cost_map"]
    assert len(cost) == 80
    for log, g in cost[::10]:
        assert g.header.frame_id == "map" and abs(_stamp(g.header) - log) < 1e-6
        o = g.info.origin
        assert o.position.z == viz.Z["cost_map"]
        x, y = _pose(log)
        assert abs(o.position.x - (x - 2 * math.cos(YAW) + 2 * math.sin(YAW))) < 1e-6
        assert abs(o.position.y - (y - 2 * math.sin(YAW) - 2 * math.cos(YAW))) < 1e-6
        d = np.asarray(g.data).reshape(40, 60)
        assert (d[30:33, 30:33] == 100).all() and d[20, 20] == 0     # perception's obstacle, the road
    perc = msgs["/viz/perception_grid"]
    assert perc[0][1].info.origin.position.z == viz.Z["perception"] > viz.Z["cost_map"]

    # planning's markers: stamped, no lifetime, in map; the planner's own robot drawing is left to /viz/robot;
    # the base_link contact lands where the dog is, not at the first TF
    log, arr = msgs["/viz/plan"][40]
    assert arr.markers[0].action == 3
    ns = {m.ns for m in arr.markers[1:]}
    assert "global_path_2d" in ns and "goal" in ns and "robot_footprint" not in ns
    for m in arr.markers[1:]:
        assert m.header.frame_id == "map" and _stamp(m.header) > BAG_T0 - 1
        assert m.lifetime.sec == 0 and m.lifetime.nanosec == 0 and not m.frame_locked
    contact = [m for m in arr.markers if m.ns == "contact"][0]
    x, y = _pose(log)
    assert abs(contact.pose.position.x - (x + 0.5 * math.cos(YAW))) < 0.05
    robot = msgs["/viz/robot"][20][1]
    assert all(m.header.frame_id == "base_link" and m.frame_locked for m in robot.markers[1:])
    assert any(m.ns == "label" and "跟踪 active" in m.text for m in robot.markers)

    # state and events
    states = [m for _, m in msgs["/viz/state"]]
    assert states[0]["mission"] == "tracking/standby" and states[-1]["mission"] == "tracking/active"
    assert abs(states[-1]["cmd_v"] - 0.8) < 1e-6 and abs(states[-1]["odom_v"] - SPEED) < 1e-6
    assert states[-1]["cost_map_footprint_cells"] == 0          # on the road, once turned over
    assert abs(states[-1]["goal_distance"] - SPEED * (6.0 - 3.95)) < 0.06
    events = [m.msg for _, m in msgs["/viz/events"]]
    assert any("跟踪 standby → 跟踪 active" in e for e in events), events
    # the planning flow, step by step with this cycle's numbers; the legend
    flow = msgs["/viz/flow"][-1][1]
    steps = [k for k in flow if k[0] in "①②③④⑤⑥"]
    assert len(steps) == 6 and flow["④ 任务状态机（Tracking）"]["状态"] == "active"
    local = [v for k, v in flow.items() if k.startswith("⑤")][0]
    assert local["参考路径（MPPI 的输入）点数"] == "3" and local["参考路径长_m"] == "2.00"
    assert "向前 0.80 m/s" in flow["⑥ 输出（/cmd_vel）"]["指令"]
    assert "/viz/legend" in msgs and "图层（3D 面板左边的话题列表可以逐个开关）" in msgs["/viz/legend"][0][1]
    assert any(m.ns == "local_reference_2d" for m in msgs["/viz/plan"][40][1].markers)
    assert any("位姿跳了" in e and "定位初始化" in e for e in events), events

    # the map around the run and the crosswalk
    route = msgs["/viz/route_map"][0][1]
    assert set(np.unique(np.asarray(route.data))) == {0, 100}
    assert route.info.origin.position.z == viz.Z["route"]
    walk = msgs["/viz/crosswalks"][0][1]
    fill = [m for m in walk.markers if m.ns == "crosswalk_fill"][0]
    assert fill.type == 11 and len(fill.points) == 6           # a quad as two triangles


@pytest.mark.parametrize("inverted", [True, False])
def test_check_planning_mcap_tells_an_inverted_costmap(tmp_path, capsys, inverted):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import check_planning_mcap

    bag = str(tmp_path / "rec.mcap")
    _write(bag, inverted)
    check_planning_mcap.main([bag])
    out = capsys.readouterr().out
    problems = [line for line in out.splitlines() if line.startswith("PROBLEM ")]
    assert any("inverted, free cells 100" in p for p in problems) == inverted, problems
    assert not any("stands on an obstacle" in p for p in problems), problems
    assert "NOTE display: /map/occupancy_grid, /planning/debug/cost_map_production/occmap" in out


def test_planning_mcap_viz_reads_maps_and_polygons(tmp_path):
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    img = viz.read_pgm(b"P5\n# comment\n3 2\n255\n" + bytes([0, 128, 255, 10, 20, 30]))
    assert img.tolist() == [[0, 128, 255], [10, 20, 30]]
    img = viz.read_pgm(b"P2\n3 1\n15\n0 15 5\n")
    assert img.tolist() == [[0, 255, 85]]
    # an L shape: the triangles cover its area exactly
    poly = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], float)
    tris = viz.triangulate(poly)
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    area = sum(abs(cross(poly[b] - poly[a], poly[c] - poly[a])) / 2 for a, b, c in tris)
    assert len(tris) == 4 and abs(area - 3.0) < 1e-9
    assert abs(viz.crosswalk_length(np.array([(0, 0), (12, 0), (12, 5), (0, 5)], float)) - 12) < 1e-9
    # a map crop: free (white) cells become 100, rows from the bottom
    meta = dict(img=np.array([[255, 0], [0, 255]], np.uint8), resolution=1.0, origin=[10.0, 20.0, 0.0], negate=0,
                free_thresh=0.25, occupied_thresh=0.65)
    data, ox, oy, w, h = viz.crop_route(meta, 0, 0, 100, 100)
    assert (ox, oy, w, h) == (10.0, 20.0, 2, 2) and data.tolist() == [[0, 100], [100, 0]]


def test_planning_mcap_viz_reads_the_image_planning_reads(tmp_path):
    """A site map directory: global_map.yaml's first floor names the image (planning's ReloadMap), not map.yaml's
    image; a <name>.yaml given directly: <name>.pgm beside it (GlobalMapManager reads input_name + ".pgm")."""
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    floor = tmp_path / "site" / "2d" / "7"
    floor.mkdir(parents=True)

    def pgm(name, value):
        (floor / name).write_bytes(b"P5\n2 1\n255\n" + bytes([value, value]))

    pgm("map.pgm", 0)
    pgm("edited.pgm", 255)
    (floor / "map.yaml").write_text("image: map.pgm\nresolution: 0.5\norigin: [1, 2, 0]\nnegate: 0\n"
                                    "free_thresh: 0.25\noccupied_thresh: 0.65\n")
    (tmp_path / "site" / "global_map.yaml").write_text(
        "floors:\n  - floor_id: 1\n    map_image_file: 2d/7/edited.pgm\n    map_yaml_file: 2d/7/map.yaml\n")
    meta = viz.load_map(str(tmp_path / "site"))
    assert meta["img"].tolist() == [[255, 255]] and meta["resolution"] == 0.5 and meta["origin"][:2] == [1.0, 2.0]

    (floor / "CJGC.yaml").write_text((floor / "map.yaml").read_text())
    pgm("CJGC.pgm", 200)
    assert viz.load_map(str(floor / "CJGC.yaml"))["img"].tolist() == [[200, 200]]


# ------------------------------------------------------------------ localization layers (loc_viz.py)
ORIGIN = (29.731405029387507, 106.78389112052452, 217.1952074889191)     # default_d20.yaml (the site)


def _kygi_schema():
    """INSPVAX's ros2msg schema as the recorder writes it: the main definition, then MSG: sections."""
    base = os.path.join(ROOT, "kygi665_msgs", "msg")
    text = open(os.path.join(base, "INSPVAX.msg")).read()
    for dep in ("Kygi665Header", "InertialSolutionStatus", "PositionOrVelocityType", "INSExtendedSolutionStatus"):
        text += "\n" + "=" * 80 + "\nMSG: kygi665_msgs/%s\n" % dep + open(os.path.join(base, dep + ".msg")).read()
    return text


def _loc_recording(path, map_pts, shift_after=None, ins_fixed_until=None, yaw=0.0):
    """A dog walking along +x at 1 m/s for 4 s: /tf, /lightning/odom (covariance[1] = 1 valid), the scans as
    lightning publishes them (current_scan_world: the map's points around the dog, 1 m off after shift_after),
    the INS (INSPVAX, RTK fixed until ins_fixed_until). yaw: the odom heading (pi: turned round, as in T48)."""
    from mcap.writer import CompressionType, Writer
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    sys.path.insert(0, TOOL)
    import loc_viz

    ts = get_typestore(Stores.ROS2_HUMBLE)
    sch = _kygi_schema()
    parts = sch.split("=" * 80)
    types = dict(get_types_from_msg(parts[0], "kygi665_msgs/msg/INSPVAX"))
    for part in parts[1:]:
        name = part.strip().splitlines()[0].split()[1].replace("/", "/msg/")
        types.update(get_types_from_msg(part.split("\n", 2)[2], name))
    ts.register(types)
    channels = {}
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start(profile="ros2", library="test")

        def put(topic, typename, msg, t, schema=None):
            if topic not in channels:
                text = schema or ts.generate_msgdef(typename, ros_version=2)[0]
                sid = w.register_schema(typename, "ros2msg", text.encode())
                channels[topic] = w.register_channel(topic, "cdr", sid)
            ns = int((t + OFFSET) * 1e9)
            w.add_message(channels[topic], ns, bytes(ts.serialize_cdr(msg, typename)), ns)

        def header(t, frame):
            ns = int(round(t * 1e9))
            return _make(ts, "std_msgs/msg/Header", stamp=_make(ts, "builtin_interfaces/msg/Time", sec=ns // 10 ** 9,
                                                                 nanosec=ns % 10 ** 9), frame_id=frame)
        for i in range(80):
            t = BAG_T0 + 0.05 * i
            x, y = 10.0 + (t - BAG_T0), 5.0
            q = _make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=float(np.sin(yaw / 2)),
                      w=float(np.cos(yaw / 2)))
            tr = _make(ts, "geometry_msgs/msg/TransformStamped", header=header(t, "map"), child_frame_id="base_link",
                       transform=_make(ts, "geometry_msgs/msg/Transform",
                                       translation=_make(ts, "geometry_msgs/msg/Vector3", x=x, y=y, z=0.0),
                                       rotation=q))
            put("/tf", "tf2_msgs/msg/TFMessage", _make(ts, "tf2_msgs/msg/TFMessage", transforms=[tr]), t)
            cov = np.zeros(36)
            cov[1] = 1.0
            pose = _make(ts, "geometry_msgs/msg/PoseWithCovariance",
                         pose=_make(ts, "geometry_msgs/msg/Pose",
                                    position=_make(ts, "geometry_msgs/msg/Point", x=x, y=y, z=0.0), orientation=q),
                         covariance=cov)
            put("/lightning/odom", "nav_msgs/msg/Odometry",
                _make(ts, "nav_msgs/msg/Odometry", header=header(t, "map"), child_frame_id="base_link", pose=pose), t)
            if i % 2 == 0:          # scans at 10 Hz
                near = map_pts[np.hypot(map_pts[:, 0] - x, map_pts[:, 1] - y) < 8.0].copy()
                if shift_after is not None and t - BAG_T0 >= shift_after:
                    near[:, 1] += 1.0           # (sliding along the floor alone would still sit on the floor)
                    near[:, 2] += 0.6
                ros = loc_viz
                msg = ros.cloud_msg(_Ros(ts), t, "map", near, {"intensity": np.zeros(len(near))})
                put("/lightning/current_scan_world", "sensor_msgs/msg/PointCloud2", msg, t)
            # the INS, 20 Hz: at the dog's position
            fixed = ins_fixed_until is None or t - BAG_T0 < ins_fixed_until
            dlat = y / 6378137.0 * 180.0 / np.pi
            dlon = x / (6378137.0 * np.cos(np.radians(ORIGIN[0]))) * 180.0 / np.pi
            ins = ts.types["kygi665_msgs/msg/INSPVAX"]
            m = _make(ts, "kygi665_msgs/msg/INSPVAX", header=header(t, "imu"),
                      ins_status=_make(ts, "kygi665_msgs/msg/InertialSolutionStatus", data=3),
                      pos_type=_make(ts, "kygi665_msgs/msg/PositionOrVelocityType", data=56 if fixed else 55),
                      latitude=ORIGIN[0] + dlat, longitude=ORIGIN[1] + dlon, height=ORIGIN[2], azimuth=90.0)
            assert ins is not None
            put("/rte/rdap/rtk/inspvax", "kygi665_msgs/msg/INSPVAX", m, t, schema=_kygi_schema())
        w.finish()


class _Ros:
    """The bits of planning_mcap_viz.Ros that cloud_msg uses, on a given typestore."""

    def __init__(self, ts):
        self.ts = ts

    def new(self, typename, **kw):
        msg = _make(self.ts, typename)          # PointField has a field called name: set the fields afterwards
        for k, v in kw.items():
            setattr(msg, k, v)
        return msg

    def header(self, t, frame):
        ns = int(round(t * 1e9))
        return _make(self.ts, "std_msgs/msg/Header",
                     stamp=_make(self.ts, "builtin_interfaces/msg/Time", sec=ns // 10 ** 9, nanosec=ns % 10 ** 9),
                     frame_id=frame)


def _loc_map(tmp_path):
    """A site map directory with 3d/global.pcd: a wall along y = 8 m and a floor grid, binary PCD."""
    xs, ys = np.meshgrid(np.arange(0.0, 30.0, 0.1), np.arange(0.0, 12.0, 0.1))
    floor = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.4)])
    wx, wz = np.meshgrid(np.arange(0.0, 30.0, 0.1), np.arange(-0.4, 2.0, 0.1))
    wall = np.column_stack([wx.ravel(), np.full(wx.size, 8.0), wz.ravel()])
    pts = np.vstack([floor, wall]).astype(np.float32)
    folder = tmp_path / "locmap" / "3d"
    folder.mkdir(parents=True)
    head = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (len(pts), len(pts)))
    body = np.column_stack([pts, np.zeros(len(pts), np.float32)]).astype("<f4").tobytes()
    (folder / "global.pcd").write_bytes(head.encode() + body)
    return str(tmp_path / "locmap"), pts.astype(float)


def test_loc_viz_scans_on_the_map_ins_and_events(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    pytest.importorskip("scipy")
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    locmap, pts = _loc_map(tmp_path)
    bag = tmp_path / "rec"
    bag.mkdir()
    _loc_recording(str(bag / "rec_0.mcap"), pts, shift_after=2.0, ins_fixed_until=3.0)
    cfg = tmp_path / "loc.yaml"
    cfg.write_text("system:\n  ins_lat: %r\n  ins_lon: %r\n  ins_height: %r\n" % ORIGIN)
    out = str(tmp_path / "viz.mcap")
    assert viz.main([str(bag), "-o", out, "--map", "none", "--overview", "none", "--loc-map", locmap,
                     "--loc-config", str(cfg)]) == 0
    text = capsys.readouterr().out
    assert "localization map:" in text and "/viz/loc/scan: 20 scans" in text, text
    msgs = _read(out)
    # the scans: on the map first, 1 m off after 2 s
    scans = msgs["/viz/loc/scan"]
    names = [f.name for f in scans[0][1].fields]
    assert names == ["x", "y", "z", "height", "match"]

    def match(m):
        arr = np.frombuffer(bytes(m.data), dtype=[(n, "<f4") for n in names])
        return arr["match"]
    assert len(scans) == 7               # read at 5 Hz (0, 0.2, ...), drawn at most 2 Hz: 0, 0.6, 1.2, ... 3.6 s
    assert np.median(match(scans[1][1])) < 0.05 and np.median(match(scans[-1][1])) > 0.3
    states = [m for _, m in msgs["/viz/loc/state"]]
    assert states[3]["valid"] == 1 and states[3]["match_inlier"] > 0.9 and states[-2]["match_inlier"] < 0.5
    # the INS in the map frame where the dog is, RTK fixed until 3 s
    assert abs(states[3]["odom_minus_ins"]) < 0.05 and states[3]["ins_fixed"] == 1 and states[-2]["ins_fixed"] == 0
    # walking forward: the heading matches the INS's direction of travel
    assert abs(states[3]["heading_off_course"]) < 5 and "median |.| 0 deg" in text and "TURNED ROUND" not in text
    flow = msgs["/viz/loc/flow"][3][1]
    assert "③ LIO 和地图配准" in flow and "laser_mapping.cc" in flow["③ LIO 和地图配准"]["代码"]
    events = [m.msg for _, m in msgs["/viz/events"]]
    assert any("贴在地图上" in e for e in events), events
    assert any("不是 RTK 固定解" in e for e in events), events
    assert "/viz/loc/map_cloud" in msgs and "/viz/loc/ins" in msgs


def test_loc_viz_finds_a_heading_turned_round(tmp_path, capsys):
    """T48: the 0921 bags replayed without system.ins_yaw_offset 180 had the dog's heading 180 deg off the way it
    walked. The INS's direction of travel comes from its positions, so it catches that whatever the INS convention."""
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    pytest.importorskip("scipy")
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    locmap, pts = _loc_map(tmp_path)
    bag = tmp_path / "rec"
    bag.mkdir()
    _loc_recording(str(bag / "rec_0.mcap"), pts, yaw=np.pi)
    cfg = tmp_path / "loc.yaml"
    cfg.write_text("system:\n  ins_lat: %r\n  ins_lon: %r\n  ins_height: %r\n" % ORIGIN)
    out = str(tmp_path / "viz.mcap")
    assert viz.main([str(bag), "-o", out, "--map", "none", "--overview", "none", "--loc-map", locmap,
                     "--loc-config", str(cfg)]) == 0
    text = capsys.readouterr().out
    assert "median |.| 180 deg" in text and "TURNED ROUND" in text, text
    msgs = _read(out)
    states = [m for _, m in msgs["/viz/loc/state"]]
    assert abs(abs(states[5]["heading_off_course"]) - 180) < 5
    events = [m.msg for _, m in msgs["/viz/events"]]
    assert any("朝向转反了" in e for e in events), events


def test_loc_viz_match_is_point_to_plane_whatever_the_map_density():
    """T48: the nearest-point distance read a standing dog 11-30% "on the map" against /lightning/map_cloud (thinned
    to 0.5 m). Point to plane (5 nearest map points, as lightning's LIO) reads the same on a 0.1 m and a 0.5 m map."""
    pytest.importorskip("scipy")
    sys.path.insert(0, TOOL)
    import loc_viz

    rng = np.random.default_rng(0)
    scan = np.column_stack([rng.uniform(0, 10, 2000), rng.uniform(0, 10, 2000), rng.normal(0, 0.02, 2000)])
    off = scan + [0.0, 0.0, 0.3]                                    # 0.3 m above the floor
    for step in (0.1, 0.5):
        g = np.arange(-1.0, 11.0 + 1e-9, step)
        floor = np.column_stack([np.repeat(g, len(g)), np.tile(g, len(g)), np.zeros(len(g) ** 2)])
        near = loc_viz.Nearest(floor)
        on = near(scan)
        assert np.mean(on < loc_viz.MATCH_GOOD) > 0.95 and np.median(on) < 0.03, step
        assert np.mean(near(off) < loc_viz.MATCH_GOOD) < 0.05, step
    # the nearest point alone: half the scan "off" a 0.5 m map
    nn = np.min(np.linalg.norm(scan[:, None, :2] - floor[None, :, :2], axis=2), axis=1)
    assert np.mean(nn < loc_viz.MATCH_GOOD) < 0.3
    # no plane (5 points on a line) or too far: the nearest point's distance
    line = np.column_stack([np.arange(0, 1, 0.1), np.zeros(10), np.zeros(10)])
    assert loc_viz.Nearest(line)([[0.45, 0.1, 0.1]])[0] == pytest.approx(0.15, abs=1e-6)  # to (0.4 or 0.5, 0, 0)
    assert loc_viz.Nearest(floor)([[5.0, 5.0, 3.0]])[0] == pytest.approx(3.0, abs=1e-6)


def test_lla_to_enu_matches_local_cartesian():
    sys.path.insert(0, TOOL)
    import loc_viz

    # 100 m north and 50 m east of the origin, 2 m up: within a millimetre of a tangent-plane conversion
    lat0, lon0, h0 = ORIGIN
    r_n = 6378137.0 / np.sqrt(1 - 0.00669437999014 * np.sin(np.radians(lat0)) ** 2)
    r_m = r_n * (1 - 0.00669437999014) / (1 - 0.00669437999014 * np.sin(np.radians(lat0)) ** 2)
    lat = lat0 + np.degrees(100.0 / (r_m + h0))
    lon = lon0 + np.degrees(50.0 / ((r_n + h0) * np.cos(np.radians(lat0))))
    e, n, u = loc_viz.lla_to_enu(lat, lon, h0 + 2.0, ORIGIN)
    assert e == pytest.approx(50.0, abs=0.01) and n == pytest.approx(100.0, abs=0.01) and u == pytest.approx(2.0, abs=0.01)
    assert loc_viz.read_loc_origin("system:\n  ins_lat:  29.7\n  ins_lon: 106.7\n  ins_height: 217.1\n") == (
        29.7, 106.7, 217.1)


def test_parse_glog_reads_old_and_new_glog_lines(tmp_path):
    """glog 0.7 (the server since T48) writes the year in front of the date; lightning colours warnings on stderr.
    The old pattern read neither, so T48's planning and localization logs gave no events."""
    sys.path.insert(0, TOOL)
    import planning_mcap_viz as viz

    log = tmp_path / "loc.log"
    log.write_text("I1113 11:35:32.023888 139889434583040 tracking.cc:181] [PLANNING]CMD:  0.01 -0.06 -0.01\n"
                   "I20261113 11:35:33.500000 139889434583040 loc_system.cc:211] kygi665 INS 朝向偏置 "
                   "system.ins_yaw_offset = 180 deg\n"
                   "\x1b[1;33mW20261113 11:35:34.000000 1 localization.cc:88] 定位质量差\x1b[m\n"
                   "[INFO] [1789980087.9] [rclcpp]: not glog\n")
    lines = viz.parse_glog(str(log), default_year=2026)
    assert [(lv, where) for _, lv, where, _ in lines] == [
        ("info", "tracking.cc:181"), ("info", "loc_system.cc:211"), ("warn", "localization.cc:88")]
    assert lines[1][0] - lines[0][0] == pytest.approx(1.476112, abs=1e-5)
    assert lines[2][3] == "定位质量差" and "ins_yaw_offset = 180" in lines[1][3]
