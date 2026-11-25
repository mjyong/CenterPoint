"""Nav_D20/planning/tools/field_viz.py (through planning_mcap_viz.py) on a made-up bag recorded on the dog: the dog
walks up to crosswalk 13 of the site map (planner index 1) while a car crosses ahead, a pedestrian stands by, a false
detection lives half a second and a second car is first seen 8 m away; the planner approaches, waits for the car and
goes; its gate drawing, the shadow commands, the lidar (a ground plane and the car) and the front fisheye (H.264
frames bigger than the spill threshold). Then the same run as a replay on the server's clock, with the fisheye taken
from the field bag."""
import math
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make
from .test_planning_viz import _stamp

TOOL = os.path.join(ROOT, "Nav_D20", "planning", "tools")
T0 = 1795000000.0
OFFSET = 4504296.0          # a replay's recorder clock
START = (90.0, 200.0)
SPEED = 0.5
DET = (0.319, 0.043, -90.0)
CAR_X = 100.0               # the car crosses ahead along +y at 5 m/s
CAMERA = "/robot/topic/rdap/front_sensor_camera"


def _sec(time):
    return time.sec + time.nanosec * 1e-9


def _dog(t):
    return START[0] + SPEED * (t - T0), START[1], 0.0


def _car(t):
    return CAR_X, START[1] - 10.0 + 5.0 * (t - T0)


def _to_det(pose, x, y):
    px, py, yaw = pose
    dx, dy, ydb = DET
    a = yaw + math.radians(ydb)
    tx, ty = px + math.cos(yaw) * dx - math.sin(yaw) * dy, py + math.sin(yaw) * dx + math.cos(yaw) * dy
    u, v = x - tx, y - ty
    return math.cos(a) * u + math.sin(a) * v, -math.sin(a) * u + math.cos(a) * v


def _rot_det(pose, vx, vy):
    a = pose[2] + math.radians(DET[2])
    return math.cos(a) * vx + math.sin(a) * vy, -math.sin(a) * vx + math.cos(a) * vy


def _mode(t):
    d = t - T0
    return 0 if d < 1 else 1 if d < 2 else 3 if d < 4 else 4


def _write(path, offset, camera=True, planner=True, lidar=True):
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

    def vec(x, y, z=0.0):
        return _make(ts, "geometry_msgs/msg/Vector3", x=float(x), y=float(y), z=float(z))

    def quat(yaw):
        return _make(ts, "geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))

    def pose(x, y, z=0.0, yaw=0.0):
        return _make(ts, "geometry_msgs/msg/Pose", position=point(x, y, z), orientation=quat(yaw))

    channels = {}
    with open(path, "wb") as fh:
        w = Writer(fh, compression=CompressionType.NONE)
        w.start(profile="ros2", library="test")

        def put(topic, typename, msg, bag_t, lag=0.02):
            if topic not in channels:
                sid = w.register_schema(typename, "ros2msg", ts.generate_msgdef(typename, ros_version=2)[0].encode())
                channels[topic] = w.register_channel(topic, "cdr", sid)
            ns = int((bag_t + offset + lag) * 1e9)
            w.add_message(channels[topic], ns, bytes(ts.serialize_cdr(msg, typename)), ns)

        def obj(oid, label, motion, pose_, x, y, vx, vy, coasting=False, size=(4.5, 1.8, 1.5)):
            ox, oy = _to_det(pose_, x, y)
            dvx, dvy = _rot_det(pose_, vx, vy)
            path = _make(ts, "nav_interface/msg/LidarPredictedPath", probability=0.8,
                         positions=[point(ox + dvx * k, oy + dvy * k) for k in (1.0, 2.0)], covariance=np.zeros(0))
            weak = _make(ts, "nav_interface/msg/LidarPredictedPath", probability=0.2,
                         positions=[point(ox, oy), point(ox, oy)], covariance=np.zeros(0))
            return _make(ts, "nav_interface/msg/LidarObject", id=oid, label=label,
                         label_name=["vehicle", "pedestrian", "cyclist"][label], motion=motion, score=0.9,
                         coasting=coasting, age=1.0, position=point(ox, oy, 0.8),
                         yaw=math.pi / 2 - math.radians(DET[2]) - pose_[2], size=vec(*size),   # map yaw pi / 2
                         velocity=vec(dvx, dvy), covariance=np.zeros(16),
                         prediction_times=np.array([1.0, 2.0], np.float32), predictions=[weak, path])

        for i in range(60):                              # 6 s at 10 Hz
            t = T0 + 0.1 * i
            x, y, yaw = _dog(t)
            put("/lightning/odom", "nav_msgs/msg/Odometry",
                _make(ts, "nav_msgs/msg/Odometry", header=header(t, "map"), child_frame_id="base_link",
                      pose=_make(ts, "geometry_msgs/msg/PoseWithCovariance", pose=pose(x, y, -0.2, yaw)),
                      twist=_make(ts, "geometry_msgs/msg/TwistWithCovariance",
                                  twist=_make(ts, "geometry_msgs/msg/Twist", linear=vec(SPEED, 0.0)))), t)
            if camera:
                frame = bytes([0, 0, 0, 1, 0x65]) + bytes([i % 256]) * 40000
                put(CAMERA, "foxglove_msgs/msg/CompressedVideo",
                    _make(ts, "foxglove_msgs/msg/CompressedVideo", timestamp=time(t), frame_id="front",
                          data=np.frombuffer(frame, np.uint8), format="h264"), t, lag=0.05)
            p = (x, y, yaw)
            objs = [obj(5, 0, 1, p, *_car(t), 0.0, 5.0), obj(6, 1, 0, p, 101.0, 203.0, 0.0, 0.0, size=(0.5, 0.5, 1.7))]
            if 2.0 <= t - T0 < 2.5:
                objs.append(obj(99, 0, 0, p, x + 5.0, y - 3.0, 0.0, 0.0))
            if t - T0 >= 3.0:
                objs.append(obj(7, 0, 1, p, x + 8.0, y + 1.0, 0.0, -1.0, coasting=t - T0 > 5.0))
            put("/perception/lidar_3d/objects", "nav_interface/msg/LidarObjectArray",
                _make(ts, "nav_interface/msg/LidarObjectArray", header=header(t, "lidar_det"),
                      world_frame="camera_init", world_pose=pose(0, 0), objects=objs), t, lag=0.03)
            # the lidar: ground 0.7 m under it, the car's points up to 1.5 m over the ground
            g = np.mgrid[-20:20:1.0, -20:20:1.0].reshape(2, -1).T
            pts = [np.c_[g, np.full(len(g), -0.7)]]
            cx, cy = _to_det(p, *_car(t))
            # det x is the dog's right: the car's length
            car = np.mgrid[-2.0:2.0:0.2, -0.8:0.8:0.2, 0.0:1.5:0.3].reshape(3, -1).T
            pts.append(np.c_[cx + car[:, 0], cy + car[:, 1], -0.7 + car[:, 2]])
            xyz = np.vstack(pts).astype(np.float32)
            cloud = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
            cloud["x"], cloud["y"], cloud["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            fields = [ts.types["sensor_msgs/msg/PointField"](name=n, offset=4 * k, datatype=7, count=1)
                      for k, n in enumerate(("x", "y", "z", "intensity"))]
            if lidar:      # a replay keeps no point clouds
                put("/robot/topic/rdap/top_point_cloud2", "sensor_msgs/msg/PointCloud2",
                    _make(ts, "sensor_msgs/msg/PointCloud2", header=header(t, "hesai_lidar"), height=1,
                          width=len(cloud), fields=fields, is_bigendian=False, point_step=16,
                          row_step=16 * len(cloud), data=np.frombuffer(cloud.tobytes(), np.uint8), is_dense=True),
                    t, lag=0.04)
            if not planner:
                continue
            mode = _mode(t)
            text = ("planning: 0\nmission_type: 1.000000\nmission_status: 3.000000\ncmd_vx: %.6f\ncmd_vy: 0\n"
                    "cmd_omega: 0\ncrosswalk_mode: %d\ncrosswalk_index: 1\ncrosswalk_s_stop: %.3f\n"
                    "crosswalk_hold: 0\ncrosswalk_go: %d\ncrosswalk_vehicles: 2\ncrosswalk_waiting_for: %d\n"
                    "crosswalk_stale: 0\ncrosswalk_unreleased: 0\ncrosswalk_heading_error: 0.1\n" % (
                        0.0 if mode == 3 else 0.5, mode, 9.0 - SPEED * (t - T0), 1 if mode == 4 else 0,
                        1 if mode in (3, 4) else 0))
            put("/planning/debug/plotjuggler/string", "std_msgs/msg/String",
                _make(ts, "std_msgs/msg/String", data=text), t)
            put("/nav_cw/shadow/cmd_vel", "geometry_msgs/msg/Twist",
                _make(ts, "geometry_msgs/msg/Twist", linear=vec(0.0 if mode == 3 else 0.5, 0.0)), t)
            put("/cmd_vel", "geometry_msgs/msg/Twist", _make(ts, "geometry_msgs/msg/Twist", linear=vec(0.3, 0.0)), t)
            if mode:
                clear = _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), action=3)
                ring = _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), ns="crosswalks", id=1,
                             type=4, action=0, scale=vec(0.25, 0, 0),
                             points=[point(99, 196), point(103, 196), point(103, 206), point(99, 206), point(99, 196)])
                cx_, cy_ = _car(t)
                box = _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), ns="vehicles", id=2, type=1,
                            action=0, pose=pose(cx_, cy_, 0.8, math.pi / 2), scale=vec(4.5, 1.8, 1.6))
                line = _make(ts, "visualization_msgs/msg/Marker", header=header(t, "map"), ns="state", id=3, type=9,
                             action=0, pose=pose(x, y, 2.0), text="%s (vehicle 5 may pass)%s" % (
                                 {1: "approach", 3: "wait", 4: "go"}[mode],
                                 ", waiting for 1" if mode == 3 else ""))
                put("/planning/debug/crosswalk", "visualization_msgs/msg/MarkerArray",
                    _make(ts, "visualization_msgs/msg/MarkerArray", markers=[clear, ring, box, line]), t)
        w.finish()


def _osm(tmp_path):
    """Two crosswalks: relation 6 far away, relation 13 ahead of the dog at x 99..103."""
    nodes, ways, rels = [], [], []
    for rid, (x0, y0) in (("6", (20.0, 20.0)), ("13", (99.0, 196.0))):
        ids = []
        for k, (dx, dy) in enumerate(((0, 0), (4, 0), (4, 10), (0, 10))):
            nid = "%s%d" % (rid, k)
            ids.append(nid)
            nodes.append('<node id="%s" lat="0" lon="0"><tag k="local_x" v="%.1f"/><tag k="local_y" v="%.1f"/>'
                         '</node>' % (nid, x0 + dx, y0 + dy))
        ways.append('<way id="w%s">%s</way>' % (rid, "".join('<nd ref="%s"/>' % n for n in ids + ids[:1])))
        rels.append('<relation id="%s"><member type="way" ref="w%s" role="outer"/><tag k="type" v="multipolygon"/>'
                    '<tag k="subtype" v="crosswalk"/></relation>' % (rid, rid))
    path = tmp_path / "semantic.osm"
    path.write_text('<?xml version="1.0"?><osm version="0.6">%s</osm>' % "".join(nodes + ways + rels))
    return str(path)


def _read(path):
    import json

    from bag_on_map import typestore
    from mcap.reader import make_reader
    ts = typestore(all_packages=True, quiet=True)
    out = {}
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(log_time_order=True):
            msg = json.loads(message.data) if channel.message_encoding == "json" else \
                ts.deserialize_cdr(message.data, schema.name)
            out.setdefault(channel.topic, []).append((message.log_time * 1e-9, msg))
    return out


def _run(tmp_path, bag, *extra):
    import planning_mcap_viz as viz
    out = str(tmp_path / ("viz_%s.mcap" % os.path.basename(bag)))
    assert viz.main([bag, "-o", out, "--map", "none", "--osm", _osm(tmp_path), "--overview", "none"] + list(extra)) \
        == 0
    return out


def test_field_bag_objects_gate_crosswalk_lidar_and_fisheye(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    import field_viz
    bag = str(tmp_path / "field.mcap")
    _write(bag, offset=0.0)
    msgs = _read(_run(tmp_path, bag))
    text = capsys.readouterr().out
    assert "commands: /nav_cw/shadow/cmd_vel (shadow mode" in text, text
    assert "planner index 0.. = 6, 13" in text and "placed by det_to_base (no TF hesai_lidar -> base_link)" in text

    # perception's car where the gate would place it, with its arrow, prediction and label
    log, arr = msgs["/viz/objects"][30]
    assert arr.markers[0].action == 3
    car = [m for m in arr.markers if m.ns == "box" and m.type == field_viz.LINE_STRIP]
    corners = np.array([(p.x, p.y) for p in car[0].points[:4]])
    cx, cy = _car(log)
    assert np.allclose(corners.mean(axis=0), (cx, cy), atol=1e-3)
    assert np.allclose(sorted(np.hypot(*np.diff(corners[[0, 1, 2]], axis=0).T)), (1.8, 4.5), atol=1e-3)
    arrow = [m for m in arr.markers if m.ns == "velocity"][0]
    assert abs(arrow.points[1].x - cx) < 1e-3 and abs(arrow.points[1].y - (cy + 5.0)) < 1e-3
    pred = [m for m in arr.markers if m.ns == "prediction"][0]          # the likelier of the two
    assert abs(pred.points[-1].y - (cy + 10.0)) < 1e-3
    labels = [m.text for m in arr.markers if m.ns == "label"]
    assert "#5 车 18 km/h" in labels and "#6 人 静止" in labels
    assert all(m.header.frame_id == "map" and abs(_stamp(m.header) - log) < 1e-6 for m in arr.markers[1:])
    # the coasting one is dashed
    late = [m for _, a in msgs["/viz/objects"][-3:] for m in a.markers if m.ns == "box"]
    assert any(m.type == field_viz.LINE_LIST for m in late)
    table = msgs["/viz/objects_table"][30][1]
    assert table["n"] == 3 and table["vehicles"] == 2 and table["pedestrians"] == 1
    dists = [o["distance_m"] for o in table["objects"]]
    assert dists == sorted(dists)                              # nearest first
    seven = [o for o in table["objects"] if o["id"] == 7][0]
    assert abs(seven["distance_m"] - math.hypot(8, 1)) < 0.06 and abs(seven["bearing_deg"] - 7) < 1

    # the gate's drawing flattened, its state line on the robot's label and in /viz/crosswalk
    gate = msgs["/viz/crosswalk_gate"][15][1]
    assert {m.ns for m in gate.markers[1:]} == {"crosswalks", "vehicles"}
    assert all(m.pose.position.z < 0.05 for m in gate.markers[1:])
    cw = [m for _, m in msgs["/viz/crosswalk"]]
    modes = [c["mode"] for c in cw]
    assert modes[0] == "none" and "approach" in modes and "wait" in modes and modes[-1] == "go"
    wait = [c for c in cw if c["mode"] == "wait"][0]
    assert wait["crosswalk"] == "13" and wait["waiting_for"] == 1 and wait["reason"] == "vehicle 5 may pass"
    assert wait["meaning"] == field_viz.MODE_MEANING["wait"] and abs(wait["heading_error_deg"] - 5.7) < 0.1
    go = cw[-1]
    assert go["go"] == 1 and go["waiting_for"] == 0 and go["going_on_with"] == 1
    robot = [m.text for _, a in msgs["/viz/robot"] for m in a.markers if m.ns == "label"]
    assert any("过马路 13 wait，等 1 辆" in r for r in robot), robot[25:30]
    states = [m for _, m in msgs["/viz/state"]]
    assert min(s["cmd_v"] for s in states) == 0.0         # the shadow command, not /cmd_vel's 0.3

    # events: the mode changes by the site map's number, the false detection, the car first seen close
    events = [m.msg for _, m in msgs["/viz/events"]]
    assert any(e.startswith("人行横道 13：approach → wait") and "等 1 辆" in e for e in events), events
    assert any("人行横道 13：wait → go" in e for e in events), events
    assert any("#99 车 只出现 5 帧（0.4 s）" in e and "可能误检" in e for e in events), events
    assert any("#7 车 在 8.1 m 处才出现" in e for e in events), events
    assert not any("#5 " in e for e in events)

    # the lidar: no ground, the car's points where the car is, under the other layers
    from planning_mcap_viz import cloud_xyz
    log, cl = msgs["/viz/lidar"][10]
    xyz, _ = cloud_xyz(cl)
    names = [f.name for f in cl.fields]
    assert names == ["x", "y", "z", "height"] and len(xyz) > 20
    h = np.frombuffer(bytes(cl.data), dtype=np.float32).reshape(-1, 4)[:, 3]
    assert h.min() > field_viz.GROUND_ABOVE and np.allclose(xyz[:, 2], field_viz.Z_LIDAR)
    cx, cy = _car(log)
    assert np.all(np.abs(xyz[:, 0] - cx) < 0.9) and np.all(np.abs(xyz[:, 1] - cy) < 2.1)

    # the fisheye as recorded, on the bag's time
    cam = msgs[CAMERA]
    assert len(cam) == 60
    for log, m in cam[::10]:
        assert abs(log - _sec(m.timestamp)) < 0.06 and m.format == "h264" and len(m.data) == 40005
    assert "/viz/legend" in msgs and "现场复盘的图层（狗上录的包；布局 field_review.json）" in msgs["/viz/legend"][0][1]

    # the layout shows what this writes
    import json
    layout = json.load(open(os.path.join(ROOT, "Nav_D20", "planning", "script", "field_review.json")))
    used = {CAMERA: layout["configById"]["Image!front"]["imageMode"]["imageTopic"]}
    for cid, cfg in layout["configById"].items():
        for p in cfg.get("paths", []):
            used[cid + " " + p["value"]] = p["value"].split(".")[0]
        for key in ("topicPath", "topicToRender"):
            if key in cfg:
                used[cid] = cfg[key]
    for topic in ("/viz/lidar", "/viz/objects", "/viz/crosswalk_gate"):
        assert layout["configById"]["3D!field"]["topics"][topic]["visible"]
        used[topic] = topic
    assert {k: v for k, v in used.items() if v not in msgs} == {}
    assert "open in Foxglove with the layout Nav_D20/planning/script/field_review.json" in text


def test_replay_takes_the_fisheye_from_the_field_bag(tmp_path, capsys):
    pytest.importorskip("mcap")
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOL)
    field = str(tmp_path / "field.mcap")
    _write(field, offset=0.0, planner=False)
    replay = str(tmp_path / "replay.mcap")
    _write(replay, offset=OFFSET, camera=False, lidar=False)
    msgs = _read(_run(tmp_path, replay, "--camera-from", field))
    text = capsys.readouterr().out
    assert "%s (foxglove_msgs/msg/CompressedVideo) from field.mcap: 60 frames" % CAMERA in text, text
    assert "/robot/topic/rdap/top_point_cloud2 of field.mcap -> /viz/lidar" in text, text
    from planning_mcap_viz import cloud_xyz
    log, cl = msgs["/viz/lidar"][10]
    xyz, _ = cloud_xyz(cl)
    cx, cy = _car(log)
    assert len(xyz) > 20 and np.all(np.abs(xyz[:, 0] - cx) < 0.9) and np.all(np.abs(xyz[:, 1] - cy) < 2.1)
    cam = msgs[CAMERA]
    assert len(cam) == 60
    for log, m in cam:
        assert abs(log - _sec(m.timestamp)) < 0.06          # the bag's time, not the replay server's clock
    assert all(T0 - 1 < log < T0 + 10 for rows in msgs.values() for log, _ in rows)


def test_crosswalk_state_without_a_site_map():
    sys.path.insert(0, TOOL)
    import field_viz

    class Conv:
        rec, notes, ros = {}, [], None
    layers = field_viz.FieldLayers(Conv())
    st = layers.crosswalk_state({"crosswalk_mode": 6.0, "crosswalk_index": 2.0, "crosswalk_waiting_for": 2.0,
                                 "crosswalk_stale": 1.0}, 0.0)
    assert st["mode"] == "cross_wait" and st["crosswalk"] == "#2" and st["waiting_for"] == 2 and st["stale"] == 1
    assert layers.crosswalk_state({"mission_type": 1.0}) is None
    assert field_viz.parse_gate_text("wait (vehicle 12 may pass), waiting for 1") == "vehicle 12 may pass"
