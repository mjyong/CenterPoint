#!/usr/bin/env python3
"""A planning recording (ros2 bag, mcap) -> a Foxglove recording that does not blink and shows the whole picture.

    python3 Nav_D20/planning/tools/planning_mcap_viz.py work_dirs/planning_io_t47_map \
        [--map <map.yaml | map dir with 2d/<id>/map.yaml | map zip>] [--osm semantic.osm] [-o OUT.mcap] \
        [--overview OUT.png] [--glog planning.INFO]

Writes <recording>/viz/planning_viz.mcap (and planning_viz.png): open it in Foxglove with the layout
Nav_D20/planning/script/planning_viz.json.

Why the raw recording blinks in Foxglove (T47 recordings):
- planning's local costmap (20 Hz) and perception's /map/occupancy_grid (10 Hz) both hang on base_link. Foxglove
  places each with the TF at its own stamp: the dog rolls and pitches 1-4 degrees while walking and the two stamps
  are 0.1 s apart, so the two tilted planes cut through each other. Both are opaque (free cells white by default):
  which one is on top changes from pixel to pixel and from frame to frame. The paths are lines at z = 0 in map
  while base_link is at z = -0.2 m (-1.1 .. 0.07 m on the T47 runs): the far end of a tilted grid rises through
  them and hides a stretch of path, then lets it go.
- planning's markers carry stamp 0. In map that does not matter; a marker in base_link (collision contacts) is
  placed with the earliest TF Foxglove has, at the start of the run.
- planning's costmap topic was inverted: the costmap's obstacle layer holds 255 = free, 0 = obstacle, and
  ros_debug_bridge converted it with (0, 1), so every free cell read 100 and the obstacles 0. On the T47 runs all
  117045 perception obstacle cells read 0 in it, and the red band in Foxglove was the road the dog walks on.
  --cost-map-values auto (default) tells which way a recording is from those cells.
- the recorder logs with the server's wall clock (November) while the data are stamped with the bag's time
  (21 September): the timeline, plots and logs do not line up with the bag or with the dog's logs.

What this writes instead, everything on one time base (the bag's stamps) and flattened onto the map plane, each layer
at its own height so that nothing fights for the same depth:
  /tf                    map -> base_link: x, y, yaw only (one publisher, 100 Hz)
  /viz/route_map         the planning map (CJGC route network by default) cropped around the run: route cells 100
  /viz/crosswalks        crosswalks of semantic.osm: filled, outlined, numbered
  /viz/cost_map          planning's local costmap, placed with the pose at its stamp; free cells 0 (transparent).
                         Recordings before the T47 review have it inverted (see below): turned the right way up
  /viz/perception_grid   perception's grid, the same way, one layer above
  /viz/global_map        planning's global map, when the recording has it
  /viz/plan              global path, local path, its footprints, tracking target, collision circles, goal
  /viz/robot             the dog in base_link (frame_locked): body, heading, velocity and command arrows, turn arc,
                         status label
  /viz/trail             the path walked so far coloured by speed, with the stops
  /viz/collision, /viz/teb, /viz/following   planning's other debug markers, flattened (when recorded)
  /viz/cloud             /fused_point_cloud in map, voxel-downsampled (when recorded)
  /viz/state             JSON per planning cycle: mission, speeds, commands, distance to the goal, offset from the
                         global path, occupied costmap cells under the dog, nearest obstacles, planning's raw keys
  /viz/events            rcl_interfaces/Log: mission changes, timeouts, stops, gaps, jumps, costmap under the dog,
                         goal, diagnostics, and planning's own warnings when its glog is there

Needs numpy, mcap, rosbags (and zstandard for compression, matplotlib for the overview).
"""
import argparse
import calendar
import glob
import io
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from bisect import bisect_right
from collections import defaultdict

import numpy as np

NAV_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_MAP = os.path.join(NAV_ROOT, "planning", "robot_factory", "data", "astar_pgm", "CJGC.yaml")
DEFAULT_LAYOUT = os.path.join(NAV_ROOT, "planning", "script", "planning_viz.json")
DEFAULT_LOC_CONFIG = os.path.join(NAV_ROOT, "slam", "lightning-lm-master", "config", "default_d20.yaml")

TOPICS = dict(
    tf="/tf", tf_static="/tf_static", odom="/lightning/odom", plan="/planning/debug/path_tracking",
    goal_marker="/planning/debug/goal_pose", goal="/goal_pose", cost_map="/planning/debug/cost_map_production/occmap",
    perception="/map/occupancy_grid", global_map="/planning/debug/global_map_manager/occmap", cmd="/cmd_vel",
    string="/planning/debug/plotjuggler/string", diagnostic="/planning/debug/diagnostic",
    collision="/planning/debug/collision_checker", teb="/planning/debug/teb_planner",
    following="/planning/debug/following", cloud="/fused_point_cloud",
    # localization (lightning; loc_viz.py)
    scan_world="/lightning/current_scan_world", map_cloud="/lightning/map_cloud", inspvax="/rte/rdap/rtk/inspvax",
    loc_event="/rhr/topic/event_info")
OTHER_MARKERS = (("collision", "/viz/collision"), ("teb", "/viz/teb"), ("following", "/viz/following"))

# heights of the flattened layers [m]
Z = dict(route=-0.06, crosswalk=-0.05, global_map=-0.055, cost_map=-0.03, perception=-0.02, trail=0.0,
         stops=0.005, global_path=0.01, footprints=0.015, reference=0.018, local_path=0.02, circles=0.025, other=0.028,
         target=0.03, goal=0.03, robot=0.04, arrows=0.06, label=0.5)

MISSION_TYPES = {0: "idle", 1: "tracking", 2: "following", 3: "recharging", 4: "skill"}   # MissionConf::Type
MISSION_STATES = {0: "off", 1: "passive", 2: "standby", 3: "active", 4: "standstill", 5: "fault"}  # MissionFSMBase
MISSION_ZH = {"idle": "空闲", "tracking": "跟踪", "following": "跟随", "recharging": "回充", "skill": "技能"}

BLUE = (0.04, 0.52, 1.0, 0.95)
GREEN = (0.19, 0.82, 0.35, 1.0)
ORANGE = (1.0, 0.62, 0.04, 1.0)
RED = (1.0, 0.27, 0.23, 1.0)
CYAN = (0.25, 0.85, 1.0, 0.9)
YELLOW = (1.0, 0.84, 0.04, 1.0)
PINK = (1.0, 0.4, 1.0, 0.18)
WHITE = (1.0, 1.0, 1.0, 1.0)
GRAY = (0.7, 0.7, 0.7, 1.0)
STATE_COLOR = {"active": GREEN, "standstill": YELLOW, "standby": (0.85, 0.85, 0.85, 1.0), "passive": GRAY,
               "off": GRAY, "fault": RED, "idle": GRAY}

# planning's /planning/debug/path_tracking namespaces: (layer, colour or None to keep, line width or None)
PLAN_STYLE = {
    "global_path_2d": ("global_path", BLUE, 0.15),
    "local_path_2d": ("local_path", GREEN, 0.08),
    "local_reference_2d": ("reference", (0.75, 0.35, 1.0, 0.9), 0.05),   # MPPI's input (recorded from T49 on)
    "local_path_footprint": ("footprints", (0.19, 0.82, 0.35, 0.35), 0.02),
    "target_pt": ("target", None, None),
    "robot_circle": ("circles", PINK, None),
    "robot_footprint": None,   # drawn from the TF on /viz/robot
    "robot_heading": None,
}

LOG = dict(debug=10, info=20, warn=30, error=40, fatal=50)
NP_BASE = {"bool": np.bool_, "byte": np.uint8, "char": np.uint8, "uint8": np.uint8, "int8": np.int8,
           "uint16": np.uint16, "int16": np.int16, "uint32": np.uint32, "int32": np.int32, "uint64": np.uint64,
           "int64": np.int64, "float32": np.float32, "float64": np.float64}

ROBOT_RATE = 10.0     # [Hz] /viz/robot
TRAIL_RATE = 2.0      # [Hz] /viz/trail
TRAIL_STEP = 0.2      # [m] spacing of the trail's points
STATIC_PERIOD = 15.0  # [s] the map layers are sent again (a seek past the first copy still finds one)
MAP_MARGIN = 40.0     # [m] around the run
FOOTPRINT_SHARE = 0.1  # occupied share of the footprint's costmap cells that counts as "standing on an obstacle"
STOP_SPEED = 0.05     # [m/s]
STOP_MIN = 2.0        # [s] a stop this long is drawn and logged
GAP_PLAN = 0.5        # [s] planning silent this long is logged
GAP_POSE = 0.3        # [s]
JUMP = 1.0            # [m] between consecutive poses
STARTUP = 2.0         # [s] poses before a jump this early are the localization initialising
CLOUD_RATE = 5.0      # [Hz]
CLOUD_VOXEL = 0.1     # [m]
CLOUD_RANGE = 30.0    # [m]


# ---------------------------------------------------------------- ROS messages

class Ros:
    """rosbags typestore with defaults for every field, so a message can be built from the fields that matter."""

    def __init__(self):
        from rosbags.typesys import Stores, get_typestore
        self.ts = get_typestore(Stores.ROS2_HUMBLE)

    def new(self, typename, **kw):
        obj = self._default(typename)
        for k, v in kw.items():
            setattr(obj, k, v)
        return obj

    def _default(self, typename):
        _, fields = self.ts.fielddefs[typename]
        return self.ts.types[typename](**{name: self._field(nt, desc) for name, (nt, desc) in fields})

    def _field(self, nt, desc):
        kind = getattr(nt, "value", nt)
        if kind == 1:
            base = desc[0]
            return "" if base == "string" else False if base == "bool" else 0.0 if base.startswith("float") else 0
        if kind == 2:
            return self._default(desc)
        sub, n = desc
        sub_kind = getattr(sub[0], "value", sub[0])
        if sub_kind == 1 and sub[1][0] in NP_BASE:
            return np.zeros(n if kind == 3 else 0, dtype=NP_BASE[sub[1][0]])
        return [self._field(*sub) for _ in range(n)] if kind == 3 else []

    def decode(self, data, typename):
        return self.ts.deserialize_cdr(data, typename)

    def register(self, name, schema_text):
        """Types the humble store does not know (kygi665_msgs/msg/INSPVAX, rhr_interface/...), from the ros2msg
        schema the recorder wrote: the main definition, then "MSG: pkg/Type" sections."""
        if name in self.ts.types:
            return
        from rosbags.typesys import get_types_from_msg
        parts = re.split(r"^=+\s*$", schema_text, flags=re.M)
        types = {}
        for k, part in enumerate(parts):
            body, typ = part, name
            if k:
                m = re.search(r"^MSG:\s*(\S+)\s*$", part, re.M)
                if not m:
                    continue
                typ = m.group(1)
                if typ.count("/") == 1:
                    typ = typ.replace("/", "/msg/")
                body = part[m.end():]
            if typ not in self.ts.types:
                types.update(get_types_from_msg(body, typ))
        self.ts.register(types)

    def encode(self, msg, typename):
        return bytes(self.ts.serialize_cdr(msg, typename))

    def msgdef(self, typename):
        return self.ts.generate_msgdef(typename, ros_version=2)[0]

    # small builders
    def time(self, t):
        ns = int(round(t * 1e9))
        sec, nsec = divmod(ns, 1000000000)
        return self.new("builtin_interfaces/msg/Time", sec=int(sec), nanosec=int(nsec))

    def header(self, t, frame):
        return self.new("std_msgs/msg/Header", stamp=self.time(t), frame_id=frame)

    def point(self, x, y, z=0.0):
        return self.new("geometry_msgs/msg/Point", x=float(x), y=float(y), z=float(z))

    def color(self, rgba):
        return self.new("std_msgs/msg/ColorRGBA", r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]),
                        a=float(rgba[3]))

    def pose(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        return self.new("geometry_msgs/msg/Pose", position=self.point(x, y, z),
                        orientation=self.new("geometry_msgs/msg/Quaternion", x=0.0, y=0.0, z=math.sin(yaw / 2),
                                             w=math.cos(yaw / 2)))

    def marker(self, t, frame, ns, mid, mtype, pose=None, scale=(1.0, 1.0, 1.0), color=WHITE, points=None,
               colors=None, text="", frame_locked=False):
        return self.new("visualization_msgs/msg/Marker", header=self.header(t, frame), ns=ns, id=int(mid),
                        type=int(mtype), action=0, pose=pose if pose is not None else self.pose(),
                        scale=self.new("geometry_msgs/msg/Vector3", x=float(scale[0]), y=float(scale[1]),
                                       z=float(scale[2])),
                        color=self.color(color), frame_locked=frame_locked,
                        points=[self.point(*p) for p in (points if points is not None else [])],
                        colors=[self.color(c) for c in (colors if colors is not None else [])], text=text)

    def delete_all(self, t, frame):
        m = self.new("visualization_msgs/msg/Marker", header=self.header(t, frame))
        m.action = 3
        return m


ARROW, CUBE, SPHERE, CYLINDER, LINE_STRIP, LINE_LIST, TEXT, TRIANGLES = 0, 1, 2, 3, 4, 5, 9, 11


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_of(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


# ---------------------------------------------------------------- reading

def mcap_files(paths):
    """The .mcap files of one or more recordings (directories or files)."""
    out = []
    for path in ([paths] if isinstance(paths, str) else paths):
        files = sorted(glob.glob(os.path.join(path, "*.mcap"))) if os.path.isdir(path) else [path]
        if not files or not all(os.path.isfile(f) for f in files):
            raise SystemExit("%s: no .mcap" % path)
        out += files
    return out


def read_recording(path, ros, topics):
    """One pass: light topics decoded, heavy ones (grids, marker arrays, clouds) kept as raw bytes."""
    from mcap.reader import make_reader

    rec = dict(tf=defaultdict(list), odom=[], cmd=[], strings=[], goals=[], plans=[], grids=defaultdict(list),
               markers=defaultdict(list), clouds=[], diagnostics=[], types={}, counts=defaultdict(int),
               loc_raw=defaultdict(list), loc_raw_events=[])
    from loc_viz import SCAN_RATE
    last_scan = -1e18
    wanted = set(topics.values())
    last_cloud = -1e18
    for f in mcap_files(path):
        with open(f, "rb") as fh:
            for schema, channel, message in make_reader(fh).iter_messages(topics=sorted(wanted),
                                                                           log_time_order=True):
                topic, typ, log = channel.topic, schema.name, message.log_time * 1e-9
                if topic not in rec["types"] and typ not in ros.ts.types and schema.encoding == "ros2msg":
                    try:
                        ros.register(typ, schema.data.decode("utf-8", "replace"))
                    except Exception as e:      # noqa: BLE001 - a type we cannot read is left out, not fatal
                        print("  %s (%s): cannot read the type (%s), left out" % (topic, typ, e))
                rec["types"][topic] = typ
                if typ not in ros.ts.types:
                    continue
                rec["counts"][topic] += 1
                data = message.data
                if typ == "tf2_msgs/msg/TFMessage":
                    m = ros.decode(data, typ)
                    for tr in m.transforms:
                        tt, q = tr.transform.translation, tr.transform.rotation
                        rec["tf"][(tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/"))].append(
                            (stamp_of(tr.header), log, tt.x, tt.y, tt.z, q.x, q.y, q.z, q.w,
                             topic == topics["tf_static"]))
                elif topic == topics["odom"]:
                    m = ros.decode(data, typ)
                    p, tw = m.pose.pose, m.twist.twist
                    # lightning: covariance[1] = localization valid (1) or failed (0), covariance[0] = system time
                    rec["odom"].append((stamp_of(m.header), log, p.position.x, p.position.y, p.position.z,
                                        yaw_of(p.orientation), tw.linear.x, tw.linear.y, tw.angular.z,
                                        float(m.pose.covariance[1])))
                elif topic == topics["cmd"]:
                    m = ros.decode(data, typ)
                    if typ.endswith("TwistStamped"):
                        m = m.twist
                    rec["cmd"].append((log, m.linear.x, m.linear.y, m.angular.z))
                elif topic == topics["string"]:
                    rec["strings"].append((log, parse_debug_string(ros.decode(data, typ).data)))
                elif topic in (topics["goal_marker"], topics["goal"]):
                    m = ros.decode(data, typ)
                    if typ.endswith("Marker") and m.action != 0:
                        continue
                    pose = m.pose
                    rec["goals"].append((log, stamp_of(m.header), m.header.frame_id, pose.position.x,
                                         pose.position.y, yaw_of(pose.orientation), topic))
                elif topic == topics["plan"]:
                    rec["plans"].append((log, data))
                elif topic in (topics["cost_map"], topics["perception"], topics["global_map"]):
                    rec["grids"][topic].append((log, data))
                elif topic == topics["diagnostic"]:
                    m = ros.decode(data, typ)
                    rec["diagnostics"].append((log, [(s.name, int(s.level), s.message) for s in m.status]))
                elif topic == topics["scan_world"]:
                    if log - last_scan >= 1.0 / SCAN_RATE - 1e-3:
                        rec["loc_raw"]["scan_world"].append((log, data))
                        last_scan = log
                elif topic in (topics["map_cloud"], topics["inspvax"]):
                    rec["loc_raw"]["map_cloud" if topic == topics["map_cloud"] else "inspvax"].append((log, data))
                elif topic == topics["loc_event"]:
                    m = ros.decode(data, typ)
                    text = " / ".join(str(v) for v in vars(m).values() if isinstance(v, str) and v)
                    rec["loc_raw_events"].append((log, text))
                elif topic == topics["cloud"]:
                    if log - last_cloud >= 1.0 / CLOUD_RATE - 1e-3:
                        rec["clouds"].append((log, data))
                        last_cloud = log
                else:
                    for key, _ in OTHER_MARKERS:
                        if topic == topics[key]:
                            rec["markers"][key].append((log, typ, data))
    return rec


def parse_debug_string(text):
    """planning's DebugDatas::ConvertToString: one "key: value" per line."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- time and pose

class Clock:
    """The recorder's log time (the server's wall clock on a replay) -> the bag's time, from messages that have both
    (odometry, TF): the newest stamp seen by then. Identity when the recording has none (a live recording)."""

    def __init__(self, log, stamp):
        log, stamp = np.asarray(log, float), np.asarray(stamp, float)
        ok = stamp > 1.0
        log, stamp = log[ok], stamp[ok]
        if len(log) == 0:
            self.log = None
            return
        order = np.argsort(log, kind="stable")
        self.log = log[order]
        self.stamp = np.maximum.accumulate(stamp[order])
        self.offset = float(np.median(self.log - self.stamp))

    def __call__(self, t):
        if self.log is None:
            return np.asarray(t, float) if np.ndim(t) else float(t)
        t = np.asarray(t, float)
        out = np.interp(t, self.log, self.stamp)
        out = np.where(t < self.log[0], t - (self.log[0] - self.stamp[0]), out)
        out = np.where(t > self.log[-1], t - (self.log[-1] - self.stamp[-1]), out)
        return out if out.ndim else float(out)


class Track:
    """map -> base_link over time: flattened (x, y, yaw) and the full pose for the point cloud."""

    def __init__(self, rows):
        a = np.asarray(rows, float)   # stamp, x, y, z, qx, qy, qz, qw
        a = a[np.argsort(a[:, 0], kind="stable")]
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 1e-6])
        a = a[keep]
        # the localization's first poses before it initialises (T47: one at the origin 142 m away): drop what comes
        # before a jump in the first STARTUP seconds
        self.dropped = 0
        jumps = np.nonzero(np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2])) > JUMP)[0]
        early = [j for j in jumps if a[j + 1, 0] - a[0, 0] < STARTUP]
        if early and len(a) - (early[-1] + 1) >= 2:
            self.dropped = early[-1] + 1
            a = a[self.dropped:]
        self.t, self.x, self.y, self.z = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        self.q = a[:, 4:8]
        qx, qy, qz, qw = self.q.T
        self.yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        d = np.hypot(np.diff(self.x), np.diff(self.y))
        self.breaks = np.nonzero(d > JUMP)[0] + 1      # a later jump splits the trail
        d[d > JUMP] = 0.0
        self.s = np.concatenate([[0.0], np.cumsum(d)])

    def __len__(self):
        return len(self.t)

    def at(self, t):
        return (float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)),
                float(np.interp(t, self.t, self.yaw)))

    def full(self, t):
        """(R 3x3, p 3) of the nearest sample."""
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        return quat_matrix(*self.q[i]), np.array([self.x[i], self.y[i], self.z[i]])

    def speed(self, t, window=0.2):
        a = np.interp(t - window / 2, self.t, self.s)
        b = np.interp(t + window / 2, self.t, self.s)
        return float((b - a) / window)


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def latest(times, t):
    """Index of the last element of the sorted list `times` that is <= t, or -1."""
    return bisect_right(times, t) - 1


# ---------------------------------------------------------------- map and crosswalks

def read_pgm(raw):
    """P5 (binary) or P2 PGM -> uint8 (H, W); header comments allowed."""
    pos, tokens = 0, []
    while len(tokens) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            pos = raw.index(b"\n", pos) + 1
            continue
        end = pos
        while not raw[end:end + 1].isspace():
            end += 1
        tokens.append(raw[pos:end])
        pos = end
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else np.dtype(">u2")
        img = np.frombuffer(raw, dtype=dtype, count=w * h, offset=pos).reshape(h, w)
    elif magic == b"P2":
        img = np.array(raw[pos:].split()[:w * h], dtype=np.int64).reshape(h, w)
    else:
        raise ValueError("not a PGM (%r)" % magic)
    if maxval != 255:
        img = img.astype(np.float64) * 255.0 / maxval
    return img.astype(np.uint8)


def read_image(name, raw):
    if name.lower().endswith((".pgm", ".pnm")):
        return read_pgm(raw)
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except ImportError:
        pass
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"))


def parse_map_yaml(text):
    """The few keys of a ROS map_server yaml (no yaml dependency). The origin inline ([x, y, yaw]) or as a block
    list (one "- value" per line; the dog's c1adbe7b, T53)."""
    out, block = {}, None
    for line in text.splitlines():
        line = line.split("#")[0]
        item = line.strip()
        if block is not None and item.startswith("-"):
            out[block] = (out[block] + "," if out[block] else "") + item[1:].strip()
            continue
        block = None
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip("'\"")
        if not out[k.strip()]:
            block = k.strip()
    origin = [float(v) for v in (out.get("origin") or "[0, 0, 0]").strip("[]").split(",") if v.strip()]
    return dict(image=out.get("image", ""), resolution=float(out.get("resolution", 0.05)), origin=origin,
                negate=int(float(out.get("negate", 0))), free_thresh=float(out.get("free_thresh", 0.196)),
                occupied_thresh=float(out.get("occupied_thresh", 0.65)))


class Source:
    """Files of a map: a directory, a zip, or a single map yaml."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.lower().endswith(".zip") else None

    def names(self):
        if self.zip:
            return self.zip.namelist()
        if os.path.isfile(self.path):
            return [self.path]
        return [os.path.relpath(p, self.path) for p in glob.glob(os.path.join(self.path, "**", "*"),
                                                                 recursive=True)]

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        full = name if os.path.isabs(name) or os.path.isfile(self.path) else os.path.join(self.path, name)
        with open(full, "rb") as f:
            return f.read()

    def find(self, pattern):
        """Names matching a regex, the shortest first (2d/<id>/map.yaml before deeper copies)."""
        return sorted((n for n in self.names() if re.search(pattern, n.replace("\\", "/"))), key=len)


def load_map(path):
    """-> dict(img uint8 HxW, resolution, origin, thresholds, image name, osm bytes or None). Reads the image
    planning reads: <name>.pgm for a <name>.yaml given directly (GlobalMapManager reads input_name + ".pgm",
    whatever the yaml's image says: CJGC.yaml says map.pgm, another map in the same folder); for a site map
    directory or zip, the first floor's map_image_file of global_map.yaml (GlobalMapManager::ReloadMap; ad68517c:
    2d/<id>/map.png, 2537 cells off its map.pgm)."""
    direct = os.path.isfile(path) and path.lower().endswith((".yaml", ".yml"))
    if direct:
        src = Source(os.path.dirname(os.path.abspath(path)))
        yaml_name = os.path.basename(path)
    else:
        src = Source(path)
        cands = src.find(r"(^|/)2d/[^/]+/map\.yaml$") or src.find(r"(^|/)map\.yaml$")
        if not cands:
            raise SystemExit("%s: no 2d/<id>/map.yaml or map.yaml" % path)
        yaml_name = cands[0]
    meta = parse_map_yaml(src.read(yaml_name).decode("utf-8", "replace"))
    folder = os.path.dirname(yaml_name)
    image = os.path.join(folder, meta["image"]) if folder else meta["image"]
    if direct:
        stem = os.path.splitext(yaml_name)[0] + ".pgm"
        if os.path.isfile(os.path.join(src.path, stem)):
            image = stem
    else:
        for g in src.find(r"(^|/)global_map\.yaml$")[:1]:
            text = src.read(g).decode("utf-8", "replace")
            img_m = re.search(r"map_image_file:\s*(\S+)", text)
            yml_m = re.search(r"map_yaml_file:\s*(\S+)", text)
            if img_m and yml_m:
                base = os.path.dirname(g)
                join = (lambda a, b: a + "/" + b if a else b) if src.zip else os.path.join
                yaml_name, image = join(base, yml_m.group(1)), join(base, img_m.group(1))
                meta = parse_map_yaml(src.read(yaml_name).decode("utf-8", "replace"))
                folder = os.path.dirname(yaml_name)
    if src.zip:
        image = image.replace("\\", "/")
    try:
        meta["img"] = read_image(image, src.read(image))
    except ImportError:          # a png without cv2 or PIL: the map yaml's own image (a pgm) instead
        image = os.path.join(folder, meta["image"]) if folder else meta["image"]
        meta["img"] = read_image(image, src.read(image.replace("\\", "/") if src.zip else image))
    meta["image_name"] = image
    meta["name"] = "%s:%s" % (os.path.basename(path.rstrip("/")), image)
    osm = os.path.join(folder, "semantic.osm") if folder else "semantic.osm"
    try:
        meta["osm"] = src.read(osm.replace("\\", "/") if src.zip else osm)
    except (KeyError, FileNotFoundError, OSError):
        meta["osm"] = None
    return meta


def crop_route(meta, xmin, ymin, xmax, ymax):
    """Crop of the map around the box: (data int8 rows from ymin up, origin x, origin y, width, height); route
    (free) cells 100, the rest 0."""
    img, res = meta["img"], meta["resolution"]
    ox, oy = meta["origin"][0], meta["origin"][1]
    h, w = img.shape
    c0 = int(np.clip(math.floor((xmin - ox) / res), 0, w))
    c1 = int(np.clip(math.ceil((xmax - ox) / res), 0, w))
    r0 = int(np.clip(h - math.ceil((ymax - oy) / res), 0, h))
    r1 = int(np.clip(h - math.floor((ymin - oy) / res), 0, h))
    if c1 <= c0 or r1 <= r0:
        return None
    sub = img[r0:r1, c0:c1][::-1].astype(np.float64)
    occ = sub / 255.0 if meta["negate"] else (255.0 - sub) / 255.0
    data = np.where(occ < meta["free_thresh"], 100, 0).astype(np.int8)
    return data, ox + c0 * res, oy + (h - r1) * res, c1 - c0, r1 - r0


def read_crosswalks(osm_bytes):
    """[(relation id, (N, 2) polygon)] of type=multipolygon subtype=crosswalk (same rule as
    dog_perception/semantic_map.py and planning's osm_parser)."""
    root = ET.fromstring(osm_bytes)
    nodes = {}
    for n in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            nodes[n.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    ways = {w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.findall("way")}
    out = []
    for r in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        if tags.get("type") != "multipolygon" or tags.get("subtype") != "crosswalk":
            continue
        outer = [m for m in r.findall("member") if m.get("type") == "way" and m.get("role") == "outer"]
        if len(outer) != 1 or outer[0].get("ref") not in ways:
            continue
        pts = [nodes[i] for i in ways[outer[0].get("ref")] if i in nodes]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            out.append((r.get("id"), np.asarray(pts, dtype=float)))
    return out


def triangulate(poly):
    """Ear clipping of a simple polygon -> list of index triples."""
    p = np.asarray(poly, float)
    idx = list(range(len(p)))
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:
        idx.reverse()
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            ab, ac = p[b] - p[a], p[c] - p[a]
            if ab[0] * ac[1] - ab[1] * ac[0] <= 1e-12:
                continue
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                if _in_triangle(p[j], p[a], p[b], p[c]):
                    inside = True
                    break
            if not inside:
                tris.append((a, b, c))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _in_triangle(q, a, b, c):
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = cross(a, b, q), cross(b, c, q), cross(c, a, q)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def crosswalk_length(poly):
    """Length across the road [m]: the longer side of the minimum-area rectangle."""
    p = np.asarray(poly, float)
    best = None
    for i in range(len(p)):
        e = p[(i + 1) % len(p)] - p[i]
        n = math.hypot(*e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        dims = (np.ptp(p @ u), np.ptp(p @ v))
        if best is None or dims[0] * dims[1] < best[0] * best[1]:
            best = dims
    return max(best) if best else 0.0


# ---------------------------------------------------------------- grids

class GridStats:
    """Occupied cells under the robot's footprint and the nearest occupied cell, for grids in base_link."""

    def __init__(self, footprint):
        self.length, self.width = footprint
        self.cache = {}

    def __call__(self, grid):
        info = grid.info
        o = info.origin
        key = (info.width, info.height, round(info.resolution, 6), round(o.position.x, 4), round(o.position.y, 4),
               round(yaw_of(o.orientation), 4))
        if key not in self.cache:
            w, h, res = info.width, info.height, info.resolution
            cx = (np.arange(w) + 0.5) * res
            cy = (np.arange(h) + 0.5) * res
            gx, gy = np.meshgrid(cx, cy)
            yaw = yaw_of(o.orientation)
            x = o.position.x + math.cos(yaw) * gx - math.sin(yaw) * gy
            y = o.position.y + math.sin(yaw) * gx + math.cos(yaw) * gy
            inside = (np.abs(x) <= self.length / 2) & (np.abs(y) <= self.width / 2)
            self.cache[key] = (inside.ravel(), np.hypot(x, y).ravel())
        inside, dist = self.cache[key]
        data = np.asarray(grid.data)
        if data.size != inside.size:
            return None, None
        occ = data >= 50
        nearest = float(dist[occ].min()) if occ.any() else None
        self.last_occupied = int(np.count_nonzero(occ))
        return int(np.count_nonzero(occ & inside)), nearest


# ---------------------------------------------------------------- output

class Out:
    def __init__(self, ros):
        self.ros = ros
        self.msgs = []
        self.channels = {}   # topic -> (kind, type, schema)
        self.seq = 0

    def ros_msg(self, topic, typename, t, msg):
        self.channels.setdefault(topic, ("ros2", typename, None))
        self._add(topic, t, self.ros.encode(msg, typename))

    def json_msg(self, topic, name, schema, t, obj):
        self.channels.setdefault(topic, ("json", name, schema))
        self._add(topic, t, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _add(self, topic, t, data):
        self.msgs.append((int(round(t * 1e9)), self.seq, topic, data))
        self.seq += 1

    def counts(self):
        c = defaultdict(int)
        for _, _, topic, _ in self.msgs:
            c[topic] += 1
        return c

    def write(self, path):
        from mcap.writer import CompressionType, Writer
        try:
            import zstandard  # noqa: F401
            comp = CompressionType.ZSTD
        except ImportError:
            try:
                import lz4  # noqa: F401
                comp = CompressionType.LZ4
            except ImportError:
                comp = CompressionType.NONE
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.msgs.sort(key=lambda m: (m[0], m[1]))
        with open(path, "wb") as f:
            w = Writer(f, chunk_size=4 * 1024 * 1024, compression=comp)
            w.start(profile="ros2", library="planning_mcap_viz")
            schema_ids, channel_ids = {}, {}
            for topic, (kind, typ, schema) in sorted(self.channels.items()):
                if kind == "ros2":
                    if typ not in schema_ids:
                        schema_ids[typ] = w.register_schema(name=typ, encoding="ros2msg",
                                                            data=self.ros.msgdef(typ).encode("utf-8"))
                    channel_ids[topic] = w.register_channel(topic=topic, message_encoding="cdr",
                                                            schema_id=schema_ids[typ])
                else:
                    sid = w.register_schema(name=typ, encoding="jsonschema",
                                            data=json.dumps(schema).encode("utf-8"))
                    channel_ids[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
            seqs = defaultdict(int)
            for t, _, topic, data in self.msgs:
                w.add_message(channel_id=channel_ids[topic], log_time=t, publish_time=t, data=data,
                              sequence=seqs[topic])
                seqs[topic] += 1
            w.finish()


STATE_SCHEMA = {
    "type": "object",
    "title": "planning_viz.State",
    "properties": {k: {"type": ["number", "null"]} for k in (
        "t", "mission_status_id", "mission_type_id", "input_timeout", "timeout_paused", "odom_v", "odom_vx",
        "odom_vy", "odom_omega", "cmd_v", "cmd_vx", "cmd_vy", "cmd_omega", "moving", "goal_x", "goal_y",
        "goal_distance", "path_offset", "global_path_points", "local_path_points", "local_path_length",
        "cost_map_footprint_cells", "cost_map_nearest", "perception_footprint_cells", "perception_nearest",
        "x", "y", "yaw")},
}
STATE_SCHEMA["properties"].update({"mission": {"type": "string"}, "mission_type": {"type": "string"},
                                   "mission_status": {"type": "string"},
                                   "planning": {"type": "object", "additionalProperties": {"type": "number"}}})


def jnum(v, nd=4):
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None



# what each mission state does (src/runnable/mission/tracking/tracking.cc: check_e_* decide the transitions,
# on_entry_s_* run once on entry, spin_s_* every cycle)
STATE_EXPLAIN = {
    "off": ("没有任务，等 /goal_pose", "收到目标（need_start）-> passive"),
    "passive": ("收到目标", "下一周期 -> standby"),
    "standby": ("进来时做一次全局规划：A* 在全局地图上从狗到终点（GlobalMapManager::PlanGlobalPath）",
                "A* 成功（global_plan_flag）-> active；失败 -> fault；要停（need_stop）-> off"),
    "active": ("每周期：从全局路径截一段局部参考（ExtractLocalPath），交给 MPPI 优化出局部路径和速度指令",
               "到终点附近、速度和朝向都满足并停稳一段时间 -> off；暂停或被挡（need_pause / nav_blocked）-> standstill"),
    "standstill": ("暂停，速度 0（输入超时、被挡、外部暂停）", "恢复（need_resume）-> active；要停 -> off"),
    "fault": ("全局规划失败", "要停 -> off"),
    "idle": ("没有任务", "收到目标 -> 跟踪任务"),
}
CODE = {
    "input": "src/runnable/communication/ros_bridge/ros_bridge.cc：订阅 /lightning/odom、/map/occupancy_grid、"
             "/goal_pose；数据超时（timeout_pause_duration）就暂停任务",
    "global": "src/runnable/preprocess/global_map_manager/global_map_manager.cc：读 CJGC 路网图（255 = 能走）；"
              "PlanGlobalPath = A*（grid_path_planner.cc，起点不在路上先挪到最近的路上）+ 平滑；ExtractLocalPath",
    "costmap": "src/runnable/preprocess/cost_map_production/cost_map_production.cc：Proc = 全部置空闲 -> "
               "AddObstacleFromMap（全局图按狗的位姿叠进来）-> AddObstacleFromPerception（感知栅格的障碍）-> 距离场；"
               "obstacle 层 255 = 空闲、0 = 障碍",
    "fsm": "src/runnable/mission/tracking/tracking.cc：Proc -> mission_fsm_ctx_.Proc()",
    "local": "Tracking::spin_s_active -> MppiPlanner::solver（src/algorithm/mppi_planner/mppi_planner.cc，"
             "参考路径 + 当前速度，代价在 critics/）-> getOptimizePath（局部路径）、getVelocityCmd（速度）",
    "output": "RosBridge 发 /cmd_vel（MPPI 的速度，base_link 里：vx 向前、vy 向左、ω 逆时针）",
}


def _num(v, fmt):
    return fmt % v if v is not None and v == v else "没有"


def describe_cmd(vx, vy, w):
    if vx is None:
        return "没有指令"
    bits = []
    if abs(vx) > 0.02:
        bits.append("%s %.2f m/s" % ("向前" if vx > 0 else "向后", abs(vx)))
    if abs(vy or 0) > 0.02:
        bits.append("%s平移 %.2f m/s" % ("向左" if vy > 0 else "向右", abs(vy)))
    if abs(w or 0) > 0.02:
        bits.append("%s %.2f rad/s" % ("左转" if w > 0 else "右转", abs(w)))
    return "，".join(bits) if bits else "停（全是 0）"


def planning_flow(s, t0):
    """One planning cycle as the code runs it, with this cycle's numbers: for learning the flow."""
    raw = s["raw"]
    plan_t = raw.get("planning")
    stat = s["mission_status"] or ("idle" if s["mission_type"] == "idle" else "")
    doing, nxt = STATE_EXPLAIN.get(stat, ("", ""))
    return {
        "时间_s": round(s["t"] - t0, 2),
        "① 输入（RosBridge）": {
            "代码": CODE["input"],
            "里程计": "x %.2f, y %.2f, 朝向 %.0f°, 速度 %s m/s" % (s["x"], s["y"], math.degrees(s["yaw"]),
                                                          _num(s["odom_v"], "%.2f")),
            "里程计到规划的延迟_s": _num(plan_t - raw["odom_ts"] if plan_t and raw.get("odom_ts") else None, "%.3f"),
            "感知栅格到规划的延迟_s": _num(plan_t - raw["occ_ts"] if plan_t and raw.get("occ_ts") else None, "%.3f"),
            "输入超时": {None: "没有", 0.0: "否", 1.0: "是"}.get(s["input_timeout"], s["input_timeout"]),
            "终点": "没有" if not s["goal"] else "(%.1f, %.1f)" % (s["goal"][1], s["goal"][2]),
        },
        "② 全局地图和全局路径（GlobalMapManager）": {
            "代码": CODE["global"],
            "全局路径点数": _num(s["global_path_points"], "%d"),
            "到终点_m": _num(s["goal_distance"], "%.2f"),
            "狗离全局路径_m": _num(s["path_offset"], "%.2f"),
        },
        "③ 局部代价地图（CostMapProduction）": {
            "代码": CODE["costmap"],
            "范围": "狗前 10 m、后 5 m、左右各 5 m，5 cm 一格，跟着狗转",
            "障碍格数": _num(s["cost_map_occupied"], "%d"),
            "狗脚下的障碍格（脚印 1.0 x 0.4 m）": _num(s["cost_map_footprint_cells"], "%d"),
            "感知栅格的障碍格数": _num(s["perception_occupied"], "%d"),
            "最近的感知障碍_m": _num(s["perception_nearest"], "%.2f"),
        },
        "④ 任务状态机（Tracking）": {
            "代码": CODE["fsm"],
            "任务": s["mission_type"] or "没有",
            "状态": stat or "没有",
            "这个状态在做什么": doing,
            "怎么离开这个状态": nxt,
        },
        "⑤ 局部规划（D20 是 MPPI，tracking_conf.planner_type；T48 之前的录包是 TEB）": {
            "代码": CODE["local"],
            "参考路径（MPPI 的输入）点数": _num(len(s["reference"]) if s["reference"] else None, "%d"),
            "参考路径长_m": _num(polyline_length(s["reference"]) if s["reference"] else None, "%.2f"),
            "优化后的局部路径点数": _num(s["local_path_points"], "%d"),
            "局部路径长_m": _num(s["local_path_length"], "%.2f"),
        },
        "⑥ 输出（/cmd_vel）": {
            "代码": CODE["output"],
            "指令": describe_cmd(s["cmd_vx"], s["cmd_vy"], s["cmd_omega"]),
            "实际速度_m/s": _num(s["odom_v"], "%.2f"),
        },
    }


FLOW_SCHEMA = {"type": "object", "title": "planning_viz.Flow", "additionalProperties": True}
LEGEND = {
    "怎么看": "3D 是俯视，跟着狗。左键拖动平移、滚轮缩放、右键拖动转视角。右边的曲线和下面的状态条和 3D 同一个时间轴，"
           "点曲线上任一处就跳到那一刻。",
    "图层（3D 面板左边的话题列表可以逐个开关）": {
        "/viz/route_map 灰蓝": "规划地图（CJGC 路网）上能走的路，约 1.7 m 宽的一条条线",
        "/viz/crosswalks 黄": "地图标的人行横道，写着编号和横穿长度",
        "/viz/cost_map 半透明红": "规划的局部代价地图里的障碍（狗周围 15 x 10 m）：路外面全是障碍，再加感知看到的",
        "/viz/perception_grid 橙": "感知栅格里的障碍（雷达看到的近处障碍）",
        "/viz/trail 彩色线": "狗走过的路，按速度着色（蓝慢、绿、黄、红快）；白圈是停下的地方",
        "/viz/plan 蓝线": "全局路径（A* 在规划地图上找的，进 standby 时算一次）",
        "/viz/plan 紫线": "局部参考路径：每周期从全局路径截出来交给 MPPI 的那一段（T49 起的录包才有）",
        "/viz/plan 绿线和绿框": "MPPI 优化后的局部路径和沿路的脚印",
        "/viz/plan 红/绿箭头": "跟踪目标点（红：要求到点时的朝向）",
        "/viz/plan 粉色圆": "规划用来查碰撞的圆",
        "/viz/plan 红圆 + 终点": "终点",
        "/viz/robot": "狗（青色，白三角是朝前）；浅蓝箭头 = 实际速度（1 s 走多远），橙箭头 = 速度指令，橙弧 = 转向指令；"
                      "后面的文字是这一刻的状态",
        "/viz/loc/map_cloud": "定位地图的点云（按高度着色）",
        "/viz/loc/scan": "这一帧雷达点云，按 LIO 的位姿放进地图；颜色是离地图最近点的距离：绿 = 贴在地图上，红 = 对不上",
        "/viz/loc/ins": "INS 的轨迹（紫 = RTK 固定解，灰 = 不是），和它现在的位置",
    },
    "面板": {
        "/viz/flow（规划流程）": "每个规划周期按代码的顺序：输入 -> 全局 -> 代价地图 -> 状态机 -> MPPI -> 输出，每步写这一周期的数和代码位置",
        "/viz/loc/flow（定位流程）": "lightning 每一步：IMU 预测 -> 点云 -> 配准 -> 健康 -> INS -> 输出",
        "/viz/state、/viz/loc/state": "画曲线用的数",
        "/viz/events": "事件：任务状态变化、超时、停下、断流、跳变、狗站到路外、定位故障、INS、规划和定位自己的警告",
    },
}

# ---------------------------------------------------------------- conversion

class Converter:
    def __init__(self, args):
        self.args = args
        self.ros = Ros()
        self.out = Out(self.ros)
        self.events = []        # (t, level, name, text)
        self.notes = []         # summary lines
        self.fixed, self.base = args.fixed_frame, args.base_frame

    def event(self, t, level, text, name="viz"):
        self.events.append((float(t), LOG[level], name, text))

    def run(self):
        a, ros = self.args, self.ros
        topics = dict(TOPICS)
        topics.update(odom=a.odom_topic, cloud=a.cloud_topic)
        rec = read_recording([a.recording] + list(a.extra), ros, topics)
        self.rec, self.topics = rec, topics
        self.build_clock_track()
        if self.track is None:
            raise SystemExit("%s: no %s -> %s on /tf and no %s: nothing to place the grids with"
                             % (a.recording, self.fixed, self.base, topics["odom"]))
        self.t0, self.t1 = float(self.track.t[0]), float(self.track.t[-1])
        self.write_tf()
        self.goals = self.collect_goals()
        self.plan_times, self.plan_info = self.convert_plans()
        self.grid_times = {}
        self.convert_grids()
        self.convert_other_markers()
        self.states = self.build_states()
        self.note_mppi()
        self.write_robot()
        self.write_trail()
        self.write_static()
        self.write_legend()
        self.convert_cloud()
        self.loc = None
        if any(self.rec["loc_raw"].values()) or self.args.loc_map:
            from loc_viz import LocLayers
            self.loc = LocLayers(self).run()
        self.collect_events()
        self.write_events()
        self.out.write(a.output)
        return self

    # -- time base, pose
    def build_clock_track(self):
        rec = self.rec
        pairs_log, pairs_stamp = [], []
        for row in rec["odom"]:
            pairs_log.append(row[1])
            pairs_stamp.append(row[0])
        edge = rec["tf"].get((self.fixed, self.base), [])
        if not pairs_log:
            for row in edge:
                if not row[9]:
                    pairs_log.append(row[1])
                    pairs_stamp.append(row[0])
        self.clock = Clock(pairs_log, pairs_stamp)
        if self.clock.log is not None:
            self.notes.append("time base: the bag's stamps; the recorder's clock was %.1f s (%.1f days) ahead"
                              % (self.clock.offset, self.clock.offset / 86400))
        rows = [(r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in edge if r[0] > 0]
        source = "/tf %s -> %s" % (self.fixed, self.base)
        if len(rows) < 2 and rec["odom"]:
            rows = [(r[0], r[2], r[3], r[4], 0.0, 0.0, math.sin(r[5] / 2), math.cos(r[5] / 2)) for r in rec["odom"]]
            source = self.topics["odom"]
        self.track = Track(rows) if len(rows) >= 2 else None
        self.raw_pose = np.asarray(sorted(rows), float) if rows else np.zeros((0, 8))
        if self.track is not None:
            self.notes.append("pose: %s, %d samples%s" % (
                source, len(self.track), ", the first %d dropped (before a start-up jump)" % self.track.dropped
                if self.track.dropped else ""))
        odom = sorted(rec["odom"])
        self.odom_t = [r[0] for r in odom]
        self.odom = odom

    def write_tf(self):
        ros, tr = self.ros, self.track
        for i in range(len(tr)):
            t, yaw = float(tr.t[i]), float(tr.yaw[i])
            ts = ros.new("geometry_msgs/msg/TransformStamped", header=ros.header(t, self.fixed),
                         child_frame_id=self.base,
                         transform=ros.new("geometry_msgs/msg/Transform",
                                           translation=ros.new("geometry_msgs/msg/Vector3", x=float(tr.x[i]),
                                                               y=float(tr.y[i]), z=0.0),
                                           rotation=ros.new("geometry_msgs/msg/Quaternion", x=0.0, y=0.0,
                                                            z=math.sin(yaw / 2), w=math.cos(yaw / 2))))
            self.out.ros_msg("/tf", "tf2_msgs/msg/TFMessage", t, ros.new("tf2_msgs/msg/TFMessage", transforms=[ts]))

    def data_time(self, header_stamp, log):
        """A message's time on the bag's time base: its stamp when it has one, else its log time converted."""
        return header_stamp if header_stamp > 1.0 else self.clock(log)

    def to_fixed(self, frame, t, x, y, yaw=0.0):
        frame = frame.lstrip("/")
        if frame in ("", self.fixed):
            return x, y, yaw
        if frame == self.base:
            rx, ry, ryaw = self.track.at(t)
            c, s = math.cos(ryaw), math.sin(ryaw)
            return rx + c * x - s * y, ry + s * x + c * y, ryaw + yaw
        return None

    # -- goals
    def collect_goals(self):
        goals = []
        for log, stamp, frame, x, y, yaw, topic in self.rec["goals"]:
            t = self.data_time(stamp, log)
            p = self.to_fixed(frame, t, x, y, yaw)
            if p is not None:
                goals.append((t, p[0], p[1], p[2]))
        goals.sort()
        return goals

    def goal_at(self, t):
        i = latest([g[0] for g in self.goals], t) if self.goals else -1
        return self.goals[i] if i >= 0 else None

    # -- markers
    def flatten_marker(self, mk, t, layer_z, color=None, width=None):
        """In place: into the fixed frame at height layer_z, stamped t, no lifetime. False if its frame is unknown."""
        ros = self.ros
        p, q = mk.pose.position, mk.pose.orientation
        placed = self.to_fixed(mk.header.frame_id, t, p.x, p.y, yaw_of(q))
        if placed is None:
            return False
        mk.header = ros.header(t, self.fixed)
        mk.pose = ros.pose(placed[0], placed[1], layer_z, placed[2])
        for pt in mk.points:
            pt.z = 0.0
        mk.lifetime = ros.new("builtin_interfaces/msg/Duration", sec=0, nanosec=0)
        mk.frame_locked = False
        if mk.type in (SPHERE, CYLINDER, CUBE):
            mk.scale.z = min(mk.scale.z, 0.01)
        if color is not None:
            mk.color = ros.color(color)
            mk.colors = []
        elif mk.color.a < 0.05:
            mk.color.a = 1.0
        if width is not None and mk.type in (LINE_STRIP, LINE_LIST):
            mk.scale.x = width
        return True

    def convert_plans(self):
        ros = self.ros
        times, info = [], []
        typ = self.rec["types"].get(self.topics["plan"])
        last_t = None
        for log, data in self.rec["plans"]:
            arr = ros.decode(data, typ)
            t = self.clock(log)
            stamps = [stamp_of(m.header) for m in arr.markers if m.action == 0 and stamp_of(m.header) > 1.0]
            if stamps:
                t = max(stamps)
            out = [ros.delete_all(t, self.fixed)]
            gp, lp, target, reference = None, None, None, None
            for mk in arr.markers:
                if mk.action != 0:
                    continue
                style = PLAN_STYLE.get(mk.ns, ("other", None, None))
                if mk.ns == "global_path_2d":
                    gp = [(pt.x, pt.y) for pt in mk.points]
                elif mk.ns == "local_path_2d":
                    lp = [(pt.x, pt.y) for pt in mk.points]
                elif mk.ns == "target_pt":
                    target = (mk.pose.position.x, mk.pose.position.y)
                elif mk.ns == "local_reference_2d":
                    reference = [(pt.x, pt.y) for pt in mk.points]
                if style is None:
                    continue
                layer, color, width = style
                if mk.ns in ("global_path_2d", "local_path_2d") and len(mk.points) < 2:
                    continue
                if self.flatten_marker(mk, t, Z[layer], color, width):
                    out.append(mk)
            goal = self.goal_at(t)
            if goal is not None:
                out.extend(self.goal_markers(t, goal))
            self.out.ros_msg("/viz/plan", "visualization_msgs/msg/MarkerArray", t,
                             ros.new("visualization_msgs/msg/MarkerArray", markers=out))
            if last_t is not None and t - last_t > GAP_PLAN:
                self.event(last_t, "warn", "规划 %.2f s 没有输出（%s）" % (t - last_t, self.topics["plan"]))
            last_t = t
            times.append(t)
            info.append(dict(global_path=gp, local_path=lp, target=target, reference=reference))
        order = np.argsort(times, kind="stable")
        return [times[i] for i in order], [info[i] for i in order]

    def goal_markers(self, t, goal):
        ros = self.ros
        _, gx, gy, gyaw = goal
        return [
            ros.marker(t, self.fixed, "goal", 1, CYLINDER, ros.pose(gx, gy, Z["goal"]), (0.6, 0.6, 0.01),
                       (1.0, 0.27, 0.23, 0.85)),
            ros.marker(t, self.fixed, "goal", 2, ARROW, ros.pose(gx, gy, Z["goal"] + 0.005, gyaw), (0.8, 0.08, 0.08),
                       WHITE),
            ros.marker(t, self.fixed, "goal", 3, TEXT, ros.pose(gx, gy + 0.9, Z["label"]), (0, 0, 0.6), WHITE,
                       text="终点"),
        ]

    def convert_other_markers(self):
        ros = self.ros
        for key, out_topic in OTHER_MARKERS:
            rows = self.rec["markers"].get(key, [])
            dropped = 0
            for log, typ, data in rows:
                m = ros.decode(data, typ)
                markers = m.markers if typ.endswith("MarkerArray") else [m]
                stamps = [stamp_of(x.header) for x in markers if x.action == 0 and stamp_of(x.header) > 1.0]
                t = max(stamps) if stamps else self.clock(log)
                out = [ros.delete_all(t, self.fixed)]
                for mk in markers:
                    if mk.action != 0:
                        continue
                    if self.flatten_marker(mk, stamp_of(mk.header) if stamp_of(mk.header) > 1.0 else t, Z["other"]):
                        mk.header = ros.header(t, self.fixed)
                        out.append(mk)
                    else:
                        dropped += 1
                self.out.ros_msg(out_topic, "visualization_msgs/msg/MarkerArray", t,
                                 ros.new("visualization_msgs/msg/MarkerArray", markers=out))
            if rows:
                self.notes.append("%s -> %s: %d arrays%s" % (self.topics[key], out_topic, len(rows),
                                                            ", %d markers in unknown frames" % dropped if dropped
                                                            else ""))

    # -- grids
    def detect_cost_map_values(self):
        """'ros' (100 = obstacle) or 'inverted' (free 100, obstacles 0: ros_debug_bridge before the T47 review),
        from the perception grid's obstacles: planning copies them into the costmap as obstacles."""
        if self.args.cost_map_values != "auto":
            return self.args.cost_map_values, "given"
        ros = self.ros
        cm = self.rec["grids"].get(self.topics["cost_map"], [])
        pg = self.rec["grids"].get(self.topics["perception"], [])
        if not cm or not pg:
            return "ros", "no perception grid to tell by: taken as 100 = obstacle"
        cm_typ, pg_typ = self.rec["types"][self.topics["cost_map"]], self.rec["types"][self.topics["perception"]]
        pg_log = [r[0] for r in pg]
        hits = total = 0
        for log, data in cm[::max(1, len(cm) // 300)]:
            i = latest(pg_log, log)
            if i < 0:
                continue
            g, p = ros.decode(data, cm_typ), ros.decode(pg[i][1], pg_typ)
            if g.header.frame_id != p.header.frame_id:
                continue
            pi, gi = p.info, g.info
            d = np.asarray(p.data).reshape(pi.height, pi.width)
            r, c = np.nonzero(d >= 50)
            x = pi.origin.position.x + (c + 0.5) * pi.resolution - gi.origin.position.x
            y = pi.origin.position.y + (r + 0.5) * pi.resolution - gi.origin.position.y
            cc, rr = np.floor(x / gi.resolution).astype(int), np.floor(y / gi.resolution).astype(int)
            ok = (cc >= 0) & (cc < gi.width) & (rr >= 0) & (rr < gi.height)
            v = np.asarray(g.data).reshape(gi.height, gi.width)[rr[ok], cc[ok]]
            hits += int(np.count_nonzero(v >= 50))
            total += int(len(v))
        if total < 50:
            return "ros", "only %d perception obstacle cells in the costmap's window: taken as 100 = obstacle" % total
        share = hits / total
        verdict = "ros" if share >= 0.5 else "inverted"
        return verdict, "perception obstacles read 100 in %.1f%% of %d costmap cells" % (100 * share, total)

    def convert_grids(self):
        ros = self.ros
        stats = GridStats(self.args.footprint)
        self.grid_stats = {}
        self.cost_values, why = self.detect_cost_map_values()
        if self.rec["grids"].get(self.topics["cost_map"]):
            self.notes.append("costmap values: %s (%s)%s" % (
                "100 = obstacle" if self.cost_values == "ros" else "inverted, free = 100: turned over", why,
                "" if self.cost_values == "ros" else "; ros_debug_bridge before the T47 review"))
        for topic, out_topic, layer in ((self.topics["cost_map"], "/viz/cost_map", "cost_map"),
                                        (self.topics["perception"], "/viz/perception_grid", "perception"),
                                        (self.topics["global_map"], "/viz/global_map", "global_map")):
            rows = self.rec["grids"].get(topic, [])
            if not rows:
                continue
            typ = self.rec["types"][topic]
            times, st, skipped, last = [], [], 0, None
            for log, data in rows:
                g = ros.decode(data, typ)
                if layer in ("cost_map", "global_map") and self.cost_values == "inverted":
                    d = np.asarray(g.data)
                    g.data = np.where(d >= 0, 100 - d, d).astype(np.int8)
                t = self.data_time(stamp_of(g.header), log)
                frame = g.header.frame_id.lstrip("/")
                fp = stats(g) + (stats.last_occupied,) if frame == self.base else (None, None, None)
                o = g.info.origin
                placed = self.to_fixed(frame, t, o.position.x, o.position.y, yaw_of(o.orientation))
                if placed is None:
                    skipped += 1
                    continue
                g.header = ros.header(t, self.fixed)
                g.info.origin = ros.pose(placed[0], placed[1], Z[layer], placed[2])
                self.out.ros_msg(out_topic, "nav_msgs/msg/OccupancyGrid", t, g)
                if last is not None and t - last > 1.0 and layer != "global_map":
                    self.event(last, "warn", "%s %.2f s 没有新栅格" % (topic, t - last))
                last = t
                times.append(t)
                st.append(fp)
            order = np.argsort(times, kind="stable")
            self.grid_stats[layer] = ([times[i] for i in order], [st[i] for i in order])
            self.notes.append("%s -> %s: %d grids%s" % (topic, out_topic, len(times),
                                                       ", %d in unknown frames" % skipped if skipped else ""))

    def grid_at(self, layer, t):
        """(occupied cells under the footprint, nearest occupied [m], occupied cells) of the newest grid."""
        times, st = self.grid_stats.get(layer, ([], []))
        i = latest(times, t)
        return st[i] if i >= 0 and t - times[i] < 1.0 else (None, None, None)

    # -- per-cycle state
    def build_states(self):
        rec = self.rec
        cmd = sorted((self.clock(r[0]),) + tuple(r[1:]) for r in rec["cmd"])
        cmd_t = [c[0] for c in cmd]
        strings = sorted((self.clock(log), d) for log, d in rec["strings"])
        cycle_times = [s[0] for s in strings] if strings else list(self.plan_times)
        string_t = [s[0] for s in strings]
        states = []
        for t in cycle_times:
            raw = strings[latest(string_t, t)][1] if strings else {}
            x, y, yaw = self.track.at(t)
            mtype = int(raw.get("mission_type", -1)) if "mission_type" in raw else None
            mstat = int(raw.get("mission_status", -1)) if "mission_status" in raw and mtype not in (None, 0) else None
            type_name = MISSION_TYPES.get(mtype, "unknown" if mtype is not None else "")
            stat_name = MISSION_STATES.get(mstat, "") if mstat is not None else ("idle" if mtype == 0 else "")
            j = latest(self.odom_t, t)
            if j >= 0 and t - self.odom_t[j] < 0.5:
                o = self.odom[j]
                ovx, ovy, ow = o[6], o[7], o[8]
            else:
                ovx = ovy = ow = None
            k = latest(cmd_t, t)
            if "cmd_vx" in raw:
                cvx, cvy, cw = raw.get("cmd_vx"), raw.get("cmd_vy", 0.0), raw.get("cmd_omega", 0.0)
            elif k >= 0 and t - cmd_t[k] < 0.5:
                cvx, cvy, cw = cmd[k][1], cmd[k][2], cmd[k][3]
            else:
                cvx = cvy = cw = None
            pi = latest(self.plan_times, t)
            plan = self.plan_info[pi] if pi >= 0 and t - self.plan_times[pi] < 1.0 else {}
            gp, lp = plan.get("global_path"), plan.get("local_path")
            goal = self.goal_at(t)
            cm_fp, cm_near, cm_occ = self.grid_at("cost_map", t)
            pg_fp, pg_near, pg_occ = self.grid_at("perception", t)
            states.append(dict(
                t=t, x=x, y=y, yaw=yaw, mission_type=type_name, mission_status=stat_name,
                mission_type_id=mtype, mission_status_id=mstat,
                input_timeout=raw.get("mission_input_timeout"), timeout_paused=raw.get("mission_is_timeout_paused"),
                odom_vx=ovx, odom_vy=ovy, odom_omega=ow,
                odom_v=math.hypot(ovx, ovy) if ovx is not None else self.track.speed(t),
                cmd_vx=cvx, cmd_vy=cvy, cmd_omega=cw,
                cmd_v=math.hypot(cvx, cvy or 0.0) if cvx is not None else None,
                goal=goal, goal_distance=math.hypot(goal[1] - x, goal[2] - y) if goal else None,
                path_offset=polyline_distance(gp, x, y) if gp and len(gp) >= 2 else None,
                global_path_points=len(gp) if gp is not None else None,
                local_path_points=len(lp) if lp is not None else None,
                local_path_length=polyline_length(lp) if lp else None,
                cost_map_footprint_cells=cm_fp, cost_map_nearest=cm_near, cost_map_occupied=cm_occ,
                perception_footprint_cells=pg_fp, perception_nearest=pg_near, perception_occupied=pg_occ,
                reference=plan.get("reference"), local_path=lp, raw=raw))
        for s in states:
            mission = "%s/%s" % (s["mission_type"], s["mission_status"]) if s["mission_type"] else ""
            moving = None
            if s["cmd_vx"] is not None:
                moving = 1 if abs(s["cmd_vx"]) > 0.02 or abs(s["cmd_vy"] or 0) > 0.02 or \
                    abs(s["cmd_omega"] or 0) > 0.02 else 0
            s["moving"] = moving
            obj = dict(t=jnum(s["t"] - self.t0, 3), mission=mission, mission_type=s["mission_type"],
                       mission_status=s["mission_status"])
            for k in ("mission_status_id", "mission_type_id", "input_timeout", "timeout_paused", "odom_v", "odom_vx",
                      "odom_vy", "odom_omega", "cmd_v", "cmd_vx", "cmd_vy", "cmd_omega", "moving", "goal_distance",
                      "path_offset", "global_path_points", "local_path_points", "local_path_length",
                      "cost_map_footprint_cells", "cost_map_nearest", "perception_footprint_cells",
                      "perception_nearest", "x", "y", "yaw"):
                obj[k] = jnum(s[k])
            obj["goal_x"] = jnum(s["goal"][1]) if s["goal"] else None
            obj["goal_y"] = jnum(s["goal"][2]) if s["goal"] else None
            obj["planning"] = {k: jnum(v, 6) for k, v in s["raw"].items() if jnum(v, 6) is not None}
            self.out.json_msg("/viz/state", "planning_viz.State", STATE_SCHEMA, s["t"], obj)
            self.out.json_msg("/viz/flow", "planning_viz.Flow", FLOW_SCHEMA, s["t"], planning_flow(s, self.t0))
        return states

    def state_at(self, t):
        i = latest([s["t"] for s in self.states], t) if self.states else -1
        return self.states[i] if i >= 0 else None

    # -- robot
    def note_mppi(self):
        """How often the local planner gave no command while active, and whether that was the robot standing on
        costmap obstacles (T49 / T50: every such cycle failed, "failed to compute a collision free path")."""
        act = [s for s in self.states if s["mission_status"] == "active" and s["cmd_vx"] is not None]
        if not act:
            return
        zero = [abs(s["cmd_vx"]) < 1e-9 and abs(s["cmd_vy"] or 0.0) < 1e-9 and abs(s["cmd_omega"] or 0.0) < 1e-9
                for s in act]
        on = [(s["cost_map_footprint_cells"] or 0) > 0 for s in act]
        n_on = sum(on)
        self.notes.append("local planner: zero command in %d of %d active cycles (%.0f%%); with costmap obstacles "
                          "under the footprint %d of %d, without %d of %d" % (
                              sum(zero), len(act), 100.0 * sum(zero) / len(act),
                              sum(z for z, o in zip(zero, on) if o), n_on,
                              sum(z for z, o in zip(zero, on) if not o), len(act) - n_on))

    def write_robot(self):
        ros, (length, width) = self.ros, self.args.footprint
        self.limit = self.footprint_limit()
        state_t = [s["t"] for s in self.states]
        n = int((self.t1 - self.t0) * ROBOT_RATE) + 1
        for i in range(n):
            t = self.t0 + i / ROBOT_RATE
            k = latest(state_t, t)
            s = self.states[k] if k >= 0 and t - state_t[k] < 1.0 else None
            j = latest(self.odom_t, t)
            if j >= 0 and t - self.odom_t[j] < 0.5:
                vx, vy = self.odom[j][6], self.odom[j][7]
            else:
                vx, vy = self.track.speed(t), 0.0
            b = self.base
            ms = [ros.delete_all(t, b),
                  ros.marker(t, b, "body", 1, CUBE, ros.pose(0, 0, Z["robot"]), (length, width, 0.01), CYAN,
                             frame_locked=True),
                  ros.marker(t, b, "body", 2, TRIANGLES, ros.pose(0, 0, Z["robot"] + 0.006), (1, 1, 1), WHITE,
                             points=[(length * 0.15, width * 0.4, 0), (length * 0.15, -width * 0.4, 0),
                                     (length * 0.5, 0, 0)], frame_locked=True)]
            if math.hypot(vx, vy) > STOP_SPEED:
                ms.append(ros.marker(t, b, "velocity", 1, ARROW, ros.pose(0, 0, Z["arrows"]), (0.06, 0.14, 0.18),
                                     (0.6, 0.95, 1.0, 1.0), points=[(0, 0, 0), (vx, vy, 0)], frame_locked=True))
            if s is not None and s["cmd_vx"] is not None:
                cvx, cvy, cw = s["cmd_vx"], s["cmd_vy"] or 0.0, s["cmd_omega"] or 0.0
                if math.hypot(cvx, cvy) > 0.02:
                    ms.append(ros.marker(t, b, "command", 1, ARROW, ros.pose(0, 0, Z["arrows"] + 0.01),
                                         (0.05, 0.12, 0.16), ORANGE, points=[(0, 0, 0), (cvx, cvy, 0)],
                                         frame_locked=True))
                if abs(cw) > 0.02:
                    r = 0.75
                    a = np.linspace(0.0, cw * 1.0, 16)
                    ms.append(ros.marker(t, b, "command", 2, LINE_STRIP, ros.pose(0, 0, Z["arrows"] + 0.01),
                                         (0.04, 0, 0), ORANGE,
                                         points=[(r * math.cos(v), r * math.sin(v), 0) for v in a],
                                         frame_locked=True))
            ms.append(ros.marker(t, b, "label", 1, TEXT, ros.pose(-length / 2 - 1.2, 0, Z["label"]), (0, 0, 0.35),
                                 STATE_COLOR.get(s["mission_status"], WHITE) if s else GRAY,
                                 text=self.label(t, s, math.hypot(vx, vy)), frame_locked=True))
            self.out.ros_msg("/viz/robot", "visualization_msgs/msg/MarkerArray", t,
                             ros.new("visualization_msgs/msg/MarkerArray", markers=ms))

    def label(self, t, s, v):
        lines = ["%.1f s" % (t - self.t0)]
        if s is None:
            lines[0] += "  规划没输出"
            lines.append("速度 %.2f m/s" % v)
            return "\n".join(lines)
        mission = MISSION_ZH.get(s["mission_type"], s["mission_type"])
        lines[0] += "  %s %s" % (mission, s["mission_status"])
        short = {"standby": "（A* 全局规划）", "active": "（MPPI 跟着走）", "standstill": "（暂停）", "fault": "（规划失败）",
                 "off": "（等目标）"}.get(s["mission_status"], "")
        lines[0] += short
        cmd = "指令 %.2f m/s  转 %.2f" % (s["cmd_v"], s["cmd_omega"] or 0.0) if s["cmd_v"] is not None else "无指令"
        lines.append("速度 %.2f m/s  %s" % (v, cmd))
        bits = []
        if s["goal_distance"] is not None:
            bits.append("到终点 %.1f m" % s["goal_distance"])
        if s["path_offset"] is not None:
            bits.append("偏离路径 %.2f m" % s["path_offset"])
        if bits:
            lines.append("  ".join(bits))
        if (s["cost_map_footprint_cells"] or 0) >= self.limit:
            lines.append("!! 脚下是障碍（代价地图 %d 格）" % s["cost_map_footprint_cells"])
        if s["timeout_paused"]:
            lines.append("!! 输入超时，任务暂停")
        elif s["input_timeout"]:
            lines.append("!! 输入超时")
        return "\n".join(lines)

    # -- trail and stops
    def find_stops(self):
        tr = self.track
        t = np.arange(self.t0, self.t1, 0.1)
        if len(t) < 2:
            return []
        v = np.array([tr.speed(x, 0.5) for x in t])
        still = v < STOP_SPEED
        stops, i = [], 0
        while i < len(t):
            if still[i]:
                j = i
                while j + 1 < len(t) and still[j + 1]:
                    j += 1
                if t[j] - t[i] >= STOP_MIN:
                    x, y, _ = tr.at(t[i])
                    stops.append((float(t[i]), float(t[j]), x, y))
                i = j + 1
            else:
                i += 1
        return stops

    def write_trail(self):
        ros, tr = self.ros, self.track
        self.stops = self.find_stops()
        n = int((self.t1 - self.t0) * TRAIL_RATE) + 1
        # points every TRAIL_STEP metres along the walked distance
        marks = np.arange(0.0, tr.s[-1] + TRAIL_STEP, TRAIL_STEP)
        pt_t = np.interp(marks, tr.s, tr.t) if tr.s[-1] > 0 else np.array([tr.t[0]])
        pt_x, pt_y = np.interp(pt_t, tr.t, tr.x), np.interp(pt_t, tr.t, tr.y)
        pt_v = np.array([tr.speed(x, 0.5) for x in pt_t])
        for i in range(n):
            t = self.t0 + i / TRAIL_RATE
            k = int(np.searchsorted(pt_t, t, side="right"))
            ms = [ros.delete_all(t, self.fixed)]
            if k >= 2:
                x, y, _ = tr.at(t)
                pieces = np.split(np.arange(k), np.searchsorted(pt_t[:k], tr.t[tr.breaks], side="left"))
                for m, piece in enumerate(pieces):
                    if len(piece) < 2:
                        continue
                    pts = [(pt_x[q], pt_y[q], 0) for q in piece]
                    cols = [speed_color(pt_v[q]) for q in piece]
                    if m == len(pieces) - 1:
                        pts.append((x, y, 0))
                        cols.append(speed_color(tr.speed(t, 0.5)))
                    ms.append(ros.marker(t, self.fixed, "trail", m, LINE_STRIP, ros.pose(0, 0, Z["trail"]),
                                         (0.09, 0, 0), WHITE, points=pts, colors=cols))
            for m, (a, b, x, y) in enumerate(self.stops):
                if a > t:
                    break
                dur = min(b, t) - a
                if dur < STOP_MIN:
                    continue
                ms.append(ros.marker(t, self.fixed, "stops", 2 * m, CYLINDER, ros.pose(x, y, Z["stops"]),
                                     (0.5, 0.5, 0.01), (1.0, 1.0, 1.0, 0.45)))
                ms.append(ros.marker(t, self.fixed, "stops", 2 * m + 1, TEXT, ros.pose(x, y - 0.7, Z["label"]),
                                     (0, 0, 0.3), (1.0, 1.0, 1.0, 0.8), text="停 %.1f s" % dur))
            self.out.ros_msg("/viz/trail", "visualization_msgs/msg/MarkerArray", t,
                             ros.new("visualization_msgs/msg/MarkerArray", markers=ms))

    # -- map, crosswalks
    def bounds(self):
        xs, ys = list(self.track.x), list(self.track.y)
        for g in self.goals:
            xs.append(g[1])
            ys.append(g[2])
        for p in self.plan_info:
            if p.get("global_path"):
                xs.extend(q[0] for q in p["global_path"])
                ys.extend(q[1] for q in p["global_path"])
        m = MAP_MARGIN
        return min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m

    def write_static(self):
        ros, a = self.ros, self.args
        self.map_meta, self.route, self.crosswalks = None, None, []
        if a.map:
            try:
                self.map_meta = load_map(a.map)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
                self.notes.append("map %s not read: %s" % (a.map, e))
        osm = None
        if a.osm:
            src = Source(a.osm)
            names = [a.osm] if os.path.isfile(a.osm) and not src.zip else src.find(r"semantic\.osm$")
            osm = src.read(names[0]) if names else None
        elif self.map_meta is not None:
            osm = self.map_meta["osm"]
        if osm:
            self.crosswalks = read_crosswalks(osm)
        xmin, ymin, xmax, ymax = self.bounds()
        if self.map_meta is not None:
            self.route = crop_route(self.map_meta, xmin, ymin, xmax, ymax)
            if self.route is None:
                self.notes.append("map %s does not cover the run" % self.map_meta["name"])
            else:
                data, ox, oy, w, h = self.route
                self.notes.append("route map: %s, %d x %d cells around the run, %d route cells"
                                  % (self.map_meta["name"], w, h, int(np.count_nonzero(data))))
        near = [(cid, poly) for cid, poly in self.crosswalks
                if poly[:, 0].max() > xmin and poly[:, 0].min() < xmax and poly[:, 1].max() > ymin
                and poly[:, 1].min() < ymax]
        if self.crosswalks:
            self.notes.append("crosswalks: %d in the map, %d near the run" % (len(self.crosswalks), len(near)))
        times = np.arange(self.t0, self.t1 + 1e-6, STATIC_PERIOD)
        for t in times:
            if self.route is not None:
                data, ox, oy, w, h = self.route
                res = self.map_meta["resolution"]
                g = ros.new("nav_msgs/msg/OccupancyGrid", header=ros.header(t, self.fixed),
                            info=ros.new("nav_msgs/msg/MapMetaData", resolution=float(res), width=int(w),
                                         height=int(h), origin=ros.pose(ox, oy, Z["route"])),
                            data=data.ravel())
                self.out.ros_msg("/viz/route_map", "nav_msgs/msg/OccupancyGrid", float(t), g)
            if near:
                self.out.ros_msg("/viz/crosswalks", "visualization_msgs/msg/MarkerArray", float(t),
                                 ros.new("visualization_msgs/msg/MarkerArray",
                                         markers=self.crosswalk_markers(float(t), near)))

    def write_legend(self):
        for t in np.arange(self.t0, self.t1 + 1e-6, STATIC_PERIOD):
            self.out.json_msg("/viz/legend", "planning_viz.Legend", FLOW_SCHEMA, float(t), LEGEND)

    def crosswalk_markers(self, t, near):
        ros = self.ros
        ms = [ros.delete_all(t, self.fixed)]
        for k, (cid, poly) in enumerate(near):
            tris = triangulate(poly)
            pts = [(poly[i][0], poly[i][1], 0) for tri in tris for i in tri]
            ms.append(ros.marker(t, self.fixed, "crosswalk_fill", k, TRIANGLES, ros.pose(0, 0, Z["crosswalk"]),
                                 (1, 1, 1), (1.0, 0.84, 0.04, 0.28), points=pts))
            ring = [(p[0], p[1], 0) for p in poly] + [(poly[0][0], poly[0][1], 0)]
            ms.append(ros.marker(t, self.fixed, "crosswalk_edge", k, LINE_STRIP,
                                 ros.pose(0, 0, Z["crosswalk"] + 0.002), (0.08, 0, 0), (1.0, 0.84, 0.04, 0.9),
                                 points=ring))
            c = poly.mean(axis=0)
            ms.append(ros.marker(t, self.fixed, "crosswalk_label", k, TEXT, ros.pose(c[0], c[1], Z["label"]),
                                 (0, 0, 0.7), (1.0, 0.9, 0.4, 0.9),
                                 text="人行横道 %s\n跨 %.0f m" % (cid, crosswalk_length(poly))))
        return ms

    # -- point cloud
    def convert_cloud(self):
        rows = self.rec["clouds"]
        if not rows:
            return
        ros = self.ros
        typ = self.rec["types"][self.topics["cloud"]]
        written, skipped = 0, 0
        for log, data in rows:
            m = ros.decode(data, typ)
            t = self.data_time(stamp_of(m.header), log)
            xyz, inten = cloud_xyz(m)
            if xyz is None:
                skipped += 1
                continue
            frame = m.header.frame_id.lstrip("/")
            to_base = self.static_to_base(frame)
            if to_base is None:
                skipped += 1
                continue
            R0, p0 = to_base
            xyz = xyz @ R0.T + p0                       # in base_link
            R, p = self.track.full(t)
            world = xyz @ R.T + p                       # in map, full 3D pose
            world[:, 2] -= p[2]                         # heights relative to the robot's base
            keep = np.hypot(world[:, 0] - p[0], world[:, 1] - p[1]) < CLOUD_RANGE
            world, inten = world[keep], (inten[keep] if inten is not None else None)
            if len(world) == 0:
                continue
            vox = np.floor(world / CLOUD_VOXEL).astype(np.int64)
            _, first = np.unique(vox, axis=0, return_index=True)
            world = world[first]
            inten = inten[first] if inten is not None else world[:, 2]
            self.out.ros_msg("/viz/cloud", "sensor_msgs/msg/PointCloud2", t, make_cloud(ros, t, self.fixed, world,
                                                                                       inten))
            written += 1
        self.notes.append("%s -> /viz/cloud: %d clouds (%d skipped)" % (self.topics["cloud"], written, skipped))

    def static_to_base(self, frame):
        if frame == self.base:
            return np.eye(3), np.zeros(3)
        rows = self.rec["tf"].get((self.base, frame))
        if rows:
            r = rows[-1]
            return quat_matrix(r[5], r[6], r[7], r[8]), np.array(r[2:5])
        rows = self.rec["tf"].get((frame, self.base))
        if rows:
            r = rows[-1]
            R = quat_matrix(r[5], r[6], r[7], r[8])
            return R.T, -R.T @ np.array(r[2:5])
        return None

    # -- events
    def collect_events(self):
        st = self.states
        prev = None
        for s in st:
            cur = (s["mission_type"], s["mission_status"])
            if cur != prev and s["mission_type"]:
                before = "%s %s" % (MISSION_ZH.get(prev[0], prev[0]), prev[1]) if prev and prev[0] else "开始"
                level = "error" if s["mission_status"] == "fault" else "info"
                self.event(s["t"], level, "任务 %s → %s %s" % (before, MISSION_ZH.get(cur[0], cur[0]), cur[1]),
                           "mission")
            prev = cur
        self._edges(st, "input_timeout", "warn", "输入超时（mission_input_timeout）", "输入恢复")
        self._edges(st, "timeout_paused", "warn", "输入超时，任务暂停", "任务从超时暂停里恢复")
        # costmap under the dog
        on, limit = False, self.footprint_limit()
        for s in st:
            c = s["cost_map_footprint_cells"]
            if c is None:
                continue
            if not on and c >= limit:
                self.event(s["t"], "warn", "代价地图里狗脚下是障碍（%d 格）：狗在规划地图的路外面，局部规划会认为起点在碰撞里"
                           % c, "costmap")
                on = True
            elif on and c == 0:
                self.event(s["t"], "info", "代价地图里脚下又是空的了", "costmap")
                on = False
        # commands: a change that holds for 0.5 s
        moving, since = None, None
        for s in st:
            m = s["moving"]
            if m is None:
                continue
            if moving is None:
                moving = m
            elif m == moving:
                since = None
            elif since is None:
                since = s["t"]
            elif s["t"] - since >= 0.5:
                self.event(since, "info", "开始发速度指令" if m else "速度指令归零", "cmd")
                moving, since = m, None
        # goals
        last = None
        for g in self.goals:
            if last is None or math.hypot(g[1] - last[1], g[2] - last[2]) > 0.2:
                self.event(g[0], "info", "终点 (%.1f, %.1f)" % (g[1], g[2]), "goal")
            last = g
        reached = False
        for s in st:
            if s["goal_distance"] is not None and s["goal_distance"] < 0.5 and not reached:
                self.event(s["t"], "info", "到终点（%.2f m）" % s["goal_distance"], "goal")
                reached = True
            elif s["goal_distance"] is not None and s["goal_distance"] > 1.0:
                reached = False
        for a, b, x, y in self.stops:
            self.event(a, "info", "停了 %.1f s（%.1f, %.1f）" % (b - a, x, y), "motion")
        # pose: gaps and jumps
        tr = self.track
        dt = np.diff(tr.t)
        for i in np.nonzero(dt > GAP_POSE)[0]:
            self.event(tr.t[i], "warn", "位姿 %.2f s 没更新" % dt[i], "pose")
        raw = self.raw_pose
        d = np.hypot(np.diff(raw[:, 1]), np.diff(raw[:, 2]))
        for i in np.nonzero(d > JUMP)[0]:
            early = raw[i + 1, 0] - raw[0, 0] < STARTUP
            self.event(raw[i + 1, 0], "warn" if early else "error", "位姿跳了 %.1f m（%.2f s 内）%s" % (
                d[i], raw[i + 1, 0] - raw[i, 0], "：定位初始化，之前的位姿不用" if early else ""), "pose")
        # diagnostics
        seen = {}
        for log, statuses in self.rec["diagnostics"]:
            t = self.clock(log)
            for name, level, message in statuses:
                key = (level, message)
                if seen.get(name) != key:
                    if level >= 1 or name in seen:
                        lv = "info" if level == 0 else "warn" if level == 1 else "error"
                        self.event(t, lv, "%s: %s" % (name, message or ("OK" if level == 0 else "")), "diagnostic")
                    seen[name] = key
        self.read_glog()

    def footprint_limit(self):
        length, width = self.args.footprint
        res = 0.05
        rows = self.rec["grids"].get(self.topics["cost_map"], [])
        if rows:
            res = self.ros.decode(rows[0][1], self.rec["types"][self.topics["cost_map"]]).info.resolution or res
        return max(1, int(FOOTPRINT_SHARE * length * width / (res * res)))

    def _edges(self, st, key, level, on_text, off_text):
        prev = None
        for s in st:
            v = s[key]
            if v is None:
                continue
            v = 1 if v >= 0.5 else 0
            if prev is not None and v != prev:
                self.event(s["t"], level if v else "info", on_text if v else off_text, "mission")
            prev = v

    def read_glog(self):
        a = self.args
        path = a.glog
        if path is None:
            cand = os.path.join(a.recording if os.path.isdir(a.recording) else os.path.dirname(a.recording),
                                "logs", "planning.INFO")
            path = cand if os.path.isfile(cand) else None
        if not path:
            return
        wall0 = self.clock.log[0] if self.clock.log is not None else None
        import time as _time
        lines = parse_glog(path, default_year=None if wall0 is None else _time.gmtime(wall0).tm_year)
        if not lines:
            return
        tz = 0.0
        if wall0 is not None:
            tz = round((lines[0][0] - wall0) / 900.0) * 900.0
        keep, last_text = 0, {}
        mission_state = None
        for naive, level, where, text in lines:
            t = self.clock(naive - tz)
            if "mission_state:" in text:
                state = text.split("mission_state:", 1)[1].strip().split()[0] if text.split("mission_state:", 1)[
                    1].strip() else ""
                if state == mission_state:
                    continue
                mission_state = state
            elif level == "info" and not any(k in text for k in GLOG_KEYWORDS):
                continue
            norm = re.sub(r"[-+]?\d+(\.\d+)?", "#", text)
            if norm in last_text and t - last_text[norm] < 5.0:
                continue
            last_text[norm] = t
            self.events.append((float(t), LOG[level], "planning", "%s] %s" % (where, text)))
            keep += 1
        self.notes.append("glog %s: %d lines, %d events (time zone %+.1f h)" % (path, len(lines), keep, tz / 3600))

    def write_events(self):
        ros = self.ros
        self.events.sort(key=lambda e: e[0])
        for t, level, name, text in self.events:
            self.out.ros_msg("/viz/events", "rcl_interfaces/msg/Log", t,
                             ros.new("rcl_interfaces/msg/Log", stamp=ros.time(t), level=level, name=name, msg=text))


GLOG_KEYWORDS = ("Start adjusted", "Goal adjusted", "in collision", "No free point", "Map switch", "crosswalks in",
                 "OSM", "SetPubFlag", "pausing", "resum", "Received goal", "goal", "A*", "replan", "fail", "Fail",
                 "Crosswalk yield", "crosswalk yield")
# I1113 11:35:32.023888 ... (old glog) or I20261113 11:35:32.023888 ... (glog 0.7, the server's since T48: the year
# in front, which the old pattern did not read, so T48's logs gave no events)
GLOG_LINE = re.compile(r"^([IWEF])(\d{4})?(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.(\d+)\s+\S+\s+([^\]]+)\]\s?(.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")    # FLAGS_colorlogtostderr (lightning) colours warnings and errors


def parse_glog(path, default_year=None):
    """[(naive epoch as if UTC, level, file:line, text)] of a glog file (stderr output has no "Log file created
    at" line: default_year then)."""
    year = None
    out = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if year is None:
                m = re.match(r"Log file created at: (\d{4})/", line)
                if m:
                    year = int(m.group(1))
                    continue
            m = GLOG_LINE.match(ANSI.sub("", line.rstrip("\n")))
            if not m:
                continue
            lv, yy, mo, d, hh, mi, ss, frac, where, text = m.groups()
            y = int(yy) if yy else (year or default_year or 1970)
            naive = calendar.timegm((y, int(mo), int(d), int(hh), int(mi), int(ss), 0, 0, 0)) + float("0." + frac)
            out.append((naive, {"I": "info", "W": "warn", "E": "error", "F": "fatal"}[lv], where.strip(), text))
    return out


def polyline_distance(poly, x, y):
    p = np.asarray(poly, float)
    a, b = p[:-1], p[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    u = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / L2, 0, 1)
    q = a + ab * u[:, None]
    return float(np.hypot(q[:, 0] - x, q[:, 1] - y).min())


def polyline_length(poly):
    p = np.asarray(poly, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def speed_color(v, vmax=2.0):
    """Blue (still) -> green -> yellow -> red (vmax and faster)."""
    stops = [(0.0, (0.25, 0.45, 1.0)), (0.33, (0.2, 0.85, 0.45)), (0.66, (1.0, 0.85, 0.1)), (1.0, (1.0, 0.25, 0.2))]
    f = min(max(v / vmax, 0.0), 1.0)
    for (f0, c0), (f1, c1) in zip(stops[:-1], stops[1:]):
        if f <= f1:
            w = (f - f0) / (f1 - f0)
            return tuple(c0[i] + w * (c1[i] - c0[i]) for i in range(3)) + (0.95,)
    return stops[-1][1] + (0.95,)


def cloud_xyz(m):
    """(N, 3) float64 and intensity (N) or None from a PointCloud2 with float32 x, y, z."""
    fields = {f.name: (f.offset, f.datatype) for f in m.fields}
    if not all(k in fields and fields[k][1] == 7 for k in "xyz"):
        return None, None
    n = m.width * m.height
    raw = np.asarray(m.data, dtype=np.uint8)
    if n == 0 or raw.size < n * m.point_step:
        return None, None
    rows = raw[:n * m.point_step].reshape(n, m.point_step)
    endian = ">" if m.is_bigendian else "<"

    def col(name, dtype):
        off = fields[name][0]
        size = np.dtype(dtype).itemsize
        return rows[:, off:off + size].copy().view(endian + dtype).ravel().astype(np.float64)

    xyz = np.stack([col("x", "f4"), col("y", "f4"), col("z", "f4")], axis=1)
    inten = None
    if "intensity" in fields:
        dt = {7: "f4", 8: "f8", 2: "u1", 4: "u2"}.get(fields["intensity"][1])
        if dt:
            inten = col("intensity", dt)
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], (inten[ok] if inten is not None else None)


def make_cloud(ros, t, frame, xyz, inten):
    pts = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    pts["x"], pts["y"], pts["z"], pts["intensity"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], inten
    fields = [ros.new("sensor_msgs/msg/PointField", name=n, offset=4 * i, datatype=7, count=1)
              for i, n in enumerate(("x", "y", "z", "intensity"))]
    return ros.new("sensor_msgs/msg/PointCloud2", header=ros.header(t, frame), height=1, width=len(pts),
                   fields=fields, is_bigendian=False, point_step=16, row_step=16 * len(pts),
                   data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True)


# ---------------------------------------------------------------- overview image

def write_overview(conv, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.collections import LineCollection
    except ImportError:
        conv.notes.append("overview: no matplotlib, skipped")
        return False
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    zh = False
    for name in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei",
                 "Microsoft YaHei", "Source Han Sans SC", "PingFang SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            zh = True
            break

    def L(cn, en):
        return cn if zh else en

    tr, st = conv.track, conv.states
    fig = plt.figure(figsize=(18, 10), dpi=110)
    gs = fig.add_gridspec(5, 2, width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#14171c")
    xmin, ymin, xmax, ymax = conv.bounds()
    if conv.route is not None:
        data, ox, oy, w, h = conv.route
        res = conv.map_meta["resolution"]
        img = np.where(data.reshape(h, w) > 0, 1.0, np.nan)
        ax.imshow(img, origin="lower", extent=(ox, ox + w * res, oy, oy + h * res), cmap="Greys_r", vmin=0, vmax=3,
                  interpolation="nearest", zorder=1)
    for k, (cid, poly) in enumerate(conv.crosswalks):
        if poly[:, 0].max() < xmin or poly[:, 0].min() > xmax or poly[:, 1].max() < ymin or poly[:, 1].min() > ymax:
            continue
        ring = np.vstack([poly, poly[:1]])
        ax.fill(ring[:, 0], ring[:, 1], color="#ffd60a", alpha=0.35, zorder=2)
        ax.plot(ring[:, 0], ring[:, 1], color="#ffd60a", lw=1.2, zorder=2)
        c = poly.mean(axis=0)
        ax.text(c[0], c[1], L("人行横道 %s", "crosswalk %s") % cid, color="#ffe680", fontsize=7, ha="center",
                zorder=6, clip_on=True)
    paths = [p.get("global_path") for p in conv.plan_info if p.get("global_path") and len(p["global_path"]) >= 2]
    if paths:
        first, last = np.asarray(paths[0]), np.asarray(paths[-1])
        ax.plot(first[:, 0], first[:, 1], color="#0a84ff", lw=2.5, alpha=0.55, zorder=3,
                label=L("全局路径（第一次）", "global path (first)"))
        ax.plot(last[:, 0], last[:, 1], color="#64d2ff", lw=1.2, ls="--", alpha=0.9, zorder=3,
                label=L("全局路径（最后一次）", "global path (last)"))
    pts = np.stack([tr.x, tr.y], axis=1)
    step = max(1, len(pts) // 4000)
    pts = pts[::step]
    tt = tr.t[::step]
    v = np.array([tr.speed(x, 0.5) for x in tt])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    short = np.hypot(*(segs[:, 1] - segs[:, 0]).T) < JUMP * step
    segs, v = segs[short], np.append(v[:-1][short], 0.0)
    lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(0, 2.0), linewidths=3, zorder=4)
    lc.set_array(v[:-1])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.035, pad=0.05, shrink=0.7)
    cb.set_label(L("走过的轨迹，按速度着色 m/s", "path walked, coloured by speed m/s"), fontsize=8)
    ax.plot(pts[0, 0], pts[0, 1], "o", color="white", ms=9, zorder=7, label=L("起点", "start"))
    for g in conv.goals[:1] + conv.goals[-1:]:
        ax.plot(g[1], g[2], "*", color="#ff453a", ms=18, zorder=7)
    if conv.goals:
        ax.plot([], [], "*", color="#ff453a", ms=12, label=L("终点", "goal"))
    limit = conv.footprint_limit()
    occ = [s for s in st if (s["cost_map_footprint_cells"] or 0) >= limit]
    if occ:
        ax.plot([s["x"] for s in occ], [s["y"] for s in occ], ".", color="#ff453a", ms=4, zorder=5,
                label=L("代价地图里脚下是障碍（规划地图的路外）", "costmap: obstacle under the dog (off the map's route)"))
    for a, b, x, y in conv.stops:
        ax.plot(x, y, "o", mfc="none", mec="white", ms=10, zorder=6)
        ax.text(x + 1.0, y, L("停 %.0f s", "stop %.0f s") % (b - a), color="white", fontsize=7, zorder=6,
                clip_on=True)
    ax.set_xlim(min(tr.x.min(), *(g[1] for g in conv.goals)) - 15 if conv.goals else tr.x.min() - 15,
                max(tr.x.max(), *(g[1] for g in conv.goals)) + 15 if conv.goals else tr.x.max() + 15)
    ax.set_ylim(min(tr.y.min(), *(g[2] for g in conv.goals)) - 15 if conv.goals else tr.y.min() - 15,
                max(tr.y.max(), *(g[2] for g in conv.goals)) + 15 if conv.goals else tr.y.max() + 15)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, facecolor="#22262d", labelcolor="white", framealpha=0.9)
    ax.set_title("%s — %.0f s, %.0f m" % (os.path.basename(conv.args.recording.rstrip("/")), conv.t1 - conv.t0,
                                         tr.s[-1]))
    ax.tick_params(labelsize=8)

    t = np.array([s["t"] - conv.t0 for s in st]) if st else np.zeros(0)

    def series(key):
        return np.array([np.nan if s[key] is None else s[key] for s in st], float)

    panels = [
        (L("速度 m/s", "speed m/s"), [("odom_v", L("实际", "odom"), "#64d2ff"), ("cmd_v", L("指令", "command"), "#ff9f0a")]),
        (L("转速 rad/s", "yaw rate rad/s"), [("odom_omega", L("实际", "odom"), "#64d2ff"),
                                            ("cmd_omega", L("指令", "command"), "#ff9f0a")]),
        (L("到终点 / 偏离路径 m", "to goal / off path m"), [("goal_distance", L("到终点", "to goal"), "#ff453a"),
                                                       ("path_offset", L("偏离全局路径", "off global path"), "#0a84ff")]),
        (L("脚下障碍格数", "cells under the dog"), [("cost_map_footprint_cells", L("代价地图", "costmap"), "#ff453a"),
                                                ("perception_footprint_cells", L("感知栅格", "perception"), "#ff9f0a")]),
    ]
    axes = []
    for row, (title, keys) in enumerate(panels):
        a2 = fig.add_subplot(gs[row, 1], sharex=axes[0] if axes else None)
        axes.append(a2)
        for key, lab, col in keys:
            y = series(key)
            if np.isfinite(y).any():
                a2.plot(t, y, color=col, lw=1.0, label=lab)
        a2.set_ylabel(title, fontsize=8)
        a2.legend(loc="upper right", fontsize=7)
        a2.grid(alpha=0.3)
        a2.tick_params(labelsize=7)
    a3 = fig.add_subplot(gs[4, 1], sharex=axes[0] if axes else None)
    names = ["off", "passive", "standby", "active", "standstill", "fault"]
    y = np.array([names.index(s["mission_status"]) if s["mission_status"] in names else np.nan for s in st], float)
    a3.step(t, y, where="post", color="#30d158")
    a3.set_yticks(range(len(names)))
    a3.set_yticklabels(names, fontsize=7)
    a3.set_ylabel(L("任务状态", "mission"), fontsize=8)
    import time as _time
    start = _time.strftime("%Y-%m-%d %H:%M:%S %Z", _time.localtime(conv.t0))
    a3.set_xlabel(L("时间 s（bag 时间，从 %s 起）", "time s (bag time from %s)") % start, fontsize=8)
    a3.grid(alpha=0.3)
    for e in conv.events:
        if e[1] >= LOG["warn"]:
            for a2 in axes + [a3]:
                a2.axvline(e[0] - conv.t0, color="#ff453a" if e[1] >= LOG["error"] else "#ffd60a", lw=0.6,
                           alpha=0.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="planning recording: a ros2 bag directory with .mcap files, or one .mcap")
    p.add_argument("extra", nargs="*", help="more recordings of the same run read with it (the replay script records "
                                            "the heavy point clouds into a separate bag that is not kept)")
    p.add_argument("-o", "--output", help="default <recording>/viz/planning_viz.mcap")
    p.add_argument("--map", default=DEFAULT_MAP if os.path.isfile(DEFAULT_MAP) else None,
                   help="planning map: map.yaml, a map directory with 2d/<id>/map.yaml (and semantic.osm), or its "
                        "zip; default the D20 planning map CJGC; 'none' for no map")
    p.add_argument("--osm", help="semantic.osm (or a map directory / zip with one) for the crosswalks; default the "
                                 "one beside --map")
    p.add_argument("--glog", help="planning's glog; default <recording>/logs/planning.INFO")
    p.add_argument("--loc-map", help="localization map directory or zip (3d/global.pcd) for /viz/loc/map_cloud and the "
                                     "scans' match; default /lightning/map_cloud of the recording")
    p.add_argument("--loc-config", default=DEFAULT_LOC_CONFIG,
                   help="lightning config with system.ins_lat / ins_lon / ins_height (the INS's map origin)")
    p.add_argument("--loc-glog", help="lightning's log; default <recording>/logs/loc.log")
    p.add_argument("--overview", help="overview image; default beside the output (.png); 'none' for no image")
    p.add_argument("--footprint", type=float, nargs=2, default=(1.0, 0.4), metavar=("LENGTH", "WIDTH"))
    p.add_argument("--cost-map-values", choices=("auto", "ros", "inverted"), default="auto",
                   help="planning's costmap: 'ros' 100 = obstacle, 'inverted' free = 100 (ros_debug_bridge before the "
                        "T47 review); auto tells from the perception obstacles")
    p.add_argument("--fixed-frame", default="map")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--odom-topic", default=TOPICS["odom"])
    p.add_argument("--cloud-topic", default=TOPICS["cloud"])
    args = p.parse_args(argv)
    if args.map and args.map.lower() == "none":
        args.map = None
    rec_dir = args.recording if os.path.isdir(args.recording) else os.path.dirname(os.path.abspath(args.recording))
    if args.output is None:
        args.output = os.path.join(rec_dir, "viz", "planning_viz.mcap")
    if args.overview is None:
        args.overview = os.path.splitext(args.output)[0] + ".png"
    elif args.overview.lower() == "none":
        args.overview = None
    conv = Converter(args).run()
    if args.overview:
        if write_overview(conv, args.overview):
            conv.notes.append("overview: %s" % args.overview)
    counts = conv.out.counts()
    print("%s -> %s (%.1f MB), %.1f s of bag time from %.3f" % (
        args.recording, args.output, os.path.getsize(args.output) / 1e6, conv.t1 - conv.t0, conv.t0))
    for line in conv.notes:
        print("  " + line)
    print("  topics: " + ", ".join("%s %d" % (k, v) for k, v in sorted(counts.items())))
    warn = sum(1 for e in conv.events if e[1] >= LOG["warn"])
    print("  events: %d (%d warnings or errors) on /viz/events" % (len(conv.events), warn))
    for e in [e for e in conv.events if e[1] >= LOG["warn"]][:12]:
        print("    %7.2f s  %s" % (e[0] - conv.t0, e[3]))
    print("  open in Foxglove with the layout %s" % os.path.relpath(DEFAULT_LAYOUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
