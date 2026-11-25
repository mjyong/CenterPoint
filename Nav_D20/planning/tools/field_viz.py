"""Field review layers for planning_mcap_viz.py: what perception saw, what the crosswalk gate made of it, the lidar and
the front fisheye, on the same time base as the planning layers. For a bag recorded on the dog (after bag_to_mcap.py)
or a replay of one (the fisheye then comes from the field bag: --camera-from).

  /viz/objects         perception's tracks (/perception/lidar_3d/objects), placed in the map the way the crosswalk gate
                       places them (CrosswalkYield::ObjectsInMap: the det_to_base extrinsic, the pose at the message's
                       stamp): footprint coloured by class (vehicle blue, pedestrian green, cyclist purple; dashed while
                       coasting), velocity arrow (where it is in 1 s), the most likely predicted path, "#id class km/h"
  /viz/objects_table   JSON per perception frame: counts, the nearest vehicle, every object with its distance and
                       bearing from the dog (left +), nearest first
  /viz/crosswalk_gate  planning's own /planning/debug/crosswalk flattened: the crosswalk it is at coloured by the mode
                       (yellow approach, red align / wait / cross_wait, green go / cross), the vehicles where the gate
                       places them (red: it waits for it, orange: a conflict it goes on with, blue: not in the way)
  /viz/crosswalk       JSON per planning cycle: the gate's mode by name and what it means, the crosswalk by the site
                       map's number, go, how many it waits for, stale, unreleased, the stop distance, the heading error
                       and the gate's own reason
  /viz/lidar           the lidar (/robot/topic/rdap/top_point_cloud2) without the ground, flattened onto the map under
                       the other layers, field "height" = above the ground
  <camera topic>       the fisheye's H.264 (foxglove_msgs/msg/CompressedVideo) as recorded, on the bag's time base
  /viz/events          + crosswalk mode changes, perception stale / unreleased, tracks born close to the dog (a
                       possible miss before) and tracks that lived less than a second (a possible false detection)
"""
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OBJECTS = "/perception/lidar_3d/objects"
GATE = "/planning/debug/crosswalk"
SHADOW_CMD = "/nav_cw/shadow/cmd_vel"
LIDAR = "/robot/topic/rdap/top_point_cloud2"
CAMERAS = {k: "/robot/topic/rdap/%s_sensor_camera" % k for k in ("front", "left", "right", "rear")}
DET_TO_BASE = (0.319, 0.043, -90.0)     # crosswalk_yield_conf det_to_base_* (objects_in_map.py)

MODES = {0: "none", 1: "approach", 2: "align", 3: "wait", 4: "go", 5: "cross", 6: "cross_wait"}
MODE_MEANING = {
    "none": "前面没有人行横道",
    "approach": "走向停止点，减速",
    "align": "在停止点原地转到面朝对面",
    "wait": "在路边站着等车",
    "go": "放行，开始过",
    "cross": "在人行横道上走",
    "cross_wait": "在人行横道上停下等车（前方可能来车）",
}
CLASS_ZH = {"vehicle": "车", "pedestrian": "人", "cyclist": "骑车"}
CLASS_COLOR = {"vehicle": (0.25, 0.62, 1.0, 1.0), "pedestrian": (0.3, 0.9, 0.4, 1.0),
               "cyclist": (0.8, 0.45, 1.0, 1.0)}
MOTION_ZH = {0: "静止", 1: "在动", 2: "停下"}

LIDAR_RATE = 5.0       # [Hz]
LIDAR_RANGE = 40.0     # [m]
LIDAR_VOXEL = 0.1      # [m]
GROUND_ABOVE = 0.25    # [m] points this far above the fitted ground are kept
GROUND_TOP = 3.0       # [m] and up to this (no tree canopy over the road)
BORN_NEAR = 12.0       # [m] a vehicle track starting this close: perception may have missed it before
SHORT_LIFE = 1.0       # [s] a track that lived less than this: may be a false detection
REASON_GAP = 0.5       # [s] the gate's state line is this fresh at most


def objects_to_map(pose, objs, det_to_base):
    """CrosswalkYield::ObjectsInMap for rows of (x, y, yaw, vx, vy, [(px, py), ...]) in the detection frame."""
    x, y, yaw = pose
    dx, dy, ydb = det_to_base
    a = yaw + math.radians(ydb)
    c, s = math.cos(a), math.sin(a)
    tx = x + math.cos(yaw) * dx - math.sin(yaw) * dy
    ty = y + math.sin(yaw) * dx + math.cos(yaw) * dy
    out = []
    for ox, oy, oyaw, vx, vy, path in objs:
        out.append((tx + c * ox - s * oy, ty + s * ox + c * oy, oyaw + a, c * vx - s * vy, s * vx + c * vy,
                    [(tx + c * px - s * py, ty + s * px + c * py) for px, py in path]))
    return out


def drop_ground(xyz, cell=2.0, max_range=30.0):
    """Mask of the points between GROUND_ABOVE and GROUND_TOP over a plane fitted to the lowest point of each cell
    (the lidar tilts with the dog: a plane, not a height)."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    near = (r > 2.0) & (r < max_range)
    if near.sum() < 50:
        return np.ones(len(xyz), bool), np.zeros(len(xyz))
    p = xyz[near]
    key = np.floor(p[:, :2] / cell).astype(np.int64)
    order = np.lexsort((p[:, 2], key[:, 1], key[:, 0]))
    k = key[order]
    first = np.ones(len(k), bool)
    first[1:] = (k[1:] != k[:-1]).any(axis=1)
    low = p[order][first]
    A = np.c_[low[:, 0], low[:, 1], np.ones(len(low))]
    w = np.ones(len(low), bool)
    coef = np.array([0.0, 0.0, float(np.percentile(low[:, 2], 20))])
    for _ in range(3):
        if w.sum() < 6:
            break
        coef = np.linalg.lstsq(A[w], low[w, 2], rcond=None)[0]
        res = low[:, 2] - A @ coef
        w = (res < 0.3) & (res > -1.0)
    height = xyz[:, 2] - (coef[0] * xyz[:, 0] + coef[1] * xyz[:, 1] + coef[2])
    return (height > GROUND_ABOVE) & (height < GROUND_TOP), height


def reduce_lidar(m):
    """A lidar PointCloud2 -> (N, 2) float32 x, y and (N) float32 height above the ground, in the sensor frame: within
    LIDAR_RANGE, without the ground, the highest point of each LIDAR_VOXEL cell (it is drawn flat: a height map)."""
    from planning_mcap_viz import cloud_xyz
    xyz, _ = cloud_xyz(m)
    if xyz is None or len(xyz) == 0:
        return None
    xyz = xyz[np.hypot(xyz[:, 0], xyz[:, 1]) < LIDAR_RANGE]
    keep, height = drop_ground(xyz)
    xyz, height = xyz[keep], height[keep]
    if len(xyz) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    order = np.argsort(-height, kind="stable")
    xyz, height = xyz[order], height[order]
    _, first = np.unique(np.floor(xyz[:, :2] / LIDAR_VOXEL).astype(np.int64), axis=0, return_index=True)
    return xyz[first, :2].astype(np.float32), height[first].astype(np.float32)


def lidar_cloud(ros, t, frame, xy, z, height):
    pts = np.zeros(len(xy), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("height", "<f4")])
    pts["x"], pts["y"], pts["z"], pts["height"] = xy[:, 0], xy[:, 1], z, height
    fields = [ros.new("sensor_msgs/msg/PointField", name=n, offset=4 * i, datatype=7, count=1)
              for i, n in enumerate(("x", "y", "z", "height"))]
    return ros.new("sensor_msgs/msg/PointCloud2", header=ros.header(t, frame), height=1, width=len(pts),
                   fields=fields, is_bigendian=False, point_step=16, row_step=16 * len(pts),
                   data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True)


def parse_gate_text(text):
    """The gate's state line (RosDebugBridge::CrosswalkDebug): "wait (vehicle 12 may pass), waiting for 1, perception
    stale" -> the reason in the brackets ("" when there is none)."""
    a, b = text.find("("), text.find(")")
    return text[a + 1:b] if 0 <= a < b else ""


class FieldLayers:
    def __init__(self, conv):
        self.c = conv
        self.ros = conv.ros
        self.rec = conv.rec
        self.notes = conv.notes
        self.names = [cid for cid, _ in getattr(conv, "crosswalks", [])]   # planner index -> the site map's number
        self.lidar_from = None
        self.gate_text, self.gate_times = [], []

    def run(self):
        self.write_objects()
        self.write_gate()
        self.write_crosswalk()
        self.write_cameras()
        self.write_lidar()
        return self

    def crosswalk_name(self, k):
        k = int(k)
        return str(self.names[k]) if 0 <= k < len(self.names) else ("#%d" % k if k >= 0 else "")

    # -- perception
    def write_objects(self):
        c, ros = self.c, self.ros
        rows = self.rec.get("objects", [])
        if not rows:
            return
        det_to_base = tuple(c.args.det_to_base)
        tracks = {}
        frames = 0
        outside = 0
        last_t = None
        for log, typ, data in rows:
            m = ros.decode(data, typ)
            t = c.data_time(stamp_of(m.header), log)
            if t < c.t0 - 0.5 or t > c.t1 + 0.5:
                outside += 1
                continue
            pose = c.track.at(t)
            objs = list(m.objects)
            det = []
            for o in objs:
                path = []
                if len(o.predictions):
                    best = max(o.predictions, key=lambda p: p.probability)
                    path = [(q.x, q.y) for q in best.positions]
                det.append((o.position.x, o.position.y, o.yaw, o.velocity.x, o.velocity.y, path))
            placed = objects_to_map(pose, det, det_to_base)
            ms = [ros.delete_all(t, c.fixed)]
            table = []
            counts = defaultdict(int)
            moving_vehicles = 0
            nearest_vehicle = None
            for k, (o, (mx, my, myaw, mvx, mvy, path)) in enumerate(zip(objs, placed)):
                label = o.label_name or {0: "vehicle", 1: "pedestrian", 2: "cyclist"}.get(int(o.label), "?")
                counts[label] += 1
                dist = math.hypot(mx - pose[0], my - pose[1])
                bearing = math.degrees(math.atan2(my - pose[1], mx - pose[0]) - pose[2])
                bearing = (bearing + 180.0) % 360.0 - 180.0
                speed = math.hypot(mvx, mvy)
                if label == "vehicle":
                    if int(o.motion) == 1:
                        moving_vehicles += 1
                    nearest_vehicle = dist if nearest_vehicle is None else min(nearest_vehicle, dist)
                col = CLASS_COLOR.get(label, (0.7, 0.7, 0.7, 1.0))
                if o.coasting:
                    col = col[:3] + (0.45,)
                L, W = max(float(o.size.x), 0.3), max(float(o.size.y), 0.3)
                cy, sy = math.cos(myaw), math.sin(myaw)
                corners = [(mx + cy * u - sy * v, my + sy * u + cy * v) for u, v in
                           ((L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2))]
                if o.coasting:   # dashed: every other edge segment
                    pts = []
                    for i in range(4):
                        a, b = corners[i], corners[(i + 1) % 4]
                        for f0 in np.arange(0.0, 1.0, 0.25):
                            pts += [(a[0] + (b[0] - a[0]) * f0, a[1] + (b[1] - a[1]) * f0, 0),
                                    (a[0] + (b[0] - a[0]) * (f0 + 0.125), a[1] + (b[1] - a[1]) * (f0 + 0.125), 0)]
                    ms.append(ros.marker(t, c.fixed, "box", k, LINE_LIST, ros.pose(0, 0, Z_OBJ), (0.08, 0, 0), col,
                                         points=pts))
                else:
                    ring = [(x, y, 0) for x, y in corners] + [(corners[0][0], corners[0][1], 0)]
                    ms.append(ros.marker(t, c.fixed, "box", k, LINE_STRIP, ros.pose(0, 0, Z_OBJ), (0.1, 0, 0), col,
                                         points=ring))
                front = ((corners[0][0] + corners[3][0]) / 2, (corners[0][1] + corners[3][1]) / 2)
                ms.append(ros.marker(t, c.fixed, "heading", k, LINE_LIST, ros.pose(0, 0, Z_OBJ), (0.06, 0, 0), col,
                                     points=[(mx, my, 0), (front[0], front[1], 0)]))
                if speed > 0.3:
                    ms.append(ros.marker(t, c.fixed, "velocity", k, ARROW, ros.pose(0, 0, Z_OBJ + 0.005),
                                         (0.1, 0.25, 0.3), col[:3] + (1.0,),
                                         points=[(mx, my, 0), (mx + mvx, my + mvy, 0)]))
                if len(path) >= 2:
                    ms.append(ros.marker(t, c.fixed, "prediction", k, LINE_STRIP, ros.pose(0, 0, Z_OBJ - 0.003),
                                         (0.05, 0, 0), col[:3] + (0.5,),
                                         points=[(mx, my, 0)] + [(x, y, 0) for x, y in path]))
                text = "#%d %s" % (int(o.id), CLASS_ZH.get(label, label))
                text += " %.0f km/h" % (speed * 3.6) if speed > 0.3 else " %s" % MOTION_ZH.get(int(o.motion), "")
                if o.coasting:
                    text += " 外推"
                ms.append(ros.marker(t, c.fixed, "label", k, TEXT, ros.pose(mx, my + W / 2 + 0.6, Z_OBJ_LABEL),
                                     (0, 0, 0.55), col[:3] + (1.0,), text=text))
                table.append(dict(id=int(o.id), cls=label, motion=MOTION_ZH.get(int(o.motion), str(int(o.motion))),
                                  score=round(float(o.score), 2), coasting=bool(o.coasting),
                                  age_s=round(float(o.age), 1), distance_m=round(dist, 1),
                                  bearing_deg=round(bearing, 0), speed_kmh=round(speed * 3.6, 1),
                                  length_m=round(L, 1), width_m=round(W, 1), x=round(mx, 2), y=round(my, 2)))
                tr = tracks.setdefault(int(o.id), dict(label=label, t0=t, d0=dist, t1=t, dmin=dist, vmax=0.0,
                                                       n=0, coast=0))
                tr.update(t1=t, dmin=min(tr["dmin"], dist), vmax=max(tr["vmax"], speed), n=tr["n"] + 1,
                          coast=tr["coast"] + (1 if o.coasting else 0))
            table.sort(key=lambda r: r["distance_m"])
            c.out.ros_msg("/viz/objects", "visualization_msgs/msg/MarkerArray", t,
                          ros.new("visualization_msgs/msg/MarkerArray", markers=ms))
            c.out.json_msg("/viz/objects_table", "field_viz.Objects", OBJECTS_SCHEMA, t, dict(
                t=round(t - c.t0, 3), n=len(objs), vehicles=counts["vehicle"], moving_vehicles=moving_vehicles,
                pedestrians=counts["pedestrian"], cyclists=counts["cyclist"],
                nearest_vehicle_m=round(nearest_vehicle, 2) if nearest_vehicle is not None else None,
                gap_s=round(t - last_t, 3) if last_t is not None else None, objects=table))
            last_t = t
            frames += 1
        self.track_events(tracks)
        self.notes.append("%s -> /viz/objects: %d frames%s, det_to_base %.3f %.3f %.1f deg" % (
            OBJECTS, frames, " (%d outside the pose's time)" % outside if outside else "", *det_to_base))

    def track_events(self, tracks):
        c = self.c
        near, short = 0, 0
        by_class = defaultdict(int)
        for tid, tr in sorted(tracks.items(), key=lambda kv: kv[1]["t0"]):
            by_class[tr["label"]] += 1
            name = CLASS_ZH.get(tr["label"], tr["label"])
            if tr["label"] in ("vehicle", "cyclist") and tr["d0"] < BORN_NEAR and tr["t0"] > c.t0 + 2.0:
                near += 1
                c.event(tr["t0"], "info", "感知：#%d %s 在 %.1f m 处才出现（之前可能漏检）" % (tid, name, tr["d0"]),
                        "perception")
            life = tr["t1"] - tr["t0"]
            if life < SHORT_LIFE and tr["t0"] > c.t0 + 1.0 and tr["t1"] < c.t1 - 1.0:
                short += 1
                c.event(tr["t0"], "info", "感知：#%d %s 只出现 %d 帧（%.1f s），最近 %.1f m（可能误检）" % (
                    tid, name, tr["n"], life, tr["dmin"]), "perception")
        self.notes.append("perception tracks: %s; %d vehicles / cyclists born within %.0f m, %d lived < %.0f s" % (
            ", ".join("%s %d" % kv for kv in sorted(by_class.items())) or "none", near, BORN_NEAR, short,
            SHORT_LIFE))

    # -- the crosswalk gate
    def write_gate(self):
        c, ros = self.c, self.ros
        rows = self.rec.get("gate", [])
        self.gate_text = []     # (t, text)
        for log, typ, data in rows:
            m = ros.decode(data, typ)
            stamps = [stamp_of(x.header) for x in m.markers if x.action == 0 and stamp_of(x.header) > 1.0]
            t = max(stamps) if stamps else c.clock(log)
            out = [ros.delete_all(t, c.fixed)]
            for mk in m.markers:
                if mk.action != 0:
                    continue
                if mk.type == TEXT:
                    self.gate_text.append((t, mk.text))
                    continue          # the state line goes on the robot's label and /viz/crosswalk
                width = None
                if mk.ns == "crosswalks":
                    width = 0.3 if mk.scale.x > 0.2 else 0.08
                if c.flatten_marker(mk, t, Z_GATE + (0.004 if mk.ns == "vehicles" else 0.0), width=width):
                    mk.header = ros.header(t, c.fixed)
                    if mk.ns == "vehicle_velocity":
                        mk.scale.x, mk.scale.y, mk.scale.z = 0.12, 0.3, 0.3
                    out.append(mk)
            c.out.ros_msg("/viz/crosswalk_gate", "visualization_msgs/msg/MarkerArray", t,
                          ros.new("visualization_msgs/msg/MarkerArray", markers=out))
        self.gate_text.sort()
        self.gate_times = [g[0] for g in self.gate_text]
        if rows:
            self.notes.append("%s -> /viz/crosswalk_gate: %d arrays" % (GATE, len(rows)))

    def reason_at(self, t):
        from planning_mcap_viz import latest
        times = self.gate_times
        i = latest(times, t + 1e-3)
        if i >= 0 and t - times[i] < REASON_GAP:
            return self.gate_text[i][1]
        return ""

    def crosswalk_state(self, raw, t=None):
        """The gate's state of one planning cycle ("key: value" of the debug string), with names; None when the
        planner does not have the gate's keys."""
        if "crosswalk_mode" not in raw:
            return None
        mode_id = int(raw["crosswalk_mode"])
        mode = MODES.get(mode_id, str(mode_id))
        k = int(raw.get("crosswalk_index", -1))
        line = self.reason_at(t) if t is not None and mode_id else ""
        waiting = int(raw.get("crosswalk_waiting_for", 0))
        go = bool(raw.get("crosswalk_go", 0))
        return dict(mode=mode, mode_id=mode_id, meaning=MODE_MEANING.get(mode, ""),
                    crosswalk=self.crosswalk_name(k) if mode_id else "", crosswalk_index=k,
                    go=1 if go else 0, hold=int(raw.get("crosswalk_hold", 0)),
                    vehicles=int(raw.get("crosswalk_vehicles", 0)), waiting_for=waiting if not go else 0,
                    going_on_with=waiting if go else 0, stale=int(raw.get("crosswalk_stale", 0)),
                    unreleased=int(raw.get("crosswalk_unreleased", 0)),
                    s_stop=round(float(raw.get("crosswalk_s_stop", 0.0)), 2),
                    heading_error_deg=round(math.degrees(float(raw.get("crosswalk_heading_error", 0.0))), 1),
                    reason=parse_gate_text(line), state_line=line)

    def write_crosswalk(self):
        c = self.c
        prev = None
        n = 0
        for s in c.states:
            st = self.crosswalk_state(s["raw"], s["t"])
            if st is None:
                continue
            n += 1
            t = s["t"]
            c.out.json_msg("/viz/crosswalk", "field_viz.Crosswalk", CROSSWALK_SCHEMA, t,
                           dict(st, t=round(t - c.t0, 3), dog_speed=round(s["odom_v"], 2)
                                if s["odom_v"] is not None else None))
            if prev is None or st["mode"] != prev["mode"]:
                if st["mode_id"] or prev is not None:
                    where = "人行横道 %s：" % st["crosswalk"] if st["crosswalk"] else (
                        "人行横道 %s：" % prev["crosswalk"] if prev and prev["crosswalk"] else "人行横道：")
                    text = "%s%s → %s（%s）" % (where, prev["mode"] if prev else "开始", st["mode"], st["meaning"])
                    if st["waiting_for"]:
                        text += "，等 %d 辆" % st["waiting_for"]
                    if st["reason"]:
                        text += "：%s" % st["reason"]
                    c.event(t, "info", text, "crosswalk")
            elif st["waiting_for"] != prev["waiting_for"] and st["mode"] in ("wait", "cross_wait", "align"):
                c.event(t, "info", "人行横道 %s：%s，等的车 %d → %d%s" % (
                    st["crosswalk"], st["mode"], prev["waiting_for"], st["waiting_for"],
                    "：%s" % st["reason"] if st["reason"] else ""), "crosswalk")
            for key, on, off in (("stale", "感知过期（规划收不到新目标），门控按看不见处理", "感知恢复"),
                                 ("unreleased", "在人行横道上但没有放行", "unreleased 解除")):
                was = prev[key] if prev else 0
                if st[key] != was and (st[key] or prev is not None):
                    c.event(t, "warn" if st[key] else "info", "人行横道 %s：%s" % (
                        st["crosswalk"] or "-", on if st[key] else off), "crosswalk")
            prev = st
        if n:
            self.notes.append("crosswalk gate: %d planning cycles -> /viz/crosswalk%s" % (
                n, "; crosswalks named by the site map (planner index 0.. = %s)" % ", ".join(map(str, self.names))
                if self.names else "; no site map: crosswalks by the planner's index"))

    # -- lidar
    def write_lidar(self):
        c, ros = self.c, self.ros
        rows = self.rec.get("lidar", [])
        if not rows:
            return
        static = c.static_to_base(rows[0][2])
        if static is not None:
            R0, p0 = static
            yaw0, src = math.atan2(R0[1, 0], R0[0, 0]), "/tf_static"
            off = (float(p0[0]), float(p0[1]))
        else:   # no TF from the lidar to the base (the dog's tree has none): the detection frame's extrinsic
            dx, dy, ydb = c.args.det_to_base
            yaw0, off, src = math.radians(ydb), (dx, dy), "det_to_base (no TF %s -> %s)" % (rows[0][2], c.base)
        written = 0
        for log, stamp, frame, xy, height in rows:
            t = c.data_time(stamp, log)
            if t < c.t0 - 0.5 or t > c.t1 + 0.5:
                continue
            x, y, yaw = c.track.at(t)
            a = yaw + yaw0
            ca, sa = math.cos(a), math.sin(a)
            bx = x + math.cos(yaw) * off[0] - math.sin(yaw) * off[1]
            by = y + math.sin(yaw) * off[0] + math.cos(yaw) * off[1]
            world = np.c_[bx + ca * xy[:, 0] - sa * xy[:, 1], by + sa * xy[:, 0] + ca * xy[:, 1]]
            c.out.ros_msg("/viz/lidar", "sensor_msgs/msg/PointCloud2", t,
                          lidar_cloud(ros, t, c.fixed, world, Z_LIDAR, height))
            written += 1
        self.notes.append("%s%s -> /viz/lidar: %d clouds at %.0f Hz without the ground, placed by %s" % (
            c.topics.get("lidar", LIDAR), " of %s" % self.lidar_from if self.lidar_from else "", written,
            c.args.lidar_rate, src))

    # -- cameras
    def write_cameras(self):
        c = self.c
        for topic, rows in sorted(self.rec.get("cameras", {}).items()):
            self.copy_camera(topic, rows, c.clock, "the recording")
        for path in c.args.camera_from or []:
            self.camera_from(path)
        self.rec["cameras"] = {}

    def copy_camera(self, topic, rows, clock, src):
        c = self.c
        if not rows:
            return
        typ = rows[0][1]
        c.camera_topics_written = True
        n = 0
        for log, _typ, data in rows:
            t = clock(log)
            if c.t0 - 1.0 <= t <= c.t1 + 1.0:
                c.out.raw_msg(topic, typ, t, data)
                n += 1
        self.notes.append("%s (%s) from %s: %d frames" % (topic, typ, src, n))

    def camera_from(self, path):
        """The fisheye of the field bag a replay was made from, and its lidar when the replay has none (the replay
        keeps no point clouds): its own odometry gives its clock -> the bag's time (the replay's localization stamps
        with the lidar's time, as on the dog)."""
        from planning_mcap_viz import Clock
        from planning_mcap_viz import read_recording
        c = self.c
        topics = {"odom": c.topics["odom"], "lidar": c.topics["lidar"]}
        topics.update(("camera_" + k, v) for k, v in c.camera_topics.items())
        only = ("odom",) if self.rec["lidar"] or c.args.lidar_rate <= 0 else ("odom", "lidar")
        rec = read_recording([path], self.ros, topics, lidar_rate=c.args.lidar_rate, only=only)
        if rec["lidar"]:
            self.rec["lidar"] = rec["lidar"]
            self.lidar_from = os.path.basename(os.path.normpath(path))
        if not rec["odom"]:
            self.notes.append("--camera-from %s: no %s to take its clock from, left out" % (path, c.topics["odom"]))
            return
        clock = Clock([r[1] for r in rec["odom"]], [r[0] for r in rec["odom"]])
        if not rec["cameras"]:
            self.notes.append("--camera-from %s: none of %s" % (path, ", ".join(c.camera_topics.values())))
        for topic, rows in sorted(rec["cameras"].items()):
            self.copy_camera(topic, rows, clock, os.path.basename(os.path.normpath(path)))


def stamp_of(header):
    from planning_mcap_viz import stamp_of as s
    return s(header)


# marker types and layer heights (planning_mcap_viz: Z, the robot at 0.04, labels at 0.5)
ARROW, LINE_STRIP, LINE_LIST, TEXT = 0, 4, 5, 9
Z_LIDAR = -0.07
Z_GATE = 0.032
Z_OBJ = 0.036
Z_OBJ_LABEL = 0.55

OBJECTS_SCHEMA = {"type": "object", "title": "field_viz.Objects", "additionalProperties": True,
                  "properties": {k: {"type": ["number", "null"]} for k in (
                      "t", "n", "vehicles", "moving_vehicles", "pedestrians", "cyclists", "nearest_vehicle_m",
                      "gap_s")}}
CROSSWALK_SCHEMA = {"type": "object", "title": "field_viz.Crosswalk", "additionalProperties": True,
                    "properties": dict({k: {"type": ["number", "null"]} for k in (
                        "t", "mode_id", "crosswalk_index", "go", "hold", "vehicles", "waiting_for", "going_on_with",
                        "stale", "unreleased", "s_stop", "heading_error_deg", "dog_speed")},
                        **{k: {"type": "string"} for k in ("mode", "meaning", "crosswalk", "reason", "state_line")})}

LEGEND = {
    "/viz/objects 框": "感知的目标（/perception/lidar_3d/objects），按规划门控的外参和位姿放进地图：蓝 = 车，绿 = 人，紫 = 骑车；"
                     "虚线框 = 这一帧没检测到、按运动外推；箭头 = 1 s 后在哪；淡线 = 最可能的预测轨迹；字 = #航迹号 类 速度",
    "/viz/crosswalk_gate": "规划自己画的门控（/planning/debug/crosswalk）：当前的人行横道按状态着色（黄 approach，"
                           "红 align / wait / cross_wait，绿 go / cross）；实心方块 = 门控看到的车（红 = 在等它，"
                           "橙 = 有冲突但放行，蓝 = 不挡路）",
    "/viz/lidar": "激光点云去掉地面、压平在最下面，按离地高度着色：有点没框 = 可能漏检，有框没点 = 可能误检",
    "前视鱼眼": "/robot/topic/rdap/front_sensor_camera（H.264），和 3D 同一个时间轴",
    "/viz/crosswalk": "每个规划周期门控的状态：mode（含义）、哪个人行横道（站点地图的编号）、go、等几辆、stale（感知过期）、"
                      "unreleased（在人行横道上没放行）、到停止点的距离、朝向差、门控自己的理由",
    "/viz/objects_table": "这一帧感知的每个目标：离狗多远、在哪个方向（左正右负，0 = 正前）、速度、外推没有，近的在前",
    "/viz/events 里的“感知：”": "离狗 12 m 内才出现的车（之前可能漏检）、只活了不到 1 s 的目标（可能误检）。是提示，要对着鱼眼和点云看",
}
