"""Localization (lightning) layers of planning_mcap_viz.py: what the localization saw, how well each scan sat on the
map, where the INS put the dog, and when lightning complained. For learning lightning's flow (Nav_D20/slam/docs/
ONBOARDING.md) as much as for debugging it.

    /viz/loc/map_cloud   the localization map (3d/global.pcd of --loc-map, or /lightning/map_cloud) around the run,
                         voxel-downsampled; field "height" above the dog's base
    /viz/loc/scan        /lightning/current_scan_world (the de-skewed scan placed with the LIO pose), 5 Hz; field
                         "match": distance to the nearest map point [m] (green: on the map, red: off it)
    /viz/loc/ins         the INS (raw INSPVAX, turned into the map frame the way lightning does it: GeographicLib
                         LocalCartesian at system.ins_lat / ins_lon / ins_height of the lightning config): its track,
                         RTK fixed in magenta, anything else grey; where it is now
    /viz/loc/state       JSON 5 Hz: localization valid (/lightning/odom covariance[1]), INS status, odom minus INS,
                         height, roll, pitch, scan size and how well it matches
    /viz/loc/flow        JSON per scan: lightning's steps with this scan's numbers and where the code is
    events               on /viz/events: valid / invalid, RTK fixed / lost, odom off the INS, scans off the map,
                         lightning's health events and its own warnings (loc.log)

Only what the recording has is drawn; the replay script records the localization topics.
"""
import math
import re

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
SCAN_RATE = 5.0          # [Hz] scans read (their match to the map is computed for each)
SCAN_DRAW_RATE = 2.0     # [Hz] scans drawn
SCAN_VOXEL = 0.15        # [m] of the drawn scans
SCAN_RANGE = 40.0        # [m] around the dog
MAP_VOXEL = 0.3          # [m] of the drawn map cloud
MATCH_VOXEL = 0.1        # [m] of the map points the scans are matched against
MAP_MARGIN = 30.0        # [m] around the run
MAP_PERIOD = 90.0        # [s] the map cloud is sent again
MATCH_GOOD = 0.2         # [m] a scan point this close to the map is "on" it
MATCH_LOW = 0.5          # share of scan points on the map below which a scan counts as off the map
INS_FAR = 1.0            # [m] odom this far from the INS is logged
INS_VALID_STATUS = 3     # kygi665 ins_status: solution good
RTK_FIXED = 56           # kygi665 pos_type: INS_RTKFIXED (lightning: NormalizeKygi665PosType)
LOC_KEYWORDS = ("定位质量", "定位故障", "重初始化", "reinit", "Reinit", "NDT", "雷达断流", "IMU drop", "INS 回调",
                "init pose", "No map chunks", "IVox", "IMU 时间异常", "loop back", "Map", "map")

FLOW_CODE = {
    "imu": "src/core/system/loc_system.cc 订阅 IMU -> Localization::ProcessIMUMsg -> LaserMapping::ProcessIMU（ESKF 预测）"
           " -> 每帧 IMU 发一次 /lightning/odom（100 Hz，T_body_imu 换到 base_link）",
    "scan": "Localization::ProcessLidarMsg -> 预处理（pointcloud_preprocess.cc）-> LidarOdomProcCloud",
    "lio": "src/core/lio/laser_mapping.cc LaserMapping::Run：SyncPackages（一帧点云 + 这段时间的 IMU）-> "
           "ImuProcess::Process（去畸变）-> 体素降采样 -> 离开上次加载处够远就按位姿加载地图块（TiledMap::LoadOnPose）、"
           "异步重建 IVox -> kf_.Update(LIDAR)：点到面 IEKF，和地图的 IVox 配准",
    "health": "LaserMapping::Run 末尾：匹配上的点数 effect_num 太少或平均残差太大算一帧失败，连续 max_fail_count 帧"
              " -> 定位故障、下一帧自动重初始化（有 RTK 固定解就用 INS 位姿，否则退回上次好的位姿 + NDT）",
    "ins": "Localization::LidarOdomProcCloud：找时间最近的 INS，RTK 固定解才当观测融进 ESKF（ProcessINS）；"
           "INS 经纬高按 config 的 system.ins_lat/lon/height 用 LocalCartesian 转到 map",
    "out": "/lightning/current_scan_world（这一帧点云按 LIO 位姿放进 map）；/lightning/odom 的 covariance[1] = 定位有效"
           "（1 有效，0 故障）、covariance[0] = 系统时间",
}


# ---------------------------------------------------------------- geodesy, files

def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)], axis=-1)


def lla_to_enu(lat, lon, h, origin):
    """East, north, up [m] of WGS84 points about origin (lat, lon, h): what GeographicLib::LocalCartesian::Forward
    gives lightning."""
    lat0, lon0, h0 = origin
    d = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float)) - \
        geodetic_to_ecef(lat0, lon0, h0)
    sl, cl = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    so, co = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    r = np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])
    return d @ r.T


def read_loc_origin(text):
    """(ins_lat, ins_lon, ins_height) of a lightning config (system section), or None."""
    vals = {}
    for k in ("ins_lat", "ins_lon", "ins_height"):
        m = re.search(r"^\s*%s\s*:\s*([-+0-9.eE]+)" % k, text, re.M)
        if not m:
            return None
        vals[k] = float(m.group(1))
    return vals["ins_lat"], vals["ins_lon"], vals["ins_height"]


def read_pcd(raw):
    """(N, 3) float64 xyz and (N,) intensity or None of a PCD (ascii or binary; not binary_compressed)."""
    head, pos = {}, 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        head[key.upper()] = rest.split()
        if key.upper() == "DATA":
            break
    fields = head["FIELDS"]
    sizes = [int(s) for s in head["SIZE"]]
    types = head["TYPE"]
    counts = [int(c) for c in head.get("COUNT", ["1"] * len(fields))]
    n = int(head.get("POINTS", [0])[0]) or int(head["WIDTH"][0]) * int(head.get("HEIGHT", ["1"])[0])
    kind = head["DATA"][0]
    if kind == "ascii":
        rows = np.array(raw[pos:].split(), dtype=np.float64).reshape(n, -1)
        cols, k = {}, 0
        for f, c in zip(fields, counts):
            cols[f] = rows[:, k]
            k += c
    elif kind == "binary":
        dt = np.dtype([(f if c == 1 else f, "%s%s%d" % ("<", {"F": "f", "I": "i", "U": "u"}[t], s), (c,) if c > 1
                       else ()) for f, s, t, c in zip(fields, sizes, types, counts)])
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=pos)
        cols = {f: arr[f] for f in fields}
    else:
        raise ValueError("PCD DATA %s is not supported" % kind)
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    inten = cols.get("intensity")
    return xyz[ok], (np.asarray(inten, np.float64)[ok] if inten is not None else None)


def voxel_first(xyz, voxel):
    """Indices of one point per voxel."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int64)
    _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    return np.sort(first)


class Nearest:
    """Distance to the nearest map point (scipy's cKDTree; None without scipy)."""

    def __init__(self, xyz):
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(xyz) if len(xyz) else None
        except ImportError:
            self.tree = None

    def __call__(self, pts, cap=5.0):
        if self.tree is None:
            return None
        d, _ = self.tree.query(pts, k=1, distance_upper_bound=cap)
        return np.minimum(d, cap)


def roll_pitch(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


# ---------------------------------------------------------------- the layers

class LocLayers:
    def __init__(self, conv):
        self.c = conv
        self.ros = conv.ros
        self.rec = conv.rec
        self.notes = conv.notes

    def run(self):
        c = self.c
        self.z0 = float(np.median(c.track.z)) if len(c.track) else 0.0
        self.map_xyz = self.load_map()
        self.write_map()
        self.scan_stats = self.write_scans()
        self.ins = self.read_ins()
        self.write_ins()
        self.write_state()
        self.events()
        return self

    # -- map
    def load_map(self):
        c, a = self.c, self.c.args
        xyz, src = None, None
        raw = self.rec["loc_raw"].get("map_cloud")
        if raw:
            typ = self.rec["types"][c.topics["map_cloud"]]
            from planning_mcap_viz import cloud_xyz
            xyz, _ = cloud_xyz(self.ros.decode(raw[-1][1], typ))
            src = c.topics["map_cloud"]
        if xyz is None and a.loc_map:
            from planning_mcap_viz import Source
            s = Source(a.loc_map)
            names = s.find(r"(^|/)3d/global\.pcd$") or s.find(r"(^|/)global\.pcd$")
            if names:
                xyz, _ = read_pcd(s.read(names[0]))
                src = "%s:%s" % (a.loc_map, names[0])
        if xyz is None or len(xyz) == 0:
            return None
        xmin, ymin, xmax, ymax = c.bounds()
        keep = (xyz[:, 0] > xmin) & (xyz[:, 0] < xmax) & (xyz[:, 1] > ymin) & (xyz[:, 1] < ymax)
        xyz = xyz[keep]
        self.notes.append("localization map: %s, %d points around the run" % (src, len(xyz)))
        return xyz

    def write_map(self):
        if self.map_xyz is None or len(self.map_xyz) == 0:
            self.nearest = None
            return
        c = self.c
        pts = self.map_xyz[voxel_first(self.map_xyz, MAP_VOXEL)]
        self.nearest = Nearest(self.map_xyz[voxel_first(self.map_xyz, MATCH_VOXEL)])
        if self.nearest.tree is None:
            self.notes.append("no scipy: scans are drawn without their match to the map")
        h = pts[:, 2] - self.z0
        for t in np.arange(c.t0, c.t1 + 1e-6, MAP_PERIOD):
            msg = cloud_msg(self.ros, float(t), c.fixed, pts, {"height": h})
            c.out.ros_msg("/viz/loc/map_cloud", "sensor_msgs/msg/PointCloud2", float(t), msg)

    # -- scans
    def write_scans(self):
        c = self.c
        rows = self.rec["loc_raw"].get("scan_world", [])
        if not rows:
            return []
        from planning_mcap_viz import cloud_xyz, stamp_of
        typ = self.rec["types"][c.topics["scan_world"]]
        stats, last_drawn = [], -1e18
        for log, data in rows:
            m = self.ros.decode(data, typ)
            t = c.data_time(stamp_of(m.header), log)
            xyz, _ = cloud_xyz(m)
            if xyz is None or len(xyz) == 0:
                continue
            n_raw = len(xyz)
            rx, ry, _ = c.track.at(t)
            xyz = xyz[np.hypot(xyz[:, 0] - rx, xyz[:, 1] - ry) < SCAN_RANGE]
            xyz = xyz[voxel_first(xyz, SCAN_VOXEL)]
            match = self.nearest(xyz) if self.nearest is not None else None
            fields = {"height": xyz[:, 2] - self.z0}
            if match is not None:
                fields["match"] = match
            if t - last_drawn >= 1.0 / SCAN_DRAW_RATE - 1e-3:
                c.out.ros_msg("/viz/loc/scan", "sensor_msgs/msg/PointCloud2", t,
                              cloud_msg(self.ros, t, c.fixed, xyz, fields))
                last_drawn = t
            st = dict(t=t, points=n_raw, drawn=len(xyz))
            if match is not None:
                st.update(match_median=float(np.median(match)), match_inlier=float(np.mean(match < MATCH_GOOD)))
            stats.append(st)
        stats.sort(key=lambda s: s["t"])
        self.notes.append("%s -> /viz/loc/scan: %d scans%s" % (
            c.topics["scan_world"], len(stats), "" if not stats or "match_inlier" not in stats[0] else
            ", on the map (< %.1f m) median %.0f%%" % (MATCH_GOOD, 100 * np.median([s["match_inlier"] for s in stats]))))
        return stats

    # -- INS
    def read_ins(self):
        c, a = self.c, self.c.args
        rows = self.rec["loc_raw"].get("inspvax", [])
        if not rows:
            return None
        origin = None
        if a.loc_config:
            try:
                origin = read_loc_origin(open(a.loc_config, encoding="utf-8", errors="replace").read())
            except OSError:
                origin = None
        if origin is None:
            self.notes.append("INS: no ins_lat / ins_lon / ins_height in %s, not drawn" % a.loc_config)
            return None
        typ = self.rec["types"][c.topics["inspvax"]]
        from planning_mcap_viz import stamp_of
        out = []
        for log, data in rows:
            m = self.ros.decode(data, typ)
            out.append((c.data_time(stamp_of(m.header), log), m.latitude, m.longitude, m.height, m.azimuth,
                        int(m.ins_status.data), int(m.pos_type.data)))
        out.sort()
        a_ = np.array(out, dtype=float)
        enu = lla_to_enu(a_[:, 1], a_[:, 2], a_[:, 3], origin)
        ins = dict(t=a_[:, 0], x=enu[:, 0], y=enu[:, 1], z=enu[:, 2], azimuth=a_[:, 4], status=a_[:, 5].astype(int),
                   pos_type=a_[:, 6].astype(int))
        ins["fixed"] = (ins["status"] == INS_VALID_STATUS) & (ins["pos_type"] == RTK_FIXED)
        tr = c.track
        inside = (ins["t"] >= tr.t[0]) & (ins["t"] <= tr.t[-1])
        dx = np.interp(ins["t"], tr.t, tr.x) - ins["x"]
        dy = np.interp(ins["t"], tr.t, tr.y) - ins["y"]
        ins["diff"] = np.where(inside, np.hypot(dx, dy), np.nan)
        self.notes.append("INS: %d messages, RTK fixed %.0f%%, odom - INS median %.2f m (origin %.6f, %.6f, %.1f)" % (
            len(out), 100 * np.mean(ins["fixed"]), np.nanmedian(ins["diff"]) if inside.any() else float("nan"),
            *origin))
        return ins

    def write_ins(self):
        if self.ins is None:
            return
        c, ros, ins = self.c, self.ros, self.ins
        from planning_mcap_viz import LINE_STRIP, SPHERE, TEXT, Z
        # points every 0.2 m along the INS track
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ins["x"]), np.diff(ins["y"])))])
        keep = np.concatenate([[0], np.nonzero(np.diff(np.floor(d / 0.2)) > 0)[0] + 1])
        for t in np.arange(c.t0, c.t1 + 1e-6, 0.5):
            k = int(np.searchsorted(ins["t"][keep], t, side="right"))
            ms = [ros.delete_all(float(t), c.fixed)]
            if k >= 2:
                idx = keep[:k]
                ms.append(ros.marker(float(t), c.fixed, "ins_track", 1, LINE_STRIP, ros.pose(0, 0, Z["trail"] + 0.003),
                                     (0.05, 0, 0), (1, 0, 1, 1),
                                     points=[(ins["x"][i], ins["y"][i], 0) for i in idx],
                                     colors=[(1.0, 0.2, 1.0, 0.9) if ins["fixed"][i] else (0.6, 0.6, 0.6, 0.9)
                                             for i in idx]))
            j = int(np.searchsorted(ins["t"], t, side="right")) - 1
            if j >= 0 and t - ins["t"][j] < 1.0:
                fixed = bool(ins["fixed"][j])
                ms.append(ros.marker(float(t), c.fixed, "ins_now", 1, SPHERE,
                                     ros.pose(ins["x"][j], ins["y"][j], Z["robot"] + 0.01), (0.35, 0.35, 0.02),
                                     (1.0, 0.2, 1.0, 0.9) if fixed else (0.6, 0.6, 0.6, 0.9)))
                ms.append(ros.marker(float(t), c.fixed, "ins_now", 2, TEXT,
                                     ros.pose(ins["x"][j], ins["y"][j] - 0.8, Z["label"]), (0, 0, 0.3),
                                     (1.0, 0.6, 1.0, 0.9),
                                     text="INS %s，离定位 %.2f m" % ("RTK 固定" if fixed else "状态 %d/%d" % (
                                         ins["status"][j], ins["pos_type"][j]), ins["diff"][j])
                                     if np.isfinite(ins["diff"][j]) else "INS"))
            c.out.ros_msg("/viz/loc/ins", "visualization_msgs/msg/MarkerArray", float(t),
                          ros.new("visualization_msgs/msg/MarkerArray", markers=ms))

    # -- state and flow
    def write_state(self):
        c = self.c
        from planning_mcap_viz import jnum
        odom = c.odom
        odom_t = c.odom_t
        scans = self.scan_stats
        scan_t = [s["t"] for s in scans]
        ins = self.ins
        self.states = []
        for t in np.arange(c.t0, c.t1 + 1e-6, 1.0 / SCAN_RATE):
            t = float(t)
            j = int(np.searchsorted(odom_t, t, side="right")) - 1 if odom_t else -1
            valid = odom[j][9] if j >= 0 and len(odom[j]) > 9 and t - odom_t[j] < 0.5 else None
            R_q = c.track.q[min(int(np.searchsorted(c.track.t, t)), len(c.track.t) - 1)]
            roll, pitch = roll_pitch(R_q)
            k = int(np.searchsorted(scan_t, t, side="right")) - 1
            sc = scans[k] if k >= 0 and t - scans[k]["t"] < 0.5 else {}
            i = int(np.searchsorted(ins["t"], t, side="right")) - 1 if ins is not None else -1
            insd = dict(fixed=bool(ins["fixed"][i]), status=int(ins["status"][i]), pos_type=int(ins["pos_type"][i]),
                        diff=float(ins["diff"][i])) if i >= 0 and t - ins["t"][i] < 0.5 else {}
            s = dict(t=t, valid=valid, z=float(np.interp(t, c.track.t, c.track.z)), roll=roll, pitch=pitch,
                     speed=c.track.speed(t), scan_points=sc.get("points"), match_median=sc.get("match_median"),
                     match_inlier=sc.get("match_inlier"), ins_fixed=insd.get("fixed"), ins_status=insd.get("status"),
                     ins_pos_type=insd.get("pos_type"), odom_minus_ins=insd.get("diff"))
            self.states.append(s)
            obj = {k: (jnum(v) if not isinstance(v, bool) else int(v)) for k, v in s.items() if k != "t"}
            obj["t"] = jnum(t - c.t0, 3)
            c.out.json_msg("/viz/loc/state", "planning_viz.LocState", LOC_STATE_SCHEMA, t, obj)
            if sc or insd or valid is not None:
                c.out.json_msg("/viz/loc/flow", "planning_viz.LocFlow", FLOW_SCHEMA, t, self.flow(s))

    def flow(self, s):
        def f(v, fmt):
            return fmt % v if v is not None and v == v else "没有"
        return {
            "时间_s": round(s["t"] - self.c.t0, 2),
            "① IMU 预测（100 Hz）": {"做什么": "每来一帧 IMU，ESKF 往前推一步，直接发 /lightning/odom", "代码": FLOW_CODE["imu"],
                                 "速度_m/s": f(s["speed"], "%.2f"), "高度_m": f(s["z"], "%.2f"),
                                 "横滚/俯仰_deg": "%s / %s" % (f(s["roll"], "%.1f"), f(s["pitch"], "%.1f"))},
            "② 点云帧（10 Hz）": {"做什么": "去畸变、降采样", "代码": FLOW_CODE["scan"],
                             "这一帧的点数": f(s["scan_points"], "%d")},
            "③ LIO 和地图配准": {"做什么": "点到面 IEKF，把这一帧贴到地图上（地图按位姿分块加载）", "代码": FLOW_CODE["lio"],
                             "贴在地图上的点（< %.1f m）" % MATCH_GOOD: f(None if s["match_inlier"] is None
                                                                  else 100 * s["match_inlier"], "%.0f%%"),
                             "到地图的距离中位_m": f(s["match_median"], "%.2f")},
            "④ 定位健康": {"做什么": "匹配点太少或残差太大算失败，连续多帧就重初始化", "代码": FLOW_CODE["health"],
                        "定位有效（odom covariance[1]）": {1.0: "有效", 0.0: "故障"}.get(s["valid"], "没有")},
            "⑤ INS 融合": {"做什么": "RTK 固定解当观测融进去", "代码": FLOW_CODE["ins"],
                         "INS": "没有" if s["ins_fixed"] is None else ("RTK 固定" if s["ins_fixed"] else
                                                                      "不是 RTK 固定（%s/%s）" % (s["ins_status"],
                                                                                              s["ins_pos_type"])),
                         "定位减 INS_m": f(s["odom_minus_ins"], "%.2f")},
            "⑥ 输出": {"代码": FLOW_CODE["out"]},
        }

    # -- events
    def events(self):
        c = self.c
        st = self.states
        prev = None
        for s in st:
            v = s["valid"]
            if v is None:
                continue
            if prev is not None and v != prev:
                c.event(s["t"], "info" if v else "error", "定位恢复有效" if v else "定位故障（/lightning/odom covariance[1] = 0）",
                        "loc")
            prev = v
        prev = None
        for s in st:
            v = s["ins_fixed"]
            if v is None:
                continue
            if prev is not None and v != prev:
                c.event(s["t"], "info" if v else "warn", "INS 回到 RTK 固定解" if v else
                        "INS 不是 RTK 固定解了（状态 %s，解类型 %s）：lightning 不再融它" % (s["ins_status"], s["ins_pos_type"]),
                        "ins")
            prev = v
        on = False
        for s in st:
            d = s["odom_minus_ins"]
            if d is None or d != d:
                continue
            if not on and d > INS_FAR:
                c.event(s["t"], "warn", "定位和 INS 差了 %.2f m" % d, "loc")
                on = True
            elif on and d < INS_FAR / 2:
                c.event(s["t"], "info", "定位和 INS 又对上了（%.2f m）" % d, "loc")
                on = False
        on = False
        for s in st:
            m = s["match_inlier"]
            if m is None:
                continue
            if not on and m < MATCH_LOW:
                c.event(s["t"], "warn", "点云只有 %.0f%% 贴在地图上：地图和现场对不上，或位姿偏了" % (100 * m), "loc")
                on = True
            elif on and m > MATCH_LOW + 0.1:
                c.event(s["t"], "info", "点云又贴回地图上了（%.0f%%）" % (100 * m), "loc")
                on = False
        for log, text in self.rec["loc_raw_events"]:
            c.event(c.clock(log), "warn", "lightning 健康事件：%s" % text, "loc")
        self.read_glog()

    def read_glog(self):
        import os
        c, a = self.c, self.c.args
        path = a.loc_glog
        if path is None:
            base = a.recording if os.path.isdir(a.recording) else os.path.dirname(a.recording)
            cand = os.path.join(base, "logs", "loc.log")
            path = cand if os.path.isfile(cand) else None
        if not path:
            return
        from planning_mcap_viz import LOG, parse_glog
        wall0 = c.clock.log[0] if c.clock.log is not None else None
        lines = parse_glog(path, default_year=None if wall0 is None else __import__("time").gmtime(wall0).tm_year)
        if not lines:
            return
        tz = round((lines[0][0] - wall0) / 900.0) * 900.0 if wall0 is not None else 0.0
        kept, last = 0, {}
        for naive, level, where, text in lines:
            if level == "info" and not any(k in text for k in LOC_KEYWORDS):
                continue
            t = c.clock(naive - tz)
            norm = re.sub(r"[-+]?\d+(\.\d+)?", "#", text)
            if norm in last and t - last[norm] < 5.0:
                continue
            last[norm] = t
            c.events.append((float(t), LOG[level], "lightning", "%s] %s" % (where, text)))
            kept += 1
        self.notes.append("lightning log %s: %d lines, %d events" % (path, len(lines), kept))


LOC_STATE_SCHEMA = {"type": "object", "title": "planning_viz.LocState",
                    "properties": {k: {"type": ["number", "null"]} for k in (
                        "t", "valid", "z", "roll", "pitch", "speed", "scan_points", "match_median", "match_inlier",
                        "ins_fixed", "ins_status", "ins_pos_type", "odom_minus_ins")}}
FLOW_SCHEMA = {"type": "object", "title": "planning_viz.Flow", "additionalProperties": True}


def cloud_msg(ros, t, frame, xyz, fields):
    """PointCloud2 with float32 x, y, z and the given float fields."""
    names = ["x", "y", "z"] + list(fields)
    pts = np.zeros(len(xyz), dtype=[(n, "<f4") for n in names])
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for n, v in fields.items():
        pts[n] = v
    pf = [ros.new("sensor_msgs/msg/PointField", name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(names)]
    return ros.new("sensor_msgs/msg/PointCloud2", header=ros.header(t, frame), height=1, width=len(pts), fields=pf,
                   is_bigendian=False, point_step=4 * len(names), row_step=4 * len(names) * len(pts),
                   data=np.frombuffer(pts.tobytes(), dtype=np.uint8), is_dense=True)
