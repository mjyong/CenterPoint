#!/usr/bin/env python3
"""Why a planning recording (ros2 bag, mcap) looks wrong in Foxglove: checks of what the 3D panel needs.

    python3 Nav_D20/planning/tools/check_planning_mcap.py <bag.mcap | bag dir> \
        [--layout Nav_D20/planning/script/planning_io.json] [--display-frame map]

1. TF tree (/tf, /tf_static): every child frame has one parent and there is no cycle. planning's TfBuffer used to
   publish each transform and its inverse (map -> base_link and base_link -> map) plus map -> ptz_* next to
   base_link -> ptz_*; a frame with several parents jumps between them and a cycle has no root.
2. Time base: Foxglove places a message with the transform at its header stamp. A topic whose frame is not the
   display frame needs the TF chain to cover its stamps; planning stamped its base_link costmap with the wall
   clock, weeks after the odometry stamps of a replayed bag (and markers with 0).
3. Marker lifetime against the publish period: a lifetime of about one period blinks in playback and a paused
   frame often shows nothing (planning used 0.05 s at 20 Hz).
4. Odometry (nav_msgs/Odometry) a walking robot can have: speed between consecutive messages, height swing,
   jumps. The T41 replay's /lightning/odom ran away between corrections (up to 32 m/s, z down to -46 m) because
   the localization's IMU-rate prediction took the kygi IMU's acceleration in g as m/s^2.
5. Occupancy grids in base_link with the robot standing on an obstacle (at least 10 % of its footprint's cells,
   --footprint, D20 1.0 x 0.4 m): the local planner refuses to plan from there ("Initial pose is in collision").
   On a route-network map (CJGC) that is the dog off the map's route.
   Planning's costmap topic was published inverted until the T47 review (the costmap's obstacle layer holds
   255 = free, 0 = obstacle and ros_debug_bridge converted it with (0, 1)): free cells read 100, obstacles 0. Told
   from perception's obstacles, which planning copies into the costmap as obstacles; reported, and turned over
   before the footprint check. (T42 and T47 read "footprint occupied in 2 of 3 grids" from the inverted topic:
   the dog was on the route then.)
6. With --layout: the layout's topics that the recording does not have.

Prints one line per problem; exit code 1 when there is any. Needs the mcap and rosbags packages.
"""
import argparse
import glob
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np

HEADER_FIRST = ("std_msgs/Header header", "std_msgs/msg/Header header")
DRAWN = ("nav_msgs/msg/OccupancyGrid", "nav_msgs/msg/Odometry", "nav_msgs/msg/Path", "sensor_msgs/msg/PointCloud2",
         "sensor_msgs/msg/LaserScan", "geometry_msgs/msg/PoseStamped", "geometry_msgs/msg/PoseArray",
         "geometry_msgs/msg/PolygonStamped") + ("visualization_msgs/msg/Marker", "visualization_msgs/msg/MarkerArray")
MAX_SPEED = 4.0       # [m/s] the dogs walk below about 2 m/s
MAX_Z_SWING = 3.0     # [m] height change over a recording
MAX_JUMP = 1.0        # [m] between consecutive odometry messages
SPEED_WINDOW = 0.2    # [s] speed over at least this: 100 Hz INS odometry 1-2 ms apart reads metres per second from
                      # centimetre jitter (T43 B)
FOOTPRINT_SHARE = 0.1  # a grid with this share of the footprint's cells occupied has the robot on an obstacle;
                       # reported when this share of the grids have it
COST_MAP = "/planning/debug/cost_map_production/occmap"
PERCEPTION = "/map/occupancy_grid"
POLARITY_EVERY = 10    # every n-th costmap is kept for the polarity check
MARKER_TYPES = ("visualization_msgs/msg/Marker", "visualization_msgs/msg/MarkerArray")
TF_TYPES = ("tf2_msgs/msg/TFMessage",)
SLACK = 1.0   # [s] a stamp this far outside the TF's time range still counts as covered


def _stamp_frame(data):
    """(stamp [s], frame_id) of a CDR message whose first field is a std_msgs/Header."""
    endian = "<" if data[1] == 1 else ">"
    sec, nsec, n = struct.unpack_from(endian + "iII", data, 4)
    return sec + nsec * 1e-9, data[16:16 + max(n - 1, 0)].decode("utf-8", "replace")


def _starts_with_header(schema):
    for line in schema.decode("utf-8", "replace").splitlines():
        line = line.split("#")[0].strip()
        if line:
            return line in HEADER_FIRST
    return False


def read(path):
    """{topic: dict(type, stamps [s], frames set, log [s], lifetimes [s])} and the TF edges
    [(parent, child, stamp, static)]."""
    from mcap.reader import make_reader
    from rosbags.typesys import Stores, get_typestore

    files = sorted(glob.glob(os.path.join(path, "*.mcap"))) if os.path.isdir(path) else [path]
    if not files:
        raise SystemExit("%s: no .mcap" % path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    topics = defaultdict(lambda: dict(type=None, stamps=[], frames=set(), log=[], lifetimes=[], xyz=[],
                                      footprint=[], grids=[]))
    edges = []
    for f in files:
        with open(f, "rb") as fh:
            for schema, channel, message in make_reader(fh).iter_messages():
                t = topics[channel.topic]
                t["type"] = schema.name if schema else None
                t["log"].append(message.log_time * 1e-9)
                if t["type"] in TF_TYPES:
                    msg = ts.deserialize_cdr(message.data, t["type"])
                    for tr in msg.transforms:
                        edges.append((tr.header.frame_id, tr.child_frame_id,
                                      tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9,
                                      channel.topic == "/tf_static"))
                elif t["type"] in MARKER_TYPES:
                    msg = ts.deserialize_cdr(message.data, t["type"])
                    for m in getattr(msg, "markers", [msg]):
                        if m.action == 3:          # DELETEALL carries no frame
                            continue
                        t["frames"].add(m.header.frame_id)
                        t["stamps"].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                        t["lifetimes"].append(m.lifetime.sec + m.lifetime.nanosec * 1e-9)
                elif t["type"] == "nav_msgs/msg/OccupancyGrid":
                    msg = ts.deserialize_cdr(message.data, t["type"])
                    t["stamps"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                    t["frames"].add(msg.header.frame_id)
                    if msg.header.frame_id == "base_link" and FOOT is not None:
                        t["footprint"].append(_footprint_cells(msg, *FOOT))
                        if channel.topic == PERCEPTION or (channel.topic == COST_MAP and
                                                           len(t["footprint"]) % POLARITY_EVERY == 1):
                            t["grids"].append((message.log_time * 1e-9, msg.info, np.asarray(msg.data, np.int8)))
                elif t["type"] == "nav_msgs/msg/Odometry":
                    msg = ts.deserialize_cdr(message.data, t["type"])
                    p = msg.pose.pose.position
                    t["stamps"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                    t["frames"].add(msg.header.frame_id)
                    t["xyz"].append((p.x, p.y, p.z))
                elif schema is not None and _starts_with_header(schema.data):
                    stamp, frame = _stamp_frame(message.data)
                    t["stamps"].append(stamp)
                    t["frames"].add(frame)
    return topics, edges


FOOT = (1.0, 0.4)      # [m] length (x) and width (y) of the robot, set by --footprint


def _footprint_cells(msg, length, width):
    """(cells >= 50, cells 0..49, all cells) under the robot's rectangle (centred on base_link, x forward)."""
    i = msg.info
    if i.width * i.height == 0 or len(msg.data) != i.width * i.height:
        return (0, 0, 0)
    d = np.asarray(msg.data, np.int16).reshape(i.height, i.width)
    xs = i.origin.position.x + (np.arange(i.width) + 0.5) * i.resolution
    ys = i.origin.position.y + (np.arange(i.height) + 0.5) * i.resolution
    cx = np.abs(xs) <= length / 2
    cy = np.abs(ys) <= width / 2
    under = d[np.ix_(cy, cx)]
    return (int((under >= 50).sum()), int(((under >= 0) & (under < 50)).sum()), int(under.size))


def cost_map_inverted(topics):
    """(True/False/None, evidence): whether the costmap reads free = 100, from the cells perception calls
    occupied (the newest perception grid before each kept costmap, both in base_link)."""
    cm, pg = topics.get(COST_MAP, {}).get("grids"), topics.get(PERCEPTION, {}).get("grids")
    if not cm or not pg:
        return None, "no %s to tell by" % PERCEPTION
    pg_log = [g[0] for g in pg]
    hits = total = 0
    for log, gi, gd in cm:
        k = int(np.searchsorted(pg_log, log, side="right")) - 1
        if k < 0:
            continue
        _, pi, pd = pg[k]
        r, c = np.nonzero(pd.reshape(pi.height, pi.width) >= 50)
        x = pi.origin.position.x + (c + 0.5) * pi.resolution - gi.origin.position.x
        y = pi.origin.position.y + (r + 0.5) * pi.resolution - gi.origin.position.y
        cc, rr = np.floor(x / gi.resolution).astype(int), np.floor(y / gi.resolution).astype(int)
        ok = (cc >= 0) & (cc < gi.width) & (rr >= 0) & (rr < gi.height)
        v = gd.reshape(gi.height, gi.width)[rr[ok], cc[ok]]
        hits += int((v >= 50).sum())
        total += int(len(v))
    if total < 50:
        return None, "only %d perception obstacle cells in the costmap" % total
    return hits / total < 0.5, "perception obstacles read 100 in %.1f%% of %d costmap cells" % (100.0 * hits / total,
                                                                                                total)


def footprint_problems(topics):
    out = []
    inverted, why = cost_map_inverted(topics)
    if inverted:
        out.append("%s: inverted, free cells 100 and obstacles 0 (%s): Foxglove shows the road as the obstacle "
                   "(ros_debug_bridge before the T47 review); tools/planning_mcap_viz.py turns it over" % (COST_MAP, why))
    for name, t in sorted(topics.items()):
        f = t["footprint"]
        if not f:
            continue
        occ = [(o if not (inverted and name == COST_MAP) else free) for o, free, n in f]
        on = [o >= FOOTPRINT_SHARE * n and n > 0 for o, (_, _, n) in zip(occ, f)]
        if np.mean(on) > FOOTPRINT_SHARE:
            out.append("%s: the robot stands on an obstacle (>= %d%% of its footprint) in %d of %d grids%s; the local "
                       "planner will not plan from there" % (name, 100 * FOOTPRINT_SHARE, int(np.sum(on)), len(on),
                                                             " (turned over)" if inverted and name == COST_MAP
                                                             else ""))
    return out


def display_notes(topics):
    """Grids drawn in the same moving frame: in Foxglove they share a plane that tilts with the robot."""
    by_frame = defaultdict(list)
    for name, t in sorted(topics.items()):
        if t["type"] == "nav_msgs/msg/OccupancyGrid":
            for f in t["frames"]:
                if f not in ("map", "odom", ""):
                    by_frame[f].append(name)
    return ["display: %s are all drawn in %s: the planes tilt with the robot and cut through each other in "
            "Foxglove (blinking); open tools/planning_mcap_viz.py's output instead" % (", ".join(v), f)
            for f, v in sorted(by_frame.items()) if len(v) > 1]


def tf_problems(edges):
    parents = defaultdict(set)
    for p, c, _, _ in edges:
        parents[c].add(p)
    out = ["TF: %s has %d parents (%s); Foxglove and RViz switch between them" % (c, len(ps), ", ".join(sorted(ps)))
           for c, ps in sorted(parents.items()) if len(ps) > 1]
    for start in sorted(parents):
        seen, f = [start], start
        while f in parents and len(parents[f]) == 1:
            f = next(iter(parents[f]))
            if f == start:
                out.append("TF: cycle %s" % " -> ".join(seen + [f]))
                break
            if f in seen:
                break
            seen.append(f)
    return sorted(set(out))


def chain(frame, display, parents):
    """Frames from frame up to display (one parent each), or None when there is no such path."""
    path, f = [frame], frame
    while f != display:
        ps = parents.get(f)
        if not ps or len(ps) != 1:
            return None
        f = next(iter(ps))
        if f in path:
            return None
        path.append(f)
    return path


def time_problems(topics, edges, display):
    parents, stamps, static = defaultdict(set), defaultdict(list), set()
    for p, c, s, st in edges:
        parents[c].add(p)
        stamps[c].append(s)
        if st:
            static.add(c)
    if display not in parents and not any(display in ps for ps in parents.values()):
        return ["TF: the display frame %s is in no transform" % display] if edges else []
    out = []
    for name, t in sorted(topics.items()):
        if not t["stamps"]:
            continue
        s = np.asarray(t["stamps"])
        for frame in sorted(t["frames"]):
            if frame in ("", display):
                if frame == "" and t["type"] in DRAWN:
                    out.append("%s: empty frame_id" % name)
                continue
            path = chain(frame, display, parents)
            if path is None:
                out.append("%s: no single TF path from %s to %s" % (name, frame, display))
                continue
            for child in path[:-1]:
                if child in static:
                    continue
                lo, hi = min(stamps[child]), max(stamps[child])
                outside = np.mean((s < lo - SLACK) | (s > hi + SLACK))
                if outside > 0.05:
                    out.append("%s (%s): %.0f%% of the header stamps are outside the %s TF stamps (%.1f .. %.1f); "
                               "stamps %.1f .. %.1f, %s" % (
                                   name, frame, 100 * outside, child, lo, hi, s.min(), s.max(),
                                   "stamp 0" if s.min() == 0 else "%.0f s apart" % abs(np.median(s) - (lo + hi) / 2)))
                    break
    return out


def marker_problems(topics):
    out = []
    for name, t in sorted(topics.items()):
        if t["type"] not in MARKER_TYPES or len(t["log"]) < 3 or not t["lifetimes"]:
            continue
        period = float(np.median(np.diff(np.sort(t["log"]))))
        life = [x for x in t["lifetimes"] if x > 0]
        if life and min(life) < 1.5 * period:
            out.append("%s: marker lifetime %.2f s against one message every %.2f s: they blink in playback and a "
                       "paused frame often has none" % (name, min(life), period))
    return out


def odom_problems(topics):
    out = []
    for name, t in sorted(topics.items()):
        if t["type"] != "nav_msgs/msg/Odometry" or len(t["xyz"]) < 3:
            continue
        xyz, s = np.asarray(t["xyz"]), np.asarray(t["stamps"])
        order = np.argsort(s, kind="stable")
        xyz, s = xyz[order], s[order]
        step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        j = np.searchsorted(s, s + SPEED_WINDOW)
        ok = j < len(s)
        i, j = np.nonzero(ok)[0], j[ok]
        speed = np.linalg.norm(xyz[j, :2] - xyz[i, :2], axis=1) / (s[j] - s[i])
        path, net = float(step.sum()), float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2]))
        what = []
        if speed.size and np.percentile(speed, 99) > MAX_SPEED:
            fast = s[i][speed > MAX_SPEED] - s[0]
            what.append("speed p99 %.1f m/s (max %.1f; over %.0f m/s from %.1f s to %.1f s)"
                        % (np.percentile(speed, 99), speed.max(), MAX_SPEED, fast.min(), fast.max()))
        if np.ptp(xyz[:, 2]) > MAX_Z_SWING:
            what.append("z %.1f .. %.1f m" % (xyz[:, 2].min(), xyz[:, 2].max()))
        if (step > MAX_JUMP).any():
            at = s[1:][step > MAX_JUMP] - s[0]
            what.append("%d jumps > %.0f m (largest %.1f m; from %.1f s to %.1f s)"
                        % ((step > MAX_JUMP).sum(), MAX_JUMP, step.max(), at.min(), at.max()))
        if what:
            out.append("%s: not a walking robot: %s; path %.0f m, net %.0f m over %.1f s" % (
                name, ", ".join(what), path, net, s[-1] - s[0]))
    return out


def layout_topics(path):
    with open(path) as f:
        cfg = json.load(f)
    names = set()
    for panel in cfg.get("configById", {}).values():
        names.update((panel.get("topics") or {}).keys())
    return names


def main(argv=None):
    global FOOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="an .mcap file or a ros2 bag directory of them")
    ap.add_argument("--layout", default=None, help="Foxglove layout JSON: report its topics the bag does not have")
    ap.add_argument("--display-frame", default="map")
    ap.add_argument("--footprint", type=float, nargs=2, default=list(FOOT), metavar=("LENGTH", "WIDTH"),
                    help="[m] robot rectangle for the occupancy check (robot_param.pb.txt: D20 1.0 x 0.4)")
    args = ap.parse_args(argv)
    FOOT = tuple(args.footprint)
    topics, edges = read(args.bag)
    problems = (tf_problems(edges) + time_problems(topics, edges, args.display_frame) + marker_problems(topics)
                + odom_problems(topics) + footprint_problems(topics))
    if args.layout:
        problems += ["layout: %s is not in the recording" % n for n in sorted(layout_topics(args.layout) - set(topics))]
    print("%s: %d topics, %d transforms" % (args.bag, len(topics), len(edges)))
    for name, t in sorted(topics.items()):
        frames = ",".join(sorted(t["frames"])) or "-"
        print("  %-48s %-36s %6d msgs  frames %s" % (name, t["type"], len(t["log"]), frames))
    for p in problems:
        print("PROBLEM " + p)
    for n in display_notes(topics):
        print("NOTE " + n)
    print("no problems found" if not problems else "%d problems" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
