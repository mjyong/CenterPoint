#!/usr/bin/env python3
"""A bag recorded on the dog (mcap; sqlite3 first through bag_to_mcap.py) -> a small bag for debugging the planner
alone: the planner's inputs under their own names, so that `ros2 bag play` feeds a planner with nothing else running,
and what the planner on the dog made of them under /field/..., to compare with (they do not collide with what the
replayed planner publishes). No lidar, cameras, IMU or INS.

    python3 Nav_D20/planning/tools/planning_debug_bag.py data/examples/20261115_x6_clip \
        -o data/examples/20261115_x6_clip/planning_debug [--goal X Y [YAW_DEG]] [--start S] [--end E]

Inputs (the planner's subscriptions, robot_factory/d20/001/planning_conf/communication_conf.pb.txt):
  /lightning/odom, /map/occupancy_grid, /tf (without the planner's own base_link -> ptz_*), /tf_static, /goal_pose,
  /perception/lidar_3d/objects as nav_interface/msg/LidarObjectArray (bags before 2026-11-25 have it as
  perception_interface/msg/LidarObjectArray: the same fields, so the same bytes under the new type),
  and /map/switch, /initialpose, /mc/state when recorded.
/goal_pose is written once, at the start, in map: the goal of a bag is often in base_link (Foxglove's Publish pose
follows the 3D panel's frame, T66) and a clip re-stamps it to the clip's start; it is put in the map with the pose at
its stamp, or given with --goal. The field planner's global path end is printed beside it.
The field planner's outputs: /planning/debug/* and /nav_cw/shadow/* -> /field/planning/debug/*, /field/nav_cw/shadow/*.

Writes <out>/<name>_0.mcap (zstd chunks; --name, default the directory's name) and metadata.yaml: a ros2 bag
directory (ros2 bag play <out>; Foxglove opens the .mcap).
"""
import argparse
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OBJECTS = "/perception/lidar_3d/objects"
OBJECTS_TYPE = "nav_interface/msg/LidarObjectArray"
LEGACY_OBJECTS_TYPE = "perception_interface/msg/LidarObjectArray"
GOAL = "/goal_pose"
INPUTS = ("/lightning/odom", "/map/occupancy_grid", "/tf", "/tf_static", GOAL, OBJECTS, "/map/switch", "/initialpose",
          "/mc/state")
FIELD_PREFIXES = ("/planning/debug/", "/nav_cw/shadow/")
FIELD = "/field"
PLANNER_TF_CHILDREN = ("ptz_",)       # d20_ptz_bridge: the replayed planner publishes these itself
GLOBAL_PATH_NS = "global_path_2d"


def mcap_paths(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mcap"))
    if not files:
        if any(f.endswith(".db3") for f in os.listdir(path)):
            raise SystemExit("%s: sqlite3; convert it first: bag_to_mcap.py %s %s_mcap" % (path, path, path))
        raise SystemExit("%s: no .mcap" % path)
    return files


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Bag:
    """The messages of one or more mcap files in log-time order, with their channels' QoS."""

    def __init__(self, path):
        from mcap.reader import make_reader
        self.files = mcap_paths(path)
        self.make_reader = make_reader
        self.channels = {}      # topic -> (schema name, schema text, offered_qos_profiles)
        for f in self.files:
            with open(f, "rb") as fh:
                s = make_reader(fh).get_summary()
                for c in s.channels.values():
                    sc = s.schemas[c.schema_id]
                    self.channels.setdefault(c.topic, (sc.name, sc.data, c.metadata.get("offered_qos_profiles", "")))

    def messages(self, topics=None):
        for f in self.files:
            with open(f, "rb") as fh:
                for schema, channel, message in self.make_reader(fh).iter_messages(topics=topics,
                                                                                   log_time_order=True):
                    yield channel.topic, schema.name, message


def out_topic(topic):
    """The topic in the debug bag, or None when it is left out."""
    if topic in INPUTS:
        return topic
    if topic.startswith(FIELD_PREFIXES):
        return FIELD + topic
    return None


def first_pose(bag, ts):
    """[(stamp, x, y, yaw)] of map -> base_link from /tf, else from /lightning/odom."""
    tf, odom = [], []
    for topic, typ, m in bag.messages(["/tf", "/lightning/odom"]):
        msg = ts.deserialize_cdr(m.data, typ)
        if topic == "/tf":
            for tr in msg.transforms:
                if tr.header.frame_id.lstrip("/") == "map" and tr.child_frame_id.lstrip("/") == "base_link":
                    t = tr.transform.translation
                    tf.append((stamp(tr.header), t.x, t.y, yaw_of(tr.transform.rotation)))
        else:
            p = msg.pose.pose
            odom.append((stamp(msg.header), p.position.x, p.position.y, yaw_of(p.orientation)))
    return sorted(tf if len(tf) >= 2 else odom)


def pose_at(poses, t):
    import bisect
    i = min(max(bisect.bisect_left([p[0] for p in poses], t), 0), len(poses) - 1)
    if i > 0 and abs(poses[i - 1][0] - t) < abs(poses[i][0] - t):
        i -= 1
    return poses[i]


def field_goal(bag, ts):
    """The field planner's global path end in map (its /planning/debug/path_tracking), or None."""
    for _topic, typ, m in bag.messages(["/planning/debug/path_tracking"]):
        for mk in ts.deserialize_cdr(m.data, typ).markers:
            if mk.ns == GLOBAL_PATH_NS and mk.action == 0 and len(mk.points) >= 2 and \
                    mk.header.frame_id.lstrip("/") in ("", "map"):
                return mk.points[-1].x, mk.points[-1].y
    return None


def make_goal(ts, recorded, poses, given, t0):
    """(PoseStamped in map, where it came from). recorded: (typ, raw) of the bag's first /goal_pose or None."""
    from planning_mcap_viz import Ros
    ros = Ros()
    if given is not None:
        x, y = given[0], given[1]
        yaw = math.radians(given[2]) if len(given) > 2 else None
        src = "--goal"
    elif recorded is not None:
        g = ts.deserialize_cdr(recorded[1], recorded[0])
        frame = g.header.frame_id.lstrip("/")
        gx, gy, gyaw = g.pose.position.x, g.pose.position.y, yaw_of(g.pose.orientation)
        if frame in ("", "map"):
            x, y, yaw, src = gx, gy, gyaw, "the bag's /goal_pose (map)"
        elif frame == "base_link":
            st = stamp(g.header)
            _, px, py, pyaw = pose_at(poses, st)
            c, s = math.cos(pyaw), math.sin(pyaw)
            x, y, yaw = px + c * gx - s * gy, py + s * gx + c * gy, pyaw + gyaw
            src = "the bag's /goal_pose (base_link %.2f %.2f) with the pose at its stamp" % (gx, gy)
        else:
            raise SystemExit("/goal_pose in %s: give the goal in map with --goal X Y [YAW_DEG]" % frame)
    else:
        return None, "none in the bag (give --goal X Y [YAW_DEG])"
    if yaw is None:
        _, px, py, _ = pose_at(poses, t0)
        yaw = math.atan2(y - py, x - px)
    msg = ros.new("geometry_msgs/msg/PoseStamped", header=ros.header(t0, "map"), pose=ros.pose(x, y, 0.0, yaw))
    return msg, src


def build(src, out, goal=None, start=None, end=None, name=None):
    from bag_on_map import typestore
    from mcap.writer import CompressionType, Writer
    ts = typestore(all_packages=True, quiet=True)
    bag = Bag(src)
    poses = first_pose(bag, ts)
    if not poses:
        raise SystemExit("%s: no map -> base_link on /tf and no /lightning/odom" % src)
    t_first = poses[0][0]
    lo = t_first + (start or 0.0)
    hi = t_first + end if end is not None else float("inf")
    recorded_goal = None
    for _topic, typ, m in bag.messages([GOAL]):
        recorded_goal = (typ, m.data)
        break
    goal_msg, goal_src = make_goal(ts, recorded_goal, poses, goal, lo)

    name = name or os.path.basename(os.path.normpath(out))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "%s_0.mcap" % name)
    counts, types, qos = Counter(), {}, {}
    first_ns = last_ns = None
    dropped_tf = Counter()
    with open(path, "wb") as fh:
        w = Writer(fh, chunk_size=4 * 1024 * 1024, compression=CompressionType.ZSTD)
        w.start(profile="ros2", library="planning_debug_bag")
        schema_ids, channel_ids = {}, {}

        def channel(topic, typ, text, profile):
            if topic not in channel_ids:
                if typ not in schema_ids:
                    schema_ids[typ] = w.register_schema(name=typ, encoding="ros2msg", data=text)
                channel_ids[topic] = w.register_channel(topic=topic, message_encoding="cdr",
                                                        schema_id=schema_ids[typ],
                                                        metadata={"offered_qos_profiles": profile})
                types[topic], qos[topic] = typ, profile
            return channel_ids[topic]

        def put(topic, typ, text, profile, log_ns, data):
            nonlocal first_ns, last_ns
            cid = channel(topic, typ, text, profile)
            w.add_message(channel_id=cid, log_time=log_ns, publish_time=log_ns, data=data, sequence=counts[topic])
            counts[topic] += 1
            first_ns = log_ns if first_ns is None else min(first_ns, log_ns)
            last_ns = log_ns if last_ns is None else max(last_ns, log_ns)

        objects_text = ts.generate_msgdef(OBJECTS_TYPE, ros_version=2)[0].encode()
        goal_written = False
        static_written = False
        for topic, typ, m in bag.messages(sorted(t for t in bag.channels if out_topic(t))):
            dst = out_topic(topic)
            t = m.log_time * 1e-9
            if topic == "/tf_static":
                if static_written and t < lo:     # the static TF from before the start, once, at the start
                    continue
            elif t < lo or t > hi:
                continue
            log_ns = max(m.log_time, int(round(lo * 1e9)))
            _sname, text, profile = bag.channels[topic]
            data = m.data
            if topic == GOAL:
                continue                            # written once, in map, below
            if topic == OBJECTS:
                if typ not in (OBJECTS_TYPE, LEGACY_OBJECTS_TYPE):
                    raise SystemExit("%s is %s, not %s" % (OBJECTS, typ, OBJECTS_TYPE))
                typ, text = OBJECTS_TYPE, objects_text
            elif topic == "/tf":
                msg = ts.deserialize_cdr(data, typ)
                keep = [tr for tr in msg.transforms
                        if not tr.child_frame_id.lstrip("/").startswith(PLANNER_TF_CHILDREN)]
                if len(keep) != len(msg.transforms):
                    dropped_tf.update(tr.child_frame_id for tr in msg.transforms if tr not in keep)
                    if not keep:
                        continue
                    msg.transforms = keep
                    data = bytes(ts.serialize_cdr(msg, typ))
            if not goal_written and goal_msg is not None and topic not in ("/tf_static",):
                # the goal after the first TF and odometry of the start: the planner transforms nothing, but a
                # player started late still has the dog's pose before the goal
                if topic in ("/tf", "/lightning/odom") and counts["/tf"] and counts["/lightning/odom"]:
                    gtyp = "geometry_msgs/msg/PoseStamped"
                    put(GOAL, gtyp, ts.generate_msgdef(gtyp, ros_version=2)[0].encode(), bag.channels.get(
                        GOAL, (None, None, ""))[2], log_ns, bytes(ts.serialize_cdr(goal_msg, gtyp)))
                    goal_written = True
            put(dst, typ, text, profile, log_ns, data)
            if topic == "/tf_static":
                static_written = True
        w.finish()
    write_metadata(out, os.path.basename(path), first_ns, last_ns, counts, types, qos)
    end_goal = field_goal(bag, ts)
    return dict(path=path, counts=counts, types=types, goal=goal_msg, goal_src=goal_src, field_goal=end_goal,
                dropped_tf=dropped_tf, start=first_ns * 1e-9 if first_ns else None,
                duration=(last_ns - first_ns) * 1e-9 if first_ns else 0.0, bag_start=t_first)


def write_metadata(out, fname, first_ns, last_ns, counts, types, qos):
    """rosbag2 metadata.yaml (the version bag_to_mcap.py's bags carry, which the Humble player reads)."""
    def q(s):
        return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") if s else "''"
    lines = ["rosbag2_bagfile_information:", "  version: 8", "  storage_identifier: mcap",
             "  duration:", "    nanoseconds: %d" % (last_ns - first_ns),
             "  starting_time:", "    nanoseconds_since_epoch: %d" % first_ns,
             "  message_count: %d" % sum(counts.values()), "  topics_with_message_count:"]
    for topic in sorted(counts):
        lines += ["  - topic_metadata:", "      name: %s" % topic, "      type: %s" % types[topic],
                  "      serialization_format: cdr", "      offered_qos_profiles: %s" % q(qos[topic]),
                  "      type_description_hash: ''", "    message_count: %d" % counts[topic]]
    lines += ["  compression_format: ''", "  compression_mode: ''", "  relative_file_paths:", "  - %s" % fname,
              "  files:", "  - path: %s" % fname, "    starting_time:",
              "      nanoseconds_since_epoch: %d" % first_ns, "    duration:",
              "      nanoseconds: %d" % (last_ns - first_ns), "    message_count: %d" % sum(counts.values()),
              "  custom_data: null", "  ros_distro: humble", ""]
    with open(os.path.join(out, "metadata.yaml"), "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="a bag recorded on the dog: an mcap bag directory or one .mcap")
    p.add_argument("-o", "--output", required=True, help="the bag directory to write")
    p.add_argument("--goal", type=float, nargs="+", metavar="X Y [YAW_DEG]", help="the goal in map")
    p.add_argument("--start", type=float, help="[s] after the bag's first pose")
    p.add_argument("--end", type=float, help="[s] after the bag's first pose")
    p.add_argument("--name", help="the .mcap is <name>_0.mcap (default the output directory's name)")
    args = p.parse_args(argv)
    if args.goal is not None and len(args.goal) not in (2, 3):
        p.error("--goal X Y [YAW_DEG]")
    if os.path.isdir(args.output) and any(f.endswith(".mcap") for f in os.listdir(args.output)):
        p.error("%s already has an .mcap" % args.output)
    r = build(args.bag, args.output, args.goal, args.start, args.end, args.name)
    total = sum(r["counts"].values())
    size = os.path.getsize(r["path"]) / 1e6
    print("%s: %d topics, %d messages, %.1f s, %.1f MB" % (r["path"], len(r["counts"]), total, r["duration"], size))
    for topic in sorted(r["counts"]):
        print("  %-52s %-45s %6d" % (topic, r["types"][topic], r["counts"][topic]))
    if r["dropped_tf"]:
        print("  /tf without the planner's own: %s" % ", ".join("%s %d" % kv for kv in sorted(r["dropped_tf"].items())))
    g = r["goal"]
    if g is None:
        print("  goal: %s" % r["goal_src"])
    else:
        gx, gy = g.pose.position.x, g.pose.position.y
        gyaw = math.degrees(yaw_of(g.pose.orientation))
        line = "  goal in map: %.3f %.3f yaw %.1f deg, from %s" % (gx, gy, gyaw, r["goal_src"])
        if r["field_goal"]:
            fx, fy = r["field_goal"]
            line += "; the field planner's global path ends at %.3f %.3f (%.2f m away)" % (
                fx, fy, math.hypot(fx - gx, fy - gy))
        print(line)
        print("  replay: replay_planning_check.sh <name> %.3f %.3f --bag %s --planning-only" % (gx, gy, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
