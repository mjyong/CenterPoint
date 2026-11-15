"""One command to record a bag on dog 8 (started by nav_cw_record.sh, which sets up ROS as root).

Records DOG_FIELD_GUIDE.md section 9's topics, plus the planner's goal marker (/goal_pose is volatile: a goal sent
before the recording started is not in the bag, T57) and the robot's motion state, as sqlite3 (the dog's ros2 bag has
no mcap plugin, T61). It stops by itself once the planner has a goal and the dog, walking or remote-controlled, has
come within --radius (1.5 m) of it: --hold (5 s) later. Otherwise Ctrl-C stops it. Either way the recorder gets one
SIGINT and writes metadata.yaml; nothing opens the database while it is written (T57: "database is locked").

The goal is the latest of /planning/debug/goal_pose (the planner republishes the last /goal_pose every cycle) and
/goal_pose (a new one sent during the recording); the dog is /lightning/odom (map frame, like the goal). A goal the
dog is already within the radius of (the last test's) does not stop it: the dog has to be outside first.

    sudo bash /opt/seres/nav_cw/nav_cw_record.sh                       # /home/nvidia/bags/<date>_cw_<n>
    sudo bash /opt/seres/nav_cw/nav_cw_record.sh -o /home/nvidia/bags/x --radius 2 --hold 5 /mc/state_extra ...
"""
import argparse
import datetime
import math
import os
import re
import signal
import subprocess
import sys
import time

# DOG_FIELD_GUIDE.md section 9 (tests/dog/test_nav_cw_record.py checks the two agree), then what this adds
TOPICS = [
    "/robot/topic/rdap/top_point_cloud2", "/rte/rdap/rtk/inspvax", "/rte/rdap/rtk/imus/data_raw",
    "/lightning/odom", "/map/occupancy_grid", "/perception/lidar_3d/objects", "/goal_pose",
    "/cmd_vel", "/mc/motion_control", "/nav_cw/shadow/cmd_vel", "/nav_cw/shadow/mc/motion_control",
    "/planning/debug/crosswalk", "/planning/debug/path_tracking", "/planning/debug/plotjuggler/string",
    "/tf", "/tf_static",
]
ADDED = ["/planning/debug/goal_pose", "/mc/state"]
GOAL_MARKER, GOAL_POSE, ODOM = "/planning/debug/goal_pose", "/goal_pose", "/lightning/odom"
CACHE = 256 * 1024 * 1024        # [B] --max-cache-size (T35: the default dropped point clouds)


class Arrival:
    """When to stop: a goal, the dog seen farther than ``radius`` from it, then within ``radius``; ``hold`` seconds
    after that. Leaving the radius again, or a new goal (moved > 0.1 m), starts over."""

    def __init__(self, radius=1.5, hold=5.0):
        self.radius, self.hold = radius, hold
        self.goal = None
        self.armed = False          # the dog was outside the radius since the goal was set
        self.since = None           # when it came within the radius
        self.dist = None

    def set_goal(self, xy):
        """-> a line to print when it is a new goal, else None."""
        if self.goal is not None and math.dist(xy, self.goal) <= 0.1:
            return None
        self.goal, self.armed, self.since = tuple(xy), False, None
        return "goal (%.2f, %.2f)" % self.goal

    def set_pose(self, xy, now):
        """-> a line to print when something changes, else None."""
        if self.goal is None:
            return None
        self.dist = math.dist(xy, self.goal)
        if self.dist > self.radius:
            was = self.since
            self.since = None
            if not self.armed:
                self.armed = True
                return "goal %.1f m away: stops %g s after the dog is within %.1f m" % (self.dist, self.hold,
                                                                                      self.radius)
            if was is not None:
                return "left the %.1f m radius (%.1f m): not stopping" % (self.radius, self.dist)
            return None
        if self.armed and self.since is None:
            self.since = now
            return "within %.1f m of the goal (%.2f m): stopping in %g s (Ctrl-C stops at once)" % (
                self.radius, self.dist, self.hold)
        return None

    def due(self, now):
        return self.since is not None and now - self.since >= self.hold


def default_output(directory, today=None):
    """<directory>/<YYYYMMDD>_cw_<n>, n one more than the largest there for that day."""
    day = today or datetime.date.today().strftime("%Y%m%d")
    pat = re.compile(r"^%s_cw_(\d+)$" % re.escape(day))
    taken = [int(m.group(1)) for m in map(pat.match, os.listdir(directory) if os.path.isdir(directory) else [])
             if m]
    return os.path.join(directory, "%s_cw_%d" % (day, max(taken, default=0) + 1))


class Stop(Exception):
    pass


def _raise_stop(signum, frame):
    raise Stop(signal.Signals(signum).name)


def ros_feed(arrival, log):
    """Subscriptions that feed ``arrival``; -> (spin(timeout), close()), or None without rclpy."""
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from visualization_msgs.msg import Marker
    except ImportError as e:
        log("no rclpy (%s): no auto stop, Ctrl-C to stop" % e)
        return None
    rclpy.init()
    # subscriptions only: no rosout, no parameter services on the robot's domain
    node = rclpy.create_node("nav_cw_record", enable_rosout=False, start_parameter_services=False)

    def goal(msg):
        p = msg.pose.position
        line = arrival.set_goal((p.x, p.y))
        if line:
            log(line)

    def odom(msg):
        p = msg.pose.pose.position
        line = arrival.set_pose((p.x, p.y), time.monotonic())
        if line:
            log(line)

    # best effort takes both reliable and best-effort publishers
    node.create_subscription(Marker, GOAL_MARKER, goal, qos_profile_sensor_data)
    node.create_subscription(PoseStamped, GOAL_POSE, goal, qos_profile_sensor_data)
    node.create_subscription(Odometry, ODOM, odom, qos_profile_sensor_data)

    def close():
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return (lambda timeout: rclpy.spin_once(node, timeout_sec=timeout)), close


def stop_recorder(rec, log, wait=60.0):
    """One SIGINT to the recorder's process group (it is not in ours: Ctrl-C reaches only this script), then wait."""
    if rec.poll() is not None:
        return rec.returncode
    try:
        os.killpg(rec.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return rec.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        log("the recorder did not stop in %.0f s: SIGTERM (metadata.yaml may be missing)" % wait)
        os.killpg(rec.pid, signal.SIGTERM)
        return rec.wait()


def chown_tree(path, uid, gid):
    for root, dirs, files in os.walk(path):
        for n in [root] + [os.path.join(root, f) for f in dirs + files]:
            os.chown(n, uid, gid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("topics", nargs="*", help="more topics to record")
    ap.add_argument("-o", "--output", help="the bag directory (must not exist; default <dir>/<date>_cw_<n>)")
    ap.add_argument("--dir", default=os.environ.get("NAV_CW_BAG_DIR", "/home/nvidia/bags"))
    ap.add_argument("--radius", type=float, default=1.5, help="[m] the dog this close to the goal ...")
    ap.add_argument("--hold", type=float, default=5.0, help="[s] ... this long before the recording stops")
    ap.add_argument("--no-auto", action="store_true", help="only Ctrl-C stops it")
    ap.add_argument("--dry-run", action="store_true", help="print the record command")
    args = ap.parse_args(argv)

    def log(line):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), line), flush=True)

    out = args.output or default_output(args.dir)
    if os.path.exists(out):
        log("%s exists; ros2 bag record would refuse it" % out)
        return 2
    topics = TOPICS + ADDED + [t for t in args.topics if t not in TOPICS + ADDED]
    cmd = ["ros2", "bag", "record", "-s", "sqlite3", "--max-cache-size", str(CACHE), "-o", out] + topics
    if args.dry_run:
        print("# domain %s" % os.environ.get("ROS_DOMAIN_ID", "0"))
        print(" ".join(cmd))
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _raise_stop)
    log("recording %s, domain %s: %d topics" % (out, os.environ.get("ROS_DOMAIN_ID", "0"), len(topics)))
    rec = subprocess.Popen(cmd, start_new_session=True)
    arrival = Arrival(args.radius, args.hold)
    feed = None if args.no_auto else ros_feed(arrival, log)
    why = None
    try:
        log("Ctrl-C to stop" + ("" if feed is None else "; or it stops %g s after the dog is within %.1f m of "
                                                       "the planner's goal" % (args.hold, args.radius)))
        while rec.poll() is None:
            if feed is not None:
                feed[0](0.1)
            else:
                time.sleep(0.1)
            if arrival.due(time.monotonic()):
                why = "at the goal"
                break
    except Stop as e:
        why = "Ctrl-C" if str(e) == "SIGINT" else str(e)
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)          # the stop below is not to be interrupted
        if feed is not None:
            feed[1]()
    if why is None:
        log("the recorder exited by itself (%s)" % rec.returncode)
        code = rec.returncode or 1
    else:
        log("stopping (%s)" % why)
        code = stop_recorder(rec, log)
        code = 0 if code in (0, -signal.SIGINT, 130) else code
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid and os.path.isdir(out):
        chown_tree(out, int(uid), int(gid))           # the nvidia user can copy and delete it
    if os.path.isfile(os.path.join(out, "metadata.yaml")):
        log("done: %s" % out)
        subprocess.run(["ros2", "bag", "info", out])
    elif os.path.isdir(out):
        log("no metadata.yaml in %s: ros2 bag reindex %s -s sqlite3" % (out, out))
        code = code or 1
    else:
        log("nothing recorded")
        code = code or 1
    log("copy to the server: /public/zjst/robot/cp_data/; then bag_to_mcap.py (DOG_FIELD_GUIDE.md section 9)")
    return code


if __name__ == "__main__":
    sys.exit(main())
