"""One command to record a bag on dog 8 (started by nav_cw_record.sh, which sets up ROS as root).

Records DOG_FIELD_GUIDE.md section 9's topics, plus the planner's goal marker (/goal_pose is volatile: a goal sent
before the recording started is not in the bag, T57) and the robot's motion state, as sqlite3 (the dog's ros2 bag has
no mcap plugin, T61). It stops by itself once the planner has a goal and the dog, walking or remote-controlled, has
come within --radius (1.5 m) of it: --hold (5 s) later. Otherwise Ctrl-C stops it. Either way the recorder gets one
SIGINT and writes metadata.yaml; nothing opens the database while it is written (T57: "database is locked").

The goal is a /goal_pose sent during the recording, else /planning/debug/goal_pose (the planner republishes the last
one every cycle); the dog is /lightning/odom (map). Foxglove publishes the goal in its display frame, base_link in
planning_live.json: such a goal is placed in map by the dog's odometry pose when it arrives, as the planner does with
its TF (T66: the goals were taken as map numbers, "317 m away", and the replay goals were wrong). A goal the dog is
already within the radius of (the last test's) does not stop it: the dog has to be outside first.

Cues (T66: the person walking the dog went on past align twice): each change of the planner's crosswalk state
(/planning/debug/plotjuggler/string crosswalk_mode) is a line saying what to do, with the terminal bell: approach
slow down, align stop and turn (which way, how far), wait stand, go / cross walk, cross_wait stop. The crosswalk is
named by the site map's numbers (planner index 0..4 = 6, 13, 22, 32, 38 on ad68517c; NAV_CW_CROSSWALK_NAMES).

It also prints every change of the motion control's state (/mc/state: mode, uuid, each motion's state) and of who
commands it (/mc/motion_control: uuid, mode, motion): T57's dog stood in mode 3 (lower body) while the planner sent
mode 2 (whole body) walks. --no-record only watches.

Notes: a line typed in the terminal (Enter) is logged with the time ("note: car from the left, the go was early"; an
empty line is "mark"). When the recording stops, the bag directory also gets record_log.txt (everything printed, the
notes, the planner's last goal as "replay goal: x y" for replay_planning_check.sh) and journal_<unit>.txt (our units'
journals over the recording: the planner's "Crosswalk yield:" lines to compare a replay with).

    sudo bash /opt/seres/nav_cw/nav_cw_record.sh                       # /home/nvidia/bags/<date>_cw_<n>
    sudo bash /opt/seres/nav_cw/nav_cw_record.sh --tag cw13_watch      # /home/nvidia/bags/<date>_cw13_watch_<n>
    sudo bash /opt/seres/nav_cw/nav_cw_record.sh -o /home/nvidia/bags/x --radius 2 --hold 5 /more/topic ...
    sudo bash /opt/seres/nav_cw/nav_cw_record.sh --no-record      # only the motion-control lines (and the goal)
"""
import argparse
import datetime
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

# DOG_FIELD_GUIDE.md section 9 (tests/dog/test_nav_cw_record.py checks the two agree), then what this adds
TOPICS = [
    "/robot/topic/rdap/top_point_cloud2",
    # four fisheyes for reviewing lidar_3d_detection labels (SENSORS.md §7); H.264, not the raw images
    "/robot/topic/rdap/front_sensor_camera", "/robot/topic/rdap/left_sensor_camera",
    "/robot/topic/rdap/right_sensor_camera", "/robot/topic/rdap/rear_sensor_camera",
    "/lidar_packets_loss",
    "/rte/rdap/rtk/inspvax", "/rte/rdap/rtk/imus/data_raw",
    "/lightning/odom", "/nav/location",
    "/map/occupancy_grid", "/perception/lidar_3d/objects", "/goal_pose",
    "/cmd_vel", "/mc/motion_control", "/nav_cw/shadow/cmd_vel", "/nav_cw/shadow/mc/motion_control",
    "/planning/debug/crosswalk", "/planning/debug/path_tracking", "/planning/debug/plotjuggler/string",
    "/planning/debug/collision_checker", "/planning/debug/diagnostic",
    "/planning/debug/cost_map_production/occmap", "/planning/debug/global_map_manager/occmap",
    "/tf", "/tf_static",
]
ADDED = ["/planning/debug/goal_pose", "/mc/state"]
GOAL_MARKER, GOAL_POSE, ODOM = "/planning/debug/goal_pose", "/goal_pose", "/lightning/odom"
CACHE = 256 * 1024 * 1024        # [B] --max-cache-size (T35: the default dropped point clouds)
MODES = {0: "unknown", 1: "debug", 2: "whole body (WBMC)", 3: "lower body (IMC)", 4: "teleop"}   # mc_interface/Mode
STATES = {1: "REJECTED", 2: "DOING", 3: "DONE", 4: "ABORTED"}                                 # MotionState


def describe_state(uuid, mode, motions, fault):
    """One line for an mc_interface/State: mode, the commanding uuid, (motion, state) pairs."""
    return "/mc/state: mode %d %s, uuid %r, %s%s" % (
        mode, MODES.get(mode, "?"), uuid,
        " ".join("%d:%s" % (m, STATES.get(s, s)) for m, s in motions) or "no motions",
        ", fault 0x%x" % fault if fault else "")


def describe_command(uuid, mode, motion, para):
    return "/mc/motion_control: uuid %r, mode %d %s, motion %d, para [%s]" % (
        uuid, mode, MODES.get(mode, "?"), motion, ", ".join("%.2f" % v for v in para))


CROSSWALK_NAMES = os.environ.get("NAV_CW_CROSSWALK_NAMES", "6,13,22,32,38").split(",")   # ad68517c, planner order
CUES = {1: "approach：慢下来，沿路线走到停止点",
        2: "停！align：原地{turn}转到面朝对面（差 {deg:.0f}°）",
        3: "wait：站着别动",
        4: "go：沿路线走过去",
        5: "cross：在人行横道上，接着走",
        6: "停！cross_wait：前方可能来车，原地站着"}


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def to_map(xy, pose):
    """A point in the dog's frame (base_link) in map, by the dog's pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * xy[0] - s * xy[1], y + s * xy[0] + c * xy[1])


def parse_debug(text):
    """The planner's /planning/debug/plotjuggler/string: "key: value" lines."""
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def cue(values):
    """-> (mode, line) for the crosswalk state in a debug string; mode None without one."""
    if "crosswalk_mode" not in values:
        return None, None
    mode = int(values["crosswalk_mode"])
    if mode == 0:
        return 0, "none：前面 40 m 没有人行横道"
    k = int(values.get("crosswalk_index", -1))
    name = CROSSWALK_NAMES[k] if 0 <= k < len(CROSSWALK_NAMES) else str(k)
    err = values.get("crosswalk_heading_error", 0.0)
    text = CUES.get(mode, "mode %d" % mode).format(turn="往左" if err > 0 else "往右", deg=abs(math.degrees(err)))
    if mode == 3 and values.get("crosswalk_waiting_for", 0) > 0:
        text += "（等 %d 辆）" % int(values["crosswalk_waiting_for"])
    return mode, "== 图上 %s 号 %s" % (name, text)


class Changes:
    """Report a value only when it changes (the topics run at 10-100 Hz)."""

    def __init__(self):
        self.last = {}

    def __call__(self, key, value):
        if self.last.get(key) == value:
            return False
        self.last[key] = value
        return True


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


def default_output(directory, today=None, tag="cw"):
    """<directory>/<YYYYMMDD>_<tag>_<n>, n one more than the largest there for that day and tag."""
    day = today or datetime.date.today().strftime("%Y%m%d")
    pat = re.compile(r"^%s_%s_(\d+)$" % (re.escape(day), re.escape(tag)))
    taken = [int(m.group(1)) for m in map(pat.match, os.listdir(directory) if os.path.isdir(directory) else [])
             if m]
    return os.path.join(directory, "%s_%s_%d" % (day, tag, max(taken, default=0) + 1))


class Notes:
    """Lines typed on stdin, as notes in the log; off when stdin is closed (or /dev/null)."""

    def __init__(self, stream, log):
        self.log, self.buf = log, b""
        try:
            self.fd = stream.fileno() if stream is not None and not stream.closed else None
        except (OSError, ValueError):
            self.fd = None

    def poll(self):
        """Read what is there from the descriptor itself (a buffered readline would keep a second line back)."""
        while self.fd is not None:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0)
                chunk = os.read(self.fd, 4096) if ready else None
            except (OSError, ValueError):
                self.fd = None
                return
            if chunk is None:
                return
            if not chunk:                     # EOF
                self.fd = None
                return
            self.buf += chunk
            *done, self.buf = self.buf.split(b"\n")
            for line in done:
                self.log("note: " + (line.decode(errors="replace").strip() or "mark"))


def save_journals(out, start, end, log):
    """journal_<unit>.txt in the bag directory: our units over [start, end] (wall clock)."""
    jc = shutil.which(os.environ.get("NAV_CW_JOURNALCTL", "journalctl"))
    if jc is None:
        log("no journalctl: no journals saved")
        return
    fmt = "%Y-%m-%d %H:%M:%S"
    since = time.strftime(fmt, time.localtime(start - 2))
    until = time.strftime(fmt, time.localtime(end + 2))
    for unit in os.environ.get("NAV_CW_RECORD_UNITS", "cw-planning cw-slam cw-nav-perception").split():
        path = os.path.join(out, "journal_%s.txt" % unit)
        with open(path, "w") as f:
            subprocess.run([jc, "-u", unit, "--since", since, "--until", until, "--no-pager", "-o", "short-iso-precise"],
                           stdout=f, stderr=subprocess.STDOUT)
    log("journals of our units saved beside the bag (journal_<unit>.txt)")


class Stop(Exception):
    pass


def _raise_stop(signum, frame):
    raise Stop(signal.Signals(signum).name)


def ros_feed(arrival, log, cues=True):
    """Subscriptions that feed ``arrival``, print the motion control's changes and the crosswalk cues;
    -> (spin(timeout), close()), or None without rclpy."""
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from visualization_msgs.msg import Marker
    except ImportError as e:
        log("no rclpy (%s): no auto stop, no motion-control lines; Ctrl-C to stop" % e)
        return None
    rclpy.init()
    # subscriptions only: no rosout, no parameter services on the robot's domain
    node = rclpy.create_node("nav_cw_record", enable_rosout=False, start_parameter_services=False)

    latest = {}

    def set_goal(xy):
        line = arrival.set_goal(xy)
        if line:
            log(line)

    def goal_marker(msg):
        if "sent" not in latest:              # a goal sent during the recording is placed from /goal_pose
            p = msg.pose.position
            set_goal((p.x, p.y))

    def goal_pose(msg):
        p = msg.pose.position
        frame = (msg.header.frame_id or "").strip().lstrip("/")
        xy = (p.x, p.y)
        if frame not in ("", "map"):
            if frame not in ("base_link", "body"):
                log("goal in %s: not placed (only base_link, by the odometry); the planner's marker is used" % frame)
                return
            latest["sent"] = True
            if "pose" not in latest:
                latest["pending_base"] = xy
                return
            xy = to_map(xy, latest["pose"])
            log("goal sent in base_link (%.2f, %.2f): (%.2f, %.2f) in map" % (p.x, p.y, xy[0], xy[1]))
        latest["sent"] = True
        latest.pop("pending_base", None)
        set_goal(xy)

    def odom(msg):
        p = msg.pose.pose.position
        latest["pose"] = (p.x, p.y, yaw_of(msg.pose.pose.orientation))
        if "pending_base" in latest:
            bx, by = latest.pop("pending_base")
            mx, my = to_map((bx, by), latest["pose"])
            log("goal sent in base_link (%.2f, %.2f): (%.2f, %.2f) in map" % (bx, by, mx, my))
            set_goal((mx, my))
        line = arrival.set_pose((p.x, p.y), time.monotonic())
        if line:
            log(line)

    # best effort takes both reliable and best-effort publishers
    node.create_subscription(Marker, GOAL_MARKER, goal_marker, qos_profile_sensor_data)
    node.create_subscription(PoseStamped, GOAL_POSE, goal_pose, qos_profile_sensor_data)
    node.create_subscription(Odometry, ODOM, odom, qos_profile_sensor_data)
    if cues:
        from std_msgs.msg import String

        def debug(msg):
            mode, line = cue(parse_debug(msg.data))
            if mode is not None and mode != latest.get("mode"):
                latest["mode"] = mode
                print("\a", end="", flush=True)      # the terminal bell: look at the screen
                log(line)

        node.create_subscription(String, "/planning/debug/plotjuggler/string", debug, qos_profile_sensor_data)
    try:
        from mc_interface.msg import Motion, State       # the planning prefix's (nav_cw_record.sh sources it last)
    except ImportError as e:
        log("no mc_interface (%s): no motion-control lines" % e)
    else:
        changed = Changes()

        def state(msg):
            motions = tuple((int(s.motion), int(s.state)) for s in msg.state)
            v = (msg.uuid, int(msg.mode), motions, int(msg.fault))
            if changed("state", v):
                log(describe_state(*v))

        def command(msg):
            para = [float(x) for x in msg.para]
            moving = any(abs(x) > 1e-3 for x in para[:3])
            if changed(("cmd", msg.uuid), (int(msg.mode), int(msg.motion), moving)):
                log(describe_command(msg.uuid, int(msg.mode), int(msg.motion), para))

        node.create_subscription(State, "/mc/state", state, qos_profile_sensor_data)
        node.create_subscription(Motion, "/mc/motion_control", command, qos_profile_sensor_data)

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


def watch_only(log, cues=True):
    """--no-record: the motion-control lines, the goal and the crosswalk cues until Ctrl-C."""
    feed = ros_feed(Arrival(), log, cues)
    if feed is None:
        return 1
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _raise_stop)
    log("watching, domain %s; Ctrl-C to stop" % os.environ.get("ROS_DOMAIN_ID", "0"))
    notes = Notes(sys.stdin, log)
    try:
        while True:
            feed[0](0.1)
            notes.poll()
    except Stop:
        pass
    finally:
        feed[1]()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("topics", nargs="*", help="more topics to record")
    ap.add_argument("-o", "--output", help="the bag directory (must not exist; default <dir>/<date>_cw_<n>)")
    ap.add_argument("--dir", default=os.environ.get("NAV_CW_BAG_DIR", "/home/nvidia/bags"))
    ap.add_argument("--tag", default="cw", help="the default name is <date>_<tag>_<n> (e.g. cw13_watch)")
    ap.add_argument("--radius", type=float, default=1.5, help="[m] the dog this close to the goal ...")
    ap.add_argument("--hold", type=float, default=5.0, help="[s] ... this long before the recording stops")
    ap.add_argument("--no-auto", action="store_true", help="only Ctrl-C stops it")
    ap.add_argument("--no-record", action="store_true", help="only print the motion-control changes (and the goal)")
    ap.add_argument("--hidden", action="store_true", help="also hidden topics (an action's _action/status, feedback)")
    ap.add_argument("--no-cues", action="store_true", help="no crosswalk cue lines and bell")
    ap.add_argument("--dry-run", action="store_true", help="print the record command")
    args = ap.parse_args(argv)

    lines = []

    def log(line):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), line)
        lines.append(line)
        print(line, flush=True)

    if args.no_record:
        return watch_only(log, cues=not args.no_cues)
    if not re.match(r"^[\w.-]+$", args.tag):
        log("--tag: letters, digits, _ . - only")
        return 2
    out = args.output or default_output(args.dir, tag=args.tag)
    if os.path.exists(out):
        log("%s exists; ros2 bag record would refuse it" % out)
        return 2
    topics = TOPICS + ADDED + [t for t in args.topics if t not in TOPICS + ADDED]
    cmd = ["ros2", "bag", "record", "-s", "sqlite3", "--max-cache-size", str(CACHE), "-o", out]
    cmd += (["--include-hidden-topics"] if args.hidden else []) + topics
    if args.dry_run:
        print("# domain %s" % os.environ.get("ROS_DOMAIN_ID", "0"))
        print(" ".join(cmd))
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _raise_stop)
    log("recording %s, domain %s: %d topics" % (out, os.environ.get("ROS_DOMAIN_ID", "0"), len(topics)))
    started = time.time()
    rec = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL)   # the notes are ours
    arrival = Arrival(args.radius, args.hold)
    feed = ros_feed(arrival, log, cues=not args.no_cues)
    auto = feed is not None and not args.no_auto
    notes = Notes(sys.stdin, log)
    why = None
    try:
        log("Ctrl-C to stop" + ("; or it stops %g s after the dog is within %.1f m of the planner's goal" % (
            args.hold, args.radius) if auto else ""))
        if notes.fd is not None:
            log("type a note and Enter to log it with the time (Enter alone: mark)")
        while rec.poll() is None:
            if feed is not None:
                feed[0](0.1)
            else:
                time.sleep(0.1)
            notes.poll()
            if auto and arrival.due(time.monotonic()):
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
    ended = time.time()
    if os.path.isdir(out):
        save_journals(out, started, ended, log)
        log("recorded %.0f s, %s to %s" % (ended - started, time.strftime("%H:%M:%S", time.localtime(started)),
                                            time.strftime("%H:%M:%S", time.localtime(ended))))
        if arrival.goal is not None:
            log("replay goal: %.3f %.3f" % arrival.goal)      # replay_planning_check.sh <name> <x> <y>
        with open(os.path.join(out, "record_log.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
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
