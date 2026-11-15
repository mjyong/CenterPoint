"""tools/deploy/nav_cw_record.{sh,py}: one command to record on dog 8; it stops by itself hold seconds after the dog
has come within radius of the planner's goal, or on Ctrl-C. Run against a fake ros2 (the recorder: a bag directory,
metadata.yaml on SIGINT) and a fake rclpy that delivers the goal and the odometry on a timeline."""
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time

import pytest

from .conftest import ROOT

SH = os.path.join(ROOT, "tools", "deploy", "nav_cw_record.sh")
PY = os.path.join(ROOT, "tools", "deploy", "nav_cw_record.py")
GUIDE = os.path.join(ROOT, "docs", "DOG_FIELD_GUIDE.md")

FAKE_ROS2 = r'''#!/usr/bin/env python3
import os, signal, sys, time
log = os.environ["FAKE_LOG"]
def note(line):
    with open(log, "a") as f:
        f.write(line + "\n")
a = sys.argv[1:]
note("ros2 " + " ".join(a))
if a[:2] == ["bag", "info"]:
    print("Files: x_0.db3")
    sys.exit(0)
out = a[a.index("-o") + 1]
os.makedirs(out)
open(os.path.join(out, os.path.basename(out) + "_0.db3"), "w").close()
if os.environ.get("FAKE_RECORD_DIES"):
    sys.exit(3)
stopping = []
def stop(sig, frame):
    note("SIGINT")
    stopping.append(time.monotonic())
signal.signal(signal.SIGINT, stop)
while not stopping or time.monotonic() - stopping[0] < 0.3:   # a second SIGINT in this time is noted too
    time.sleep(0.02)
open(os.path.join(out, "metadata.yaml"), "w").write("rosbag2_bagfile_information: {}\n")
'''

FAKE_RCLPY = r'''import json, os, time
_t0 = [None]
_feed = []
def init():
    _t0[0] = time.monotonic()
    _feed[:] = sorted(json.loads(os.environ.get("FAKE_FEED", "[]")), key=lambda e: e[0])
def ok():
    return _t0[0] is not None
def shutdown():
    _t0[0] = None
class _Node:
    def __init__(self):
        self.subs = {}
    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs[topic] = (msg_type, cb)
    def destroy_node(self):
        pass
def create_node(name, **kw):
    assert kw == {"enable_rosout": False, "start_parameter_services": False}
    return _Node()
def spin_once(node, timeout_sec=None):
    time.sleep(timeout_sec or 0.0)
    now = time.monotonic() - _t0[0]
    while _feed and _feed[0][0] <= now:
        t, topic, *args = _feed.pop(0)
        if topic in node.subs:
            msg_type, cb = node.subs[topic]
            cb(msg_type.fake(*args))
'''

FAKE_MSG = r'''class _P:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0
class _Pose:
    def __init__(self, x, y):
        self.position = _P(x, y)
class {name}:
    @classmethod
    def fake(cls, x, y):
        m = cls()
        if {nested}:
            class _PP:
                pass
            m.pose = _PP()
            m.pose.pose = _Pose(x, y)
        else:
            m.pose = _Pose(x, y)
        return m
'''


FAKE_MC = r'''class _S:
    def __init__(self, motion, state):
        self.motion, self.state = motion, state
class State:
    @classmethod
    def fake(cls, mode, uuid, motions, fault=0):
        m = cls()
        m.mode, m.uuid, m.fault = mode, uuid, fault
        m.state = [_S(a, b) for a, b in motions]
        return m
class Motion:
    @classmethod
    def fake(cls, uuid, mode, motion, para):
        m = cls()
        m.uuid, m.mode, m.motion, m.para = uuid, mode, motion, para
        return m
'''


@pytest.fixture
def dog(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ros2").write_text(FAKE_ROS2)
    (bin_ / "ros2").chmod(0o755)
    py = tmp_path / "fakepy"
    (py / "rclpy").mkdir(parents=True)
    (py / "rclpy" / "__init__.py").write_text(FAKE_RCLPY)
    (py / "rclpy" / "qos.py").write_text("qos_profile_sensor_data = 'best effort'\n")
    for pkg, name, nested in (("geometry_msgs", "PoseStamped", False), ("nav_msgs", "Odometry", True),
                              ("visualization_msgs", "Marker", False)):
        (py / pkg).mkdir()
        (py / pkg / "__init__.py").write_text("")
        (py / pkg / "msg.py").write_text(FAKE_MSG.format(name=name, nested=nested))
    (py / "mc_interface").mkdir()
    (py / "mc_interface" / "__init__.py").write_text("")
    (py / "mc_interface" / "msg.py").write_text(FAKE_MC)
    prefix = tmp_path / "prefix"
    for p in ("slam", "nav_perception", "planning"):
        (prefix / "install" / p).mkdir(parents=True)
        (prefix / "install" / p / "setup.bash").write_text('echo "sourced %s" >> "$FAKE_LOG"\n' % p)
    setup = tmp_path / "setup.bash"
    setup.write_text('export PATH="%s:$PATH"; export PYTHONPATH="%s"\n' % (bin_, py))
    bags = tmp_path / "bags"
    log = tmp_path / "log.txt"
    log.write_text("")
    (bin_ / "journalctl").write_text('#!/bin/bash\necho "journal $*"\n')
    (bin_ / "journalctl").chmod(0o755)

    def env(**extra):
        e = dict(os.environ, NAV_CW_ROS_SETUP=str(setup), NAV_CW_PREFIX=str(prefix), NAV_CW_RECORD_ANY_USER="1",
                 NAV_CW_BAG_DIR=str(bags), FAKE_LOG=str(log), PYTHONUNBUFFERED="1",
                 NAV_CW_JOURNALCTL=str(bin_ / "journalctl"))
        for k in ("ROS_DOMAIN_ID", "SUDO_UID", "SUDO_GID", "PYTHONPATH"):
            e.pop(k, None)
        e.update({k: (json.dumps(v) if k == "FAKE_FEED" else str(v)) for k, v in extra.items()})
        return e

    def start(*args, stdin=subprocess.DEVNULL, **extra):
        # its own process group, as the terminal's foreground group: killpg(SIGINT) is a Ctrl-C
        return subprocess.Popen(["bash", SH, *args], env=env(**extra), stdout=subprocess.PIPE, stdin=stdin,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def wait_recording(p, timeout=10.0):
        t0 = time.monotonic()
        while "ros2 bag record" not in log.read_text():
            assert p.poll() is None and time.monotonic() - t0 < timeout, log.read_text()
            time.sleep(0.05)

    start.env, start.log, start.bags, start.prefix, start.wait_recording = env, log, bags, prefix, wait_recording
    return start


def test_the_topics_are_the_field_guides(dog):
    spec = importlib.util.spec_from_file_location("nav_cw_record", PY)
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    text = open(GUIDE, encoding="utf-8").read()
    sec = text[text.index("## 9."):text.index("## 10.")]
    block = sec[sec.index("ros2 bag record"):sec.index("```", sec.index("ros2 bag record"))]
    assert re.findall(r"(/[\w/]+)", block.split("\n", 1)[1]) == rec.TOPICS
    assert "/planning/debug/goal_pose" in rec.ADDED            # /goal_pose is volatile (T57: 0 messages)


def test_ctrl_c_stops_the_recorder_once_and_it_writes_metadata(dog):
    p = dog()
    dog.wait_recording(p)
    time.sleep(0.5)
    os.killpg(p.pid, signal.SIGINT)                           # Ctrl-C: the terminal's whole foreground group
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out
    log = dog.log.read_text()
    assert log.count("SIGINT") == 1, log                      # the recorder is not in that group: one SIGINT
    assert log.index("sourced slam") < log.index("sourced nav_perception") < log.index("sourced planning")
    rec = next(l for l in log.splitlines() if l.startswith("ros2 bag record"))
    assert "-s sqlite3" in rec and "--max-cache-size 268435456" in rec and "/planning/debug/goal_pose" in rec
    bag = rec.split(" -o ")[1].split()[0]
    assert re.search(r"/\d{8}_cw_1$", bag) and os.path.isfile(os.path.join(bag, "metadata.yaml"))
    assert "stopping (Ctrl-C)" in out and "ros2 bag info " + bag in log and "done: " + bag in out


def test_it_stops_hold_seconds_after_the_dog_is_within_the_radius(dog):
    feed = [[0.3, "/planning/debug/goal_pose", 10.0, 0.0]] + \
           [[0.4 + 0.1 * k, "/lightning/odom", min(9.0, 6.0 + 0.5 * k), 0.0] for k in range(60)]
    t0 = time.monotonic()
    p = dog("--hold", "1", FAKE_FEED=feed)
    out, _ = p.communicate(timeout=30)
    took = time.monotonic() - t0
    assert p.returncode == 0, out
    assert "goal (10.00, 0.00)" in out and "goal 4.0 m away" in out
    assert "within 1.5 m of the goal (1.50 m): stopping in 1 s" in out and "stopping (at the goal)" in out
    assert dog.log.read_text().count("SIGINT") == 1
    assert 1.4 < took < 15, took                              # inside at 0.9 s, stops 1 s later


def test_the_last_tests_goal_under_the_dog_does_not_stop_it(dog):
    feed = [[0.1 * k, t, x, 0.0] for k in range(1, 40) for t, x in (("/planning/debug/goal_pose", 0.5),
                                                                       ("/lightning/odom", 0.0))]
    p = dog("--hold", "0.5", FAKE_FEED=feed)
    dog.wait_recording(p)
    time.sleep(3.0)
    assert p.poll() is None                                   # never outside the radius: not armed
    os.killpg(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0 and "stopping (Ctrl-C)" in out and "stopping in" not in out, out


def test_a_new_goal_during_the_recording_arms_it(dog):
    # the old goal under the dog, then /goal_pose 8 m away; the dog drives there
    feed = [[0.2, "/planning/debug/goal_pose", 0.5, 0.0], [0.3, "/lightning/odom", 0.0, 0.0],
            [0.6, "/goal_pose", 8.0, 0.0]] + [[0.7 + 0.1 * k, "/lightning/odom", min(7.5, 0.5 * k), 0.0]
                                                for k in range(40)]
    p = dog("--hold", "0.5", FAKE_FEED=feed)
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 0 and "goal (8.00, 0.00)" in out and "stopping (at the goal)" in out, out


def test_a_recorder_that_dies_is_reported(dog):
    p = dog(FAKE_RECORD_DIES="1")
    out, _ = p.communicate(timeout=20)
    assert p.returncode != 0 and "the recorder exited by itself (3)" in out and "reindex" in out, out


def test_dry_run_and_the_rehearsal_domain(dog):
    (dog.prefix / "nav_cw.env").write_text('NAV_CW_SHADOW="on"\nROS_DOMAIN_ID="42"\n')
    r = subprocess.run(["bash", SH, "--dry-run", "-o", "/x/y", "/extra/topic"], env=dog.env(), capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "domain 42" in r.stdout and "ros2 bag record -s sqlite3" in r.stdout and " -o /x/y " in r.stdout
    assert r.stdout.rstrip().endswith("/extra/topic") and "ros2 bag record" not in dog.log.read_text()


def test_an_existing_bag_is_refused(dog):
    (dog.bags / "b").mkdir(parents=True)
    r = subprocess.run(["bash", SH, "-o", str(dog.bags / "b")], env=dog.env(), capture_output=True, text=True)
    assert r.returncode == 2 and "exists" in r.stdout


def test_arrival_and_numbering(tmp_path):
    spec = importlib.util.spec_from_file_location("nav_cw_record", PY)
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)
    a = rec.Arrival(radius=1.5, hold=5.0)
    assert a.set_pose((0, 0), 0.0) is None and not a.due(100.0)           # no goal
    assert a.set_goal((10, 0)) and a.set_goal((10.05, 0)) is None         # the same goal again
    a.set_pose((0, 0), 1.0)
    a.set_pose((9, 0), 2.0)
    assert not a.due(6.9) and a.due(7.0)
    assert "left" in a.set_pose((5, 0), 3.0) and not a.due(10.0)          # leaving cancels
    a.set_pose((9.5, 0), 4.0)
    assert a.due(9.0)
    for n in ("20261115_cw_1", "20261115_cw_3", "20261114_cw_9", "other"):
        (tmp_path / n).mkdir()
    assert rec.default_output(str(tmp_path), "20261115").endswith("20261115_cw_4")
    assert rec.default_output(str(tmp_path / "none"), "20261116").endswith("20261116_cw_1")


def test_motion_control_changes_are_printed_once_each(dog):
    # T57: the dog stood in mode 3 while the planner sent mode 2 walks
    feed = [[0.2 + 0.05 * k, "/mc/state", 3, "", [[20, 3]]] for k in range(10)] + \
           [[0.3 + 0.05 * k, "/mc/motion_control", "planning", 2, 20, [0.0, 0.0, 0.0]] for k in range(4)] + \
           [[0.5 + 0.05 * k, "/mc/motion_control", "planning", 2, 20, [0.27, 0.09, 0.0]] for k in range(6)] + \
           [[0.9, "/mc/state", 2, "planning", [[20, 2]]]]
    p = dog("--no-auto", FAKE_FEED=feed)
    dog.wait_recording(p)
    time.sleep(1.5)
    os.killpg(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out
    assert out.count("/mc/state: mode 3 lower body (IMC), uuid '', 20:DONE") == 1
    assert out.count("/mc/motion_control: uuid 'planning', mode 2 whole body (WBMC), motion 20") == 2   # still, moving
    assert "para [0.27, 0.09, 0.00]" in out and "/mc/state: mode 2 whole body (WBMC), uuid 'planning', 20:DOING" in out


def test_no_record_only_watches_and_hidden_adds_action_topics(dog):
    p = dog("--no-record", FAKE_FEED=[[0.2, "/mc/state", 3, "", [[20, 3]]]])
    time.sleep(1.5)
    os.killpg(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0 and "watching, domain 0" in out and "/mc/state: mode 3" in out, out
    assert "ros2 bag record" not in dog.log.read_text()
    r = subprocess.run(["bash", SH, "--dry-run", "--hidden", "-o", "/x/z"], env=dog.env(), capture_output=True,
                       text=True)
    assert "--include-hidden-topics /robot/topic/rdap/top_point_cloud2" in r.stdout, r.stdout


def test_notes_journals_and_the_replay_goal_go_beside_the_bag(dog):
    feed = [[0.2, "/planning/debug/goal_pose", 12.5, -3.25]]
    p = dog("--tag", "cw13_watch", stdin=subprocess.PIPE, FAKE_FEED=feed)
    dog.wait_recording(p)
    p.stdin.write("car from the left, the go was early\n")
    p.stdin.write("\n")
    p.stdin.flush()
    time.sleep(1.0)
    os.killpg(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out
    bag = next(d for d in dog.bags.iterdir())
    assert re.match(r"^\d{8}_cw13_watch_1$", bag.name)
    rec_log = (bag / "record_log.txt").read_text()
    assert "note: car from the left, the go was early" in rec_log and "note: mark" in rec_log
    assert "replay goal: 12.500 -3.250" in rec_log and "stopping (Ctrl-C)" in rec_log
    j = (bag / "journal_cw-planning.txt").read_text()
    assert j.startswith("journal -u cw-planning --since ") and "--until" in j and "short-iso-precise" in j
    assert {f.name for f in bag.iterdir()} >= {"journal_cw-slam.txt", "journal_cw-nav-perception.txt",
                                               "metadata.yaml"}
    assert subprocess.run(["bash", SH, "--tag", "a/b"], env=dog.env(), capture_output=True).returncode == 2
