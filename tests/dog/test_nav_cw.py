"""tools/deploy/nav_cw.sh: switching dog 8 between the robot team's navigation services and ours, against a fake
systemctl (unit states in files)."""
import os
import subprocess

import pytest

from .conftest import ROOT

SCRIPT = os.path.join(ROOT, "tools", "deploy", "nav_cw.sh")
ORIG = ["lidar-3d-detection.service", "nav_perception.service", "slam.service", "planning.service"]
NEW = ["cw-nav-perception.service", "cw-slam.service", "cw-planning.service"]

FAKE = r"""#!/bin/bash
# fake systemctl: $STATE/<unit> holds active/inactive, $STATE/<unit>.restarts the restart count; a unit listed in
# $STATE/broken starts but is dead (or restarting) by the time it is checked
s="$STATE"; cmd="$1"; shift
echo "$cmd $*" >> "$s/log"
case "$cmd" in
  is-active) q=0; [[ "$1" == --quiet ]] && { q=1; shift; }; st=$(cat "$s/$1" 2>/dev/null || echo inactive)
             (( q )) || echo "$st"; [[ "$st" == active ]] && exit 0; exit 3 ;;
  start) if grep -qx "$1" "$s/broken" 2>/dev/null; then
           echo failed > "$s/$1"; echo $(( $(cat "$s/$1.restarts" 2>/dev/null || echo 0) + 1 )) > "$s/$1.restarts"
         else echo active > "$s/$1"; fi ;;
  stop) echo inactive > "$s/$1" ;;
  show) cat "$s/${@: -1}.restarts" 2>/dev/null || echo 0 ;;
  cat) [[ -e "$s/$1.unit" ]] || exit 1 ;;
esac
"""

# fake ros2: `topic info -v <topic>` answers as our planner and localization would, from the installed
# communication_conf (the repository's names without one) and nav_cw.env; FAKE_OLD_PLANNER: a planner that ignores
# the config's output names. Every call is logged to $STATE/log.
GRAPH = r"""#!/bin/bash
echo "ros2 $*" >> "$STATE/log"
echo "domain ${ROS_DOMAIN_ID:-unset}" >> "$STATE/domains"
[[ "$1 $2" == "topic info" ]] || exit 0
topic="${@: -1}"
conf=$(find "$NAV_CW_ROOT/install" -name communication_conf.pb.txt 2>/dev/null | head -1)
name() {
  if [[ -n "$conf" ]]; then sed -n -E "s/^[[:space:]]*$1[[:space:]]*\{[[:space:]]*name:[[:space:]]*\"([^\"]*)\".*/\1/p" "$conf"
  else echo "$2"; fi
}
up() { "$SYSTEMCTL" is-active --quiet "$1"; }
ep() { printf 'Node name: %s\nNode namespace: /\nEndpoint type: %s\n\n' "$1" "$2"; }
echo "Type: x"
if up cw-planning.service; then
  pubs="$(name pub_cmd_vel /cmd_vel) $(name pub_motion /mc/motion_control)"
  [[ -n "${FAKE_OLD_PLANNER:-}" ]] && pubs="/cmd_vel /mc/motion_control"
  for p in $pubs; do if [[ "$p" == "$topic" ]]; then ep planning PUBLISHER; fi; done
  if [[ "$(name sub_map_switch /map/switch)" == "$topic" ]]; then ep planning SUBSCRIPTION; fi
fi
if up cw-slam.service; then
  sw=$(sed -n 's/.*map_switch_topic:=\([^" ]*\).*/\1/p' "$NAV_CW_ROOT/nav_cw.env" 2>/dev/null)
  if [[ "${sw:-/map/switch}" == "$topic" ]]; then ep mode_switch_node SUBSCRIPTION; fi
fi
exit 0
"""


COMM = """confs{
    ros_bridge_conf{
        sub_map_switch              { name: "/map/switch" }
        sub_goal_pose               { name: "/goal_pose"           queue_size: 10 }
        pub_cmd_vel                 { name: "/cmd_vel"             queue_size: 10 }
        pub_motion                  { name: "/mc/motion_control"   queue_size: 10 }
        pub_flag: true
    }
}
"""


MISSION = ("teb_planner_conf{\n  dynamic_constraints{\n    max_vx: 3.0\n    reference_velocity: 2.0\n  }\n}\n"
           "mppi_planner_conf{\n  dynamic_constraints{\n    max_vx: 3.0\n    max_vy: 0.5\n  }\n"
           "  reference_velocity_critic{\n    enabled: true\n    reference_velocity: 1.5\n  }\n}\n")
CONF_DIR = "install/planning/planning/lib/planning/robot_factory/d20/001/planning_conf"


def _install(nav_cw):
    """the planning configs of an install (what a reinstall puts back)"""
    conf = nav_cw / CONF_DIR
    conf.mkdir(parents=True, exist_ok=True)
    (conf / "mission_conf.pb.txt").write_text(MISSION)
    (conf / "communication_conf.pb.txt").write_text(COMM)
    for f in conf.glob("*.repo"):
        f.unlink()
    return conf


@pytest.fixture
def dog(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    fake = tmp_path / "systemctl"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    graph = tmp_path / "ros2"
    graph.write_text(GRAPH)
    graph.chmod(0o755)
    for u in ORIG:
        (state / u).write_text("active")
    for u in NEW:
        (state / (u + ".unit")).write_text("")
    _install(tmp_path / "nav_cw")

    def run(*args, broken=()):
        (state / "broken").write_text("".join(u + "\n" for u in broken))
        env = dict(os.environ, STATE=str(state), SYSTEMCTL=str(fake), JOURNALCTL="true", NAV_CW_WAIT="0",
                   ROS2=str(graph), NAV_CW_VERIFY_WAIT="0", NAV_CW_ROOT=str(tmp_path / "nav_cw"))
        return subprocess.run(["bash", SCRIPT, *args], env=env, capture_output=True, text=True)

    def states():
        return {u: (state / u).read_text().strip() if (state / u).exists() else "inactive" for u in ORIG + NEW}

    run.states = states
    run.log = lambda: (state / "log").read_text().splitlines()
    return run


def test_status_of_the_robot_teams(dog):
    r = dog("status")
    assert r.returncode == 0 and "running: theirs" in r.stdout


def test_use_new_then_back(dog):
    r = dog("use-new")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "running: ours" in r.stdout
    assert all(dog.states()[u] == "active" for u in NEW) and all(dog.states()[u] == "inactive" for u in ORIG)
    log = [l for l in dog.log() if l.startswith(("stop", "start"))]
    # planning stops first and starts last
    assert log[:4] == ["stop planning.service", "stop slam.service", "stop nav_perception.service",
                       "stop lidar-3d-detection.service"]          # the .deb's node too: ours runs it in the perception
    assert log[4:] == ["start cw-nav-perception.service", "start cw-slam.service", "start cw-planning.service"]
    r = dog("use-original")
    assert r.returncode == 0 and "running: theirs" in r.stdout
    assert all(dog.states()[u] == "active" for u in ORIG) and all(dog.states()[u] == "inactive" for u in NEW)


def test_use_new_puts_theirs_back_when_ours_does_not_come_up(dog):
    r = dog("use-new", broken=["cw-planning.service"])
    assert r.returncode == 1
    assert "cw-planning.service is not active" in r.stdout and "running: theirs" in r.stdout
    assert all(dog.states()[u] == "active" for u in ORIG) and all(dog.states()[u] == "inactive" for u in NEW)


def test_use_new_without_our_units_changes_nothing(dog, tmp_path):
    os.remove(tmp_path / "state" / "cw-slam.service.unit")
    r = dog("use-new")
    assert r.returncode == 1 and "no cw-slam.service" in r.stdout
    assert not [l for l in dog.log() if l.startswith(("stop", "start"))]


def test_configure_points_the_configs_at_the_map(tmp_path):
    prefix = tmp_path / "nav_cw"
    loc = prefix / "slam/lightning/share/lightning/config/default_d20.yaml"
    plan = prefix / "planning/lib/planning/robot_factory/d20/001/planning_conf/preprocess_conf.pb.txt"
    vin = prefix / "planning/lib/planning/robot_factory/default_robot_vin"
    for f in (loc, plan, vin):
        f.parent.mkdir(parents=True, exist_ok=True)
    loc.write_text("system:\n  map_path: /opt/seres/nav/data/ad68517c\n  with_ins: true\n")
    plan.write_text('confs{\n    global_map_manager_conf{\n        input_dir: "/../../data/astar_pgm/"\n'
                    '        startup_map_dir: "/old"\n    }\n}\n')
    vin.write_text("d30/001\n")
    m = tmp_path / "maps" / "ad68517c"
    m.mkdir(parents=True)
    (m / "global_map.yaml").write_text("floors: []\n")
    r = subprocess.run(["bash", SCRIPT, "configure", str(prefix), str(m) + "/"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "  map_path: %s\n" % m in loc.read_text() and "with_ins: true" in loc.read_text()
    p = plan.read_text()
    assert p.count("startup_map_dir") == 1 and 'startup_map_dir: "%s"' % m in p
    assert p.index("global_map_manager_conf") < p.index("startup_map_dir") < p.index("input_dir")
    assert vin.read_text().strip() == "d20/001"


def test_configure_points_localization_at_the_share_copy_the_launch_reads(tmp_path):
    prefix = tmp_path / "nav_cw"
    share = prefix / "install/slam/lightning/share/lightning/config/default_d20.yaml"
    lib = prefix / "install/slam/lightning/lib/lightning/config/default_d20.yaml"
    plan = prefix / "install/planning/planning/lib/planning/robot_factory/d20/001/planning_conf/preprocess_conf.pb.txt"
    vin = prefix / "install/planning/planning/lib/planning/robot_factory/default_robot_vin"
    for f in (share, lib, plan, vin):
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith(".yaml"):
            f.write_text("  map_path: /old\n")
    plan.write_text("confs{\n    global_map_manager_conf{\n    }\n}\n")
    vin.write_text("d30/001\n")
    m = tmp_path / "map"
    m.mkdir()
    (m / "global_map.yaml").write_text("floors: []\n")
    r = subprocess.run(["bash", SCRIPT, "configure", str(prefix), str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "map_path: %s" % m in share.read_text()
    assert "map_path: /old" in lib.read_text()


def test_configure_needs_a_site_map(tmp_path):
    r = subprocess.run(["bash", SCRIPT, "configure", str(tmp_path), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "no global_map.yaml" in r.stdout


def test_configure_edits_the_installed_copy_not_the_sources(tmp_path):
    prefix = tmp_path / "nav_cw"
    rel = "planning/robot_factory/d20/001/planning_conf/preprocess_conf.pb.txt"
    files = {}
    for where in ("src/Nav_D20", "install/planning/planning/lib"):
        f = prefix / where / rel
        f.parent.mkdir(parents=True)
        f.write_text("confs{\n    global_map_manager_conf{\n    }\n}\n")
        (f.parents[3] / "default_robot_vin").write_text("d30/001\n")
        files[where] = f
    loc = prefix / "install/slam/lightning/share/lightning/config/default_d20.yaml"
    loc.parent.mkdir(parents=True)
    loc.write_text("  map_path: /x\n")
    m = tmp_path / "map"
    m.mkdir()
    (m / "global_map.yaml").write_text("floors: []\n")
    r = subprocess.run(["bash", SCRIPT, "configure", str(prefix), str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "startup_map_dir" in files["install/planning/planning/lib"].read_text()
    assert "startup_map_dir" not in files["src/Nav_D20"].read_text()


def test_configure_turns_the_debug_topics_on(tmp_path):
    prefix = tmp_path / "nav_cw"
    conf = prefix / "install/planning/planning/lib/planning/robot_factory/d20/001/planning_conf"
    conf.mkdir(parents=True)
    (conf / "preprocess_conf.pb.txt").write_text("confs{\n    global_map_manager_conf{\n    }\n}\n")
    (conf / "communication_conf.pb.txt").write_text("    ros_debug_bridge_conf{\n        enable_global_map_pub: false\n"
                                                    "        pub_flag: false\n    }\n")
    (conf.parents[2] / "default_robot_vin").write_text("d20/001\n")   # robot_factory/
    loc = prefix / "install/slam/lightning/share/lightning/config/default_d20.yaml"
    loc.parent.mkdir(parents=True)
    loc.write_text("  map_path: /x\n")
    m = tmp_path / "map"
    m.mkdir()
    (m / "global_map.yaml").write_text("floors: []\n")
    r = subprocess.run(["bash", SCRIPT, "configure", str(prefix), str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    c = (conf / "communication_conf.pb.txt").read_text()
    assert "enable_global_map_pub: true" in c and "pub_flag: true" in c
    (conf / "communication_conf.pb.txt").write_text("        pub_flag: false\n")
    r = subprocess.run(["bash", SCRIPT, "configure", str(prefix), str(m)], capture_output=True, text=True,
                       env=dict(os.environ, NAV_CW_DEBUG="0"))
    assert "pub_flag: false" in (conf / "communication_conf.pb.txt").read_text()


def test_the_live_layout_shows_only_what_the_bridge_lets_through():
    """planning_live.json's topics all pass nav_cw_foxglove.sh's whitelist, and the goal tool publishes /goal_pose."""
    import json
    import re
    layout = json.load(open(os.path.join(ROOT, "Nav_D20", "planning", "script", "planning_live.json"), encoding="utf-8"))
    cfg = layout["configById"]
    topics = {t for t, v in cfg["3D!live"]["topics"].items() if v.get("visible", True)}   # hidden: the full level
    for k, v in cfg.items():
        for p in v.get("paths", []):
            topics.add(p["value"].split(".")[0])
        for key in ("topicPath", "topicToRender"):
            if key in v:
                topics.add(v[key].split(".")[0])
    assert cfg["3D!live"]["publish"]["poseTopic"] == "/goal_pose"
    topics.add("/goal_pose")
    script = open(os.path.join(ROOT, "tools", "deploy", "nav_cw_foxglove.sh"), encoding="utf-8").read()
    block = script[script.index("TOPICS=("):]
    patterns = re.findall(r"'([^']+)'", block[:block.index("\n)\n")])
    missing = [t for t in sorted(topics) if not any(re.search(p, t) for p in patterns)]
    assert not missing, missing
    assert "^/goal_pose$" in script.split("client_topic_whitelist")[1].split("\n")[0]   # only the goal
    assert "/planning/debug/crosswalk" in topics


def test_the_full_level_lets_the_hidden_layers_through():
    import json
    import re
    layout = json.load(open(os.path.join(ROOT, "Nav_D20", "planning", "script", "planning_live.json"), encoding="utf-8"))
    hidden = [t for t, v in layout["configById"]["3D!live"]["topics"].items() if not v.get("visible", True)]
    script = open(os.path.join(ROOT, "tools", "deploy", "nav_cw_foxglove.sh"), encoding="utf-8").read()
    basic = script[script.index("TOPICS=("):]
    full = script[script.index('== full ]]'):]
    patterns = re.findall(r"'([^']+)'", basic[:basic.index("\n)\n")]) + \
        re.findall(r"'([^']+)'", full[:full.index("\nfi")])
    assert hidden and not [t for t in hidden if not any(re.search(p, t) for p in patterns)]




@pytest.fixture
def root(tmp_path):
    """an installed tree (mission_conf, communication_conf), the fake systemctl (cw-planning running) and the fake
    ros2 graph"""
    conf = _install(tmp_path / "nav_cw")
    state = tmp_path / "state"
    state.mkdir()
    (state / "cw-planning.service").write_text("active")
    for name, text in (("systemctl", FAKE), ("ros2", GRAPH)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)

    def run(*args, old_planner=False):
        env = dict(os.environ, NAV_CW_ROOT=str(tmp_path / "nav_cw"), SYSTEMCTL=str(tmp_path / "systemctl"),
                   ROS2=str(tmp_path / "ros2"), STATE=str(state), NAV_CW_VERIFY_WAIT="0")
        if old_planner:
            env["FAKE_OLD_PLANNER"] = "1"
        return subprocess.run(["bash", SCRIPT, *args], capture_output=True, text=True, env=env)

    run.reinstall = lambda: _install(tmp_path / "nav_cw")
    run.conf = conf / "mission_conf.pb.txt"
    run.comm = conf / "communication_conf.pb.txt"
    run.env = tmp_path / "nav_cw" / "nav_cw.env"
    run.calls = lambda: (state / "log").read_text() if (state / "log").exists() else ""
    run.state = state
    return run


def test_shadow_mode_renames_the_planners_command_topics_and_checks_the_graph(root):
    r = root("shadow", "on")
    assert r.returncode == 0, r.stdout + r.stderr
    c = root.comm.read_text()
    assert 'pub_cmd_vel                 { name: "/nav_cw/shadow/cmd_vel"' in c
    assert 'pub_motion                  { name: "/nav_cw/shadow/mc/motion_control"' in c
    assert "queue_size: 10" in c.split("pub_motion")[1].split("\n")[0]           # the rest of the line kept
    assert 'NAV_CW_SHADOW="on"' in root.env.read_text()
    assert "restart cw-planning.service" in root.calls() and "shadow:           on" in r.stdout
    assert "ok    planning     PUBLISHER         not on /mc/motion_control" in r.stdout
    assert "ok    planning     PUBLISHER         on /nav_cw/shadow/mc/motion_control" in r.stdout
    r = root("shadow", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert root.comm.read_text() == COMM and "shadow:           off" in r.stdout
    assert "NAV_CW_SHADOW" not in root.env.read_text()
    assert "ok    planning     PUBLISHER         on /mc/motion_control" in r.stdout


def test_shadow_mode_stops_a_planner_that_still_drives_the_dog(root):
    """the planner ignores --ros-args (rclcpp::init(0, nullptr)): a shadow mode that the planner does not honour
    must not leave it driving the dog"""
    r = root("shadow", "on", old_planner=True)
    assert r.returncode == 3
    assert "WRONG planning     PUBLISHER         still on /mc/motion_control" in r.stdout
    assert "SHADOW NOT IN EFFECT" in r.stdout and "stop cw-planning.service" in root.calls()
    assert (root.state / "cw-planning.service").read_text().strip() == "inactive"


def test_appmap_ignore_moves_both_off_the_apps_switch(root):
    (root.state / "cw-slam.service").write_text("active")
    r = root("appmap", "ignore")
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'sub_map_switch              { name: "/nav_cw/map/switch" }' in root.comm.read_text()
    assert 'NAV_CW_SLAM_ARGS="map_switch_topic:=/nav_cw/map/switch"' in root.env.read_text()
    assert 'NAV_CW_APPMAP="ignore"' in root.env.read_text()
    assert "restart cw-slam.service" in root.calls() and "restart cw-planning.service" in root.calls()
    assert "ok    planning     SUBSCRIPTION      not on /map/switch" in r.stdout
    assert "ok    mode_switch_node SUBSCRIPTION      not on /map/switch" in r.stdout
    assert "appmap:           ignore" in r.stdout
    r = root("appmap", "follow")
    assert r.returncode == 0, r.stdout + r.stderr
    assert root.comm.read_text() == COMM and "NAV_CW_SLAM_ARGS" not in root.env.read_text()
    assert "ok    mode_switch_node SUBSCRIPTION      on /map/switch" in r.stdout


def test_check_reports_the_graph(root):
    r = root("check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "graph (shadow off, appmap follow)" in r.stdout


def test_debug_switches(root):
    r = root("debug", "planning", "on")
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'ros2 service call /planning/debug/set_pub_flag std_srvs/srv/SetBool {data: true}' in root.calls()
    r = root("debug", "perception", "on")
    assert 'LIDAR3D_ARGS="-p publish_cloud:=true -p publish_tf:=true"' in root.env.read_text()
    assert "cw-nav-perception.service not running" in r.stdout          # restarted only when it runs
    root("debug", "perception", "off")
    assert "LIDAR3D_ARGS" not in root.env.read_text()


def test_speed_caps_only_the_mppi(root):
    r = root("speed", "0.8")
    assert r.returncode == 0, r.stdout + r.stderr
    c = root.conf.read_text()
    teb, mppi = c.split("mppi_planner_conf")
    assert "max_vx: 3.0" in teb and "reference_velocity: 2.0" in teb       # the TEB block untouched
    assert "max_vx: 0.8" in mppi and "reference_velocity: 0.8" in mppi and "max_vy: 0.5" in mppi
    assert 'NAV_CW_SPEED="0.8"' in root.env.read_text() and "restart cw-planning.service" in root.calls()
    assert root("speed", "fast").returncode == 2
    assert "  speed cap:        0.8 m/s\n" in root("switches").stdout             # once (T59: "0.8 m/s0.8")
    root("speed", "0.5")                                  # a second cap: the kept file stays the installed one
    r = root("speed", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    c = root.conf.read_text().split("mppi_planner_conf")[1]
    assert "max_vx: 3.0" in c and "reference_velocity: 1.5" in c
    assert "NAV_CW_SPEED" not in root.env.read_text() and "off (repository value)" in root("switches").stdout


def test_the_switches_hold_across_a_reinstall(root):
    """a reinstall (T60 step 7) puts the repository's configs back; the switches kept in nav_cw.env go in again"""
    (root.state / "cw-planning.service").write_text("inactive")
    for args in (("shadow", "on"), ("appmap", "ignore"), ("speed", "0.8")):
        assert root(*args).returncode == 0
    root.reinstall()
    assert "NOT APPLIED" in root("switches").stdout
    r = root("apply")
    assert r.returncode == 0, r.stdout + r.stderr
    c = root.comm.read_text()
    assert '"/nav_cw/shadow/mc/motion_control"' in c and '"/nav_cw/map/switch"' in c
    assert "max_vx: 0.8" in root.conf.read_text().split("mppi_planner_conf")[1]
    assert "NOT APPLIED" not in r.stdout
    root("speed", "off")
    assert root.conf.read_text() == MISSION


def test_use_new_writes_the_switches_in_before_starting_ours(dog, tmp_path):
    env = tmp_path / "nav_cw" / "nav_cw.env"
    env.write_text('NAV_CW_SHADOW="on"\n')
    r = dog("use-new")
    assert r.returncode == 0, r.stdout + r.stderr
    comm = (tmp_path / "nav_cw" / CONF_DIR / "communication_conf.pb.txt").read_text()
    assert '"/nav_cw/shadow/mc/motion_control"' in comm
    assert "ok    planning     PUBLISHER         not on /mc/motion_control" in r.stdout


def test_a_rehearsal_runs_ours_on_domain_42_in_place_of_their_navigation(dog, tmp_path):
    env = tmp_path / "nav_cw" / "nav_cw.env"
    r = dog("rehearsal", "on")
    assert r.returncode == 1 and "rehearsal needs shadow on" in r.stdout          # never without shadow mode
    assert all(dog.states()[u] == "active" for u in ORIG)
    env.write_text('NAV_CW_SHADOW="on"\n')
    r = dog("rehearsal", "on")
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(dog.states()[u] == "inactive" for u in ORIG) and all(dog.states()[u] == "active" for u in NEW)
    e = env.read_text()
    assert 'ROS_DOMAIN_ID="42"' in e and 'PLANNING_TIMEOUT_WALL_CLOCK="1"' in e and 'NAV_CW_REHEARSAL="on"' in e
    assert "domain 42" in (tmp_path / "state" / "domains").read_text()           # the graph checked on domain 42
    assert "ok    planning     PUBLISHER         not on /mc/motion_control" in r.stdout
    assert "REHEARSAL ON" in dog("switches").stdout
    r = dog("use-new")
    assert r.returncode == 1 and "rehearsal off first" in r.stdout
    r = dog("rehearsal", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(dog.states()[u] == "active" for u in ORIG) and all(dog.states()[u] == "inactive" for u in NEW)
    assert "ROS_DOMAIN_ID" not in env.read_text() and "NAV_CW_REHEARSAL" not in env.read_text()
    assert 'NAV_CW_SHADOW="on"' in env.read_text() and "running: theirs" in r.stdout


def test_no_rehearsal_while_ours_drive_the_dog(dog, tmp_path):
    (tmp_path / "nav_cw" / "nav_cw.env").write_text('NAV_CW_SHADOW="on"\n')
    assert dog("use-new").returncode == 0
    r = dog("rehearsal", "on")
    assert r.returncode == 1 and "use-original first" in r.stdout
    assert all(dog.states()[u] == "active" for u in NEW)


def test_a_rehearsal_that_does_not_come_up_puts_theirs_back(dog, tmp_path):
    env = tmp_path / "nav_cw" / "nav_cw.env"
    env.write_text('NAV_CW_SHADOW="on"\n')
    r = dog("rehearsal", "on", broken=["cw-slam.service"])
    assert r.returncode == 1 and "running: theirs" in r.stdout
    assert all(dog.states()[u] == "active" for u in ORIG)
    assert "ROS_DOMAIN_ID" not in env.read_text()
