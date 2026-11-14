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


@pytest.fixture
def dog(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    fake = tmp_path / "systemctl"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    for u in ORIG:
        (state / u).write_text("active")
    for u in NEW:
        (state / (u + ".unit")).write_text("")

    def run(*args, broken=()):
        (state / "broken").write_text("".join(u + "\n" for u in broken))
        env = dict(os.environ, STATE=str(state), SYSTEMCTL=str(fake), JOURNALCTL="true", NAV_CW_WAIT="0")
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
    """an installed tree with a mission_conf, a fake systemctl (cw-planning running) and a fake ros2"""
    conf = tmp_path / "nav_cw/install/planning/planning/lib/planning/robot_factory/d20/001/planning_conf"
    conf.mkdir(parents=True)
    (conf / "mission_conf.pb.txt").write_text(
        "teb_planner_conf{\n  dynamic_constraints{\n    max_vx: 3.0\n    reference_velocity: 2.0\n  }\n}\n"
        "mppi_planner_conf{\n  dynamic_constraints{\n    max_vx: 3.0\n    max_vy: 0.5\n  }\n"
        "  reference_velocity_critic{\n    enabled: true\n    reference_velocity: 1.5\n  }\n}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "systemctl").write_text('#!/bin/bash\necho "$*" >> "%s/calls"\n'
                                     '[[ "$1" == is-active ]] && [[ "${@: -1}" == cw-planning.service ]] && exit 0\n'
                                     '[[ "$1" == is-active ]] && exit 3\nexit 0\n' % tmp_path)
    (bin_ / "ros2").write_text('#!/bin/bash\necho "ros2 $*" >> "%s/calls"\n' % tmp_path)
    for f in ("systemctl", "ros2"):
        (bin_ / f).chmod(0o755)

    def run(*args):
        env = dict(os.environ, NAV_CW_ROOT=str(tmp_path / "nav_cw"), SYSTEMCTL=str(bin_ / "systemctl"),
                   ROS2=str(bin_ / "ros2"))
        return subprocess.run(["bash", SCRIPT, *args], capture_output=True, text=True, env=env)

    run.conf = conf / "mission_conf.pb.txt"
    run.env = tmp_path / "nav_cw" / "nav_cw.env"
    run.calls = lambda: (tmp_path / "calls").read_text() if (tmp_path / "calls").exists() else ""
    return run


def test_shadow_mode_remaps_the_planners_commands(root):
    r = root("shadow", "on")
    assert r.returncode == 0, r.stdout + r.stderr
    env = root.env.read_text()
    assert "NAV_CW_PLANNING_ARGS=" in env and "-r /cmd_vel:=/nav_cw/shadow/cmd_vel" in env
    assert "restart cw-planning.service" in root.calls() and "shadow:           on" in r.stdout
    r = root("shadow", "off")
    assert "NAV_CW_PLANNING_ARGS" not in root.env.read_text() and "shadow:           off" in r.stdout


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
    assert "speed cap:        0.8 m/s" in root("switches").stdout
    root("speed", "0.5")                                  # a second cap: the kept file stays the installed one
    r = root("speed", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    c = root.conf.read_text().split("mppi_planner_conf")[1]
    assert "max_vx: 3.0" in c and "reference_velocity: 1.5" in c
    assert "NAV_CW_SPEED" not in root.env.read_text() and "off (repository value)" in root("switches").stdout
