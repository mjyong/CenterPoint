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
    topics = set(cfg["3D!live"]["topics"])
    for k, v in cfg.items():
        for p in v.get("paths", []):
            topics.add(p["value"].split(".")[0])
        for key in ("topicPath", "topicToRender"):
            if key in v:
                topics.add(v[key].split(".")[0])
    assert cfg["3D!live"]["publish"]["poseTopic"] == "/goal_pose"
    topics.add("/goal_pose")
    script = open(os.path.join(ROOT, "tools", "deploy", "nav_cw_foxglove.sh"), encoding="utf-8").read()
    block = script[script.index("TOPICS=("):script.index(")\nread")]
    patterns = re.findall(r"'([^']+)'", block)
    missing = [t for t in sorted(topics) if not any(re.search(p, t) for p in patterns)]
    assert not missing, missing
    assert "^/goal_pose$" in script.split("client_topic_whitelist")[1].split("\n")[0]   # only the goal
    assert "/planning/debug/crosswalk" in topics
