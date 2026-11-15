"""replay_dog_bag.sh reads the last replay goal, plays sensors and perception objects, and lines the
replay's Crosswalk yield lines up with the field journal and the notes. A fake replay stands in for
replay_planning_check.sh; nothing here starts the stack."""
import os
import subprocess
import textwrap

from .conftest import ROOT

SH = os.path.join(ROOT, "Nav_D20", "planning", "script", "replay_dog_bag.sh")

FAKE_REPLAY = textwrap.dedent(r'''
    #!/bin/bash
    {
      echo "x=$2"
      echo "y=$3"
      echo "bag=$5"
      echo "extra=${BAG_PLAY_EXTRA-}"
      echo "map=${PLANNING_STARTUP_MAP-}"
    } > "$NAV_FAKE_OUT"
    mkdir -p "$NAV_IO_DIR/logs"
    echo "08:59:58.000000000" > "$NAV_IO_DIR/logs/play_start.log"
    cat > "$NAV_IO_DIR/logs/planning.INFO" << 'EOF'
    I1115 09:00:00.000000  1 main.cc:1] planning start
    I1115 09:00:05.000000  1 tracking.cc:125] Crosswalk yield: wait (vehicle) crosswalk 1
    I1115 09:00:20.000000  1 tracking.cc:125] Crosswalk yield: go (clear) crosswalk 1
    EOF
''')

FAKE_CONVERT = textwrap.dedent(r'''
    #!/usr/bin/env python3
    import os, sys
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst)
    open(os.path.join(dst, "metadata.yaml"), "w").write("converted\n")
    open(os.environ["CONVERT_LOG"], "a").write(src + " -> " + dst + "\n")
''')


def _bag(tmp, goal="10.000 20.000"):
    os.makedirs(tmp)
    open(os.path.join(tmp, "record_log.txt"), "w").write(
        "[17:32:30] recording /home/nvidia/bags/cw, domain 0: 18 topics\n"
        "[17:32:32] watching\n"
        "[17:32:33] replay goal: 1.000 2.000\n"
        "[17:32:40] note: go was early\n"
        "[17:33:00] replay goal: %s\n" % goal)
    open(os.path.join(tmp, "journal_cw-planning.txt"), "w").write(
        "2026-11-15T17:32:35.100000+0800 host cw-planning[1]: I1115 08:44:35.100000 1 tracking.cc:125] "
        "Crosswalk yield: wait (vehicle) crosswalk 1\n"
        "Nov 15 17:32:50.100000 host cw-planning[1]: I1115 08:44:50.100000 1 tracking.cc:125] "
        "Crosswalk yield: go (clear) crosswalk 1\n")
    open(os.path.join(tmp, "metadata.yaml"), "w").write("storage_identifier: sqlite3\n")
    open(os.path.join(tmp, "x.db3"), "w").close()


def _run(tmp, *args, env):
    fake = os.path.join(tmp, "fake_replay.sh")
    open(fake, "w").write(FAKE_REPLAY)
    os.chmod(fake, 0o755)
    io = os.path.join(tmp, "io")
    out = os.path.join(tmp, "called.txt")
    env = dict(env)
    env.update(NAV_REPLAY_CHECK=fake, NAV_IO_DIR=io, NAV_FAKE_OUT=out)
    proc = subprocess.run(["bash", SH, *args], cwd=tmp, env=env, text=True, capture_output=True)
    return proc, io, out


def test_goal_play_method_and_sequence(tmp_path):
    bag = str(tmp_path / "cw")
    _bag(bag)
    os.makedirs(bag + "_mcap")
    env = dict(os.environ)
    env["NAV_BAG_TO_MCAP"] = "/no/such/converter"
    proc, io, out = _run(str(tmp_path), bag, "t68_fake", env=env)
    assert proc.returncode == 0, proc.stderr
    called = open(out).read()
    assert "x=10.000" in called and "y=20.000" in called
    assert "x=1.000" not in called
    assert called.split("bag=", 1)[1].splitlines()[0] == bag + "_mcap"
    extra = called.split("extra=", 1)[1].splitlines()[0]
    for topic in ("/robot/topic/rdap/top_point_cloud2", "/rte/rdap/rtk/inspvax",
                  "/rte/rdap/rtk/imus/data_raw", "/perception/lidar_3d/objects"):
        assert topic in extra
    assert "--topics" in extra
    assert "ad68517c" in called
    seq = open(os.path.join(io, "sequence.md")).read()
    assert "goal 10.000 20.000" in seq
    field = [line for line in seq.splitlines() if line.startswith("|") and "Crosswalk yield" in line or "note:" in line]
    texts = "\n".join(field)
    assert texts.index("wait (vehicle)") < texts.index("note: go was early") < texts.index("go (clear)")
    # both columns in seconds after the bag started: the field from the recording line, the replay from play_start
    # + the 1 s --delay; the journal's stamp in either journalctl format
    assert "| 5.1 | Crosswalk yield: wait (vehicle) crosswalk 1 | |" in seq
    assert "| 10.0 | note: go was early | |" in seq
    assert "| 20.1 | Crosswalk yield: go (clear) crosswalk 1 | |" in seq
    assert "| 6.0 | | Crosswalk yield: wait (vehicle) crosswalk 1 |" in seq
    assert "| 21.0 | | Crosswalk yield: go (clear) crosswalk 1 |" in seq


def test_sqlite_bag_is_converted_when_mcap_is_missing(tmp_path):
    bag = str(tmp_path / "cw")
    _bag(bag)
    conv = os.path.join(tmp_path, "convert.py")
    open(conv, "w").write(FAKE_CONVERT)
    env = dict(os.environ)
    env["NAV_BAG_TO_MCAP"] = conv
    env["CONVERT_LOG"] = str(tmp_path / "convert.log")
    proc, io, out = _run(str(tmp_path), bag, env=env)
    assert proc.returncode == 0, proc.stderr
    assert open(env["CONVERT_LOG"]).read().strip() == "%s -> %s_mcap" % (bag, bag)
    assert open(out).read().split("bag=", 1)[1].splitlines()[0] == bag + "_mcap"


def test_missing_goal_points_at_bag_on_map(tmp_path):
    bag = str(tmp_path / "cw")
    os.makedirs(bag)
    open(os.path.join(bag, "record_log.txt"), "w").write("[17:32:32] watching\n")
    proc = subprocess.run(["bash", SH, bag], cwd=str(tmp_path), env=dict(os.environ), text=True, capture_output=True)
    assert proc.returncode == 1
    assert "bag_on_map.py" in proc.stderr


def test_the_planners_transformed_goal_wins_over_the_record_log(tmp_path):
    """T66: Foxglove sent the goal in base_link; record_log.txt has those numbers, the planner's log the map ones."""
    bag = str(tmp_path / "cw")
    _bag(bag)
    os.makedirs(bag + "_mcap")
    with open(os.path.join(bag, "journal_cw-planning.txt"), "a") as f:
        f.write("2026-11-15T17:32:31.000000+0800 host cw-planning[1]: I1115 08:44:31.0 1 ros_bridge.cc:653] "
                "[PLANNING]Transformed goal from base_link to map: (188.25, 470.5)\n")
    proc, io, out = _run(str(tmp_path), bag, "t66_fake", env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    called = open(out).read()
    assert "x=188.25" in called and "y=470.5" in called and "规划日志" in proc.stdout
    proc, io, out = _run(str(tmp_path), bag, "t66_fake", env=dict(os.environ, NAV_REPLAY_GOAL="1.5 -2"))
    assert "x=1.5" in open(out).read() and "NAV_REPLAY_GOAL" in proc.stdout
