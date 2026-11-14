"""One environment for the replays (Nav_D20/docker): build_stack.sh builds each module into its own prefix against its
own interface copies; nav_env.sh runs each process with its module's prefix in front."""
import os
import subprocess

from .conftest import ROOT

BUILD = os.path.join(ROOT, "Nav_D20", "docker", "build_stack.sh")
ENV = os.path.join(ROOT, "Nav_D20", "planning", "script", "nav_env.sh")


def _dry(*args, **env):
    e = dict(os.environ, NAV_DRY_RUN="1", NAV_CUDA_ARCH="89", **env)
    r = subprocess.run(["bash", BUILD, *args], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [l for l in r.stdout.splitlines() if l.startswith("colcon ")]
    return r.stdout, cmds


def _opt(cmd, name):
    """the words after --<name> up to the next option"""
    w = cmd.split()
    i = w.index("--" + name)
    out = []
    for x in w[i + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def test_each_module_its_own_prefix_and_interface_copies(tmp_path):
    out, cmds = _dry(str(tmp_path / "nd"))
    by = {}
    for c in cmds:
        prefix = _opt(c, "install-base")[0]
        by.setdefault(os.path.basename(prefix), []).append(c)
    assert sorted(by) == ["nav_perception", "planning", "slam"]
    for m, cs in by.items():
        for c in cs:
            assert _opt(c, "install-base") == [str(tmp_path / "nd" / "install" / m)]
            assert _opt(c, "build-base") == [str(tmp_path / "nd" / "build" / m)]
    nav = os.path.join(ROOT, "Nav_D20")
    # planning and slam take their interfaces from the central seres_interface (their aarch64 builds do)
    assert _opt(by["planning"][0], "base-paths") == [os.path.join(nav, "seres_interface")]
    assert "perception_interface" in _opt(by["planning"][0], "packages-select")
    assert _opt(by["slam"][0], "base-paths") == [os.path.join(nav, "seres_interface")]
    assert "kygi665_msgs" in _opt(by["slam"][0], "packages-select")
    assert _opt(by["planning"][-1], "packages-select") == ["planning"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in by["planning"][-1]
    assert _opt(by["slam"][-1], "packages-select") == ["lightning"]
    # the robot team's perception, with our lidar_3d_detection in it: its own common_interface
    assert any(_opt(c, "base-paths") == [os.path.join(nav, "perception_combo_ros", "common_interface")]
               for c in by["nav_perception"])
    assert _opt(by["nav_perception"][-1], "packages-select") == ["lidar_seg", "grid_map_memory_node",
                                                                  "lidar_3d_detection"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in by["nav_perception"][-1]


def test_octomap_msgs_from_source_only_when_no_prefix_has_them(tmp_path):
    out, cmds = _dry(str(tmp_path), "nav_perception", AMENT_PREFIX_PATH="/nonexistent")
    assert any("octomap_related" in c for c in cmds)
    ros = tmp_path / "ros"
    (ros / "share" / "octomap_msgs").mkdir(parents=True)
    (ros / "share" / "pcl_msgs").mkdir(parents=True)
    out, cmds = _dry(str(tmp_path), "nav_perception", AMENT_PREFIX_PATH=str(ros))
    assert not any("octomap_related" in c for c in cmds)


def test_nav_env_runs_a_process_with_its_modules_prefix_first(tmp_path):
    root = tmp_path / "install"
    for m in ("nav_perception", "slam", "planning"):
        (root / m).mkdir(parents=True)
        (root / m / "setup.bash").write_text('export ORDER="${ORDER:-}%s "\n' % m)
    fake_ros = tmp_path / "opt_ros"
    fake_ros.mkdir()
    script = r"""
set -euo pipefail
source "%s"
source() { if [[ "$1" == /opt/ros/humble/setup.bash ]]; then builtin source "%s/setup.bash"; else builtin source "$@"; fi; }
nav_env_setup
echo "all: $ORDER"
echo "bin: $(nav_bin slam lightning run_loc_online)"
( nav_exec slam bash -c 'echo "slam: $ORDER"' )
""" % (ENV, fake_ros)
    (fake_ros / "setup.bash").write_text("export ORDER=ros\\ \n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env=dict(os.environ, NAV_STACK_ROOT=str(root)))
    assert r.returncode == 0, r.stdout + r.stderr
    lines = dict(l.split(": ", 1) for l in r.stdout.splitlines())
    assert lines["all"].split() == ["ros", "nav_perception", "slam", "planning"]   # planning last
    assert lines["bin"] == "%s/slam/lightning/lib/lightning/run_loc_online" % root
    assert lines["slam"].split()[-1] == "slam"                    # its own prefix sourced last = in front


def test_nav_env_without_the_image_is_the_old_micromamba_layout():
    r = subprocess.run(["bash", "-c", 'source "%s"; nav_bin planning planning planning' % ENV], capture_output=True,
                       text=True, env={k: v for k, v in os.environ.items() if k != "NAV_STACK_ROOT"})
    assert r.stdout.strip() == "/public/zjst/robot/planning_humble_env/colcon/install/planning/lib/planning/planning"
