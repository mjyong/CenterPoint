"""Nav_D20/docker/dog_rootfs.sh and run_arm64.sh (T60): dog 8's root filesystem as an arm64 image, and builds in it.
The tar runs for real, against a made-up dog root behind a fake ssh: secrets, homes, the robot team's installs and
our build trees must stay out; the system and our deps go in."""
import os
import subprocess
import tarfile

import pytest

from .conftest import ROOT

ROOTFS = os.path.join(ROOT, "Nav_D20", "docker", "dog_rootfs.sh")
RUN = os.path.join(ROOT, "Nav_D20", "docker", "run_arm64.sh")

# fake ssh: runs the command here, with the dog's / at $FAKE_ROOT and without sudo / nice / ionice
FAKE_SSH = r"""#!/bin/bash
[[ "$1" == -C ]] && shift
shift
cmd="$*"
cmd="${cmd//sudo -n nice -n 19 ionice -c3 /}"
cmd="${cmd//sudo -n /}"
cmd="${cmd//tar -C \/ /tar -C $FAKE_ROOT }"
[[ "$cmd" == "ls -A /opt/seres"* ]] && cmd="${cmd//\/opt\/seres/$FAKE_ROOT/opt/seres}"
[[ "$cmd" == *"du -xsh"* ]] && exit 0
exec bash -c "$cmd"
"""

FILES = {
    # in
    "usr/lib/aarch64-linux-gnu/libfoo.so.1": True,
    "usr/local/cuda-12.6/bin/nvcc": True,
    "opt/ros/humble/setup.bash": True,
    "opt/seres/nav_cw/deps/lib/libosqp.so": True,
    "etc/passwd": True,
    "etc/ssh/sshd_config": True,
    # out: secrets
    "etc/shadow": False,
    "etc/gshadow": False,
    "etc/ssh/ssh_host_ed25519_key": False,
    "etc/ssh/ssh_host_rsa_key.pub": False,
    "etc/NetworkManager/system-connections/wifi.nmconnection": False,
    "etc/netplan/01-wifi.yaml": False,
    # out: people's files, logs, the robot team's installs, our build trees and maps
    "home/nvidia/bags/x.mcap": False,
    "root/.bash_history": False,
    "var/log/syslog": False,
    "opt/seres/nav/planning/bin/planning": False,
    "opt/seres/nav_perception/lib/libglog.so.0": False,
    "opt/seres/nav_cw/install/planning/setup.bash": False,
    "opt/seres/nav_cw/maps/ad68517c/global_map.yaml": False,
    "opt/seres/nav_cw/src/Nav_D20/README.md": False,
    "tmp/x": False,
    "usr/src/linux-headers/Makefile": False,
}


@pytest.fixture
def dog(tmp_path):
    root = tmp_path / "dogroot"
    for rel in FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "ssh").write_text(FAKE_SSH)
    (bin_ / "docker").write_text('#!/bin/bash\necho "docker $*" >> "%s/docker.log"\n' % tmp_path)
    for f in ("ssh", "docker"):
        (bin_ / f).chmod(0o755)

    def run(*args, **extra):
        env = dict(os.environ, FAKE_ROOT=str(root), NAV_SSH=str(bin_ / "ssh"), NAV_DOCKER=str(bin_ / "docker"),
                   **extra)
        return subprocess.run(["bash", ROOTFS, *args], capture_output=True, text=True, env=env)

    run.docker_log = lambda: (tmp_path / "docker.log").read_text()
    return run


def test_the_tar_has_the_system_and_our_deps_but_no_secrets_homes_or_their_installs(dog, tmp_path):
    r = dog(str(tmp_path / "out"), "20261115")
    assert r.returncode == 0, r.stdout + r.stderr
    tar = tmp_path / "out" / "dog8_rootfs_20261115.tar"
    with tarfile.open(tar) as t:
        names = {n[2:] if n.startswith("./") else n for n in t.getnames()}
    wrong = [rel for rel, keep in FILES.items() if (rel in names) != keep]
    assert not wrong, wrong
    assert (tmp_path / "out" / "dog8_rootfs_20261115.list.txt").exists()
    log = dog.docker_log()
    assert "import --platform linux/arm64" in log and "dog8-rootfs:20261115" in log
    assert "tag dog8-rootfs:20261115 dog8-rootfs:latest" in log


def test_a_dry_run_only_prints(dog, tmp_path):
    r = dog(str(tmp_path / "out"), "x", NAV_DRY_RUN="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--exclude=./opt/seres/nav " in r.stdout and "--exclude=./opt/seres/nav_cw/install" in r.stdout
    assert "--exclude=./opt/seres/nav_cw/deps" not in r.stdout and "--exclude=./opt/seres/nav_cw " not in r.stdout
    assert not (tmp_path / "out").exists()


# fake docker: a container is a file $DSTATE/<name> holding its mounts; inspect, run -d, exec, start, rm
FAKE_DOCKER = r"""#!/bin/bash
echo "docker $*" >> "$DSTATE/log"
case "$1" in
  inspect) f="$DSTATE/${@: -1}"; [[ -e "$f" ]] || exit 1
           if [[ "$3" == *State.Running* ]]; then echo true; else cat "$f"; fi ;;
  run) name=""; mounts=""; prev=""
       for a in "$@"; do
         [[ "$prev" == --name ]] && name="$a"
         [[ "$prev" == -v ]] && mounts="${mounts}${a%%:*}=${a#*:};"
         prev="$a"
       done
       echo "$mounts" > "$DSTATE/$name"; echo cid ;;
  rm) rm -f "$DSTATE/${@: -1}" ;;
esac
exit 0
"""


@pytest.fixture
def arm64(tmp_path):
    state = tmp_path / "docker"
    state.mkdir()
    fake = tmp_path / "fake_docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)

    def run(work, *args, **extra):
        env = dict(os.environ, NAV_DOCKER=str(fake), DSTATE=str(state), NAV_JOBS="12", NAV_ARM64_NAME="nav-arm64-t",
                   **extra)
        return subprocess.run(["bash", RUN, str(work), *args], capture_output=True, text=True, env=env)

    run.log = lambda: (state / "log").read_text().splitlines() if (state / "log").exists() else []
    return run


def test_builds_run_at_the_dogs_paths_without_network(tmp_path, arm64):
    r = arm64(tmp_path / "w", "bash", "build.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    log = arm64.log()
    create = next(l for l in log if l.startswith("docker run -d"))
    assert "--platform linux/arm64" in create and "--network none" in create and create.endswith("sleep infinity")
    for d in ("src", "build", "install", "log"):
        assert "%s:/opt/seres/nav_cw/%s" % (tmp_path / "w" / d, d) in create
        assert (tmp_path / "w" / d).is_dir()
    assert "/opt/seres/nav_cw/deps:" not in create            # the dog's deps come from the image, not a mount
    chmod = next(l for l in log if " chmod " in l)
    assert chmod.startswith("docker exec -u 0 nav-arm64-t chmod -R a+rX") and "cmake/osqp" in chmod   # T60
    run = log[-1]
    assert run.startswith("docker exec -u %d:%d" % (os.getuid(), os.getgid()))
    assert "NAV_CUDA_ARCH=87" in run and "NAV_JOBS=12" in run and run.endswith("nav-arm64-t bash build.sh")


def test_the_kept_container_is_reused_and_checked(tmp_path, arm64):
    assert arm64(tmp_path / "w", "true").returncode == 0
    assert arm64(tmp_path / "w", "bash", "b.sh").returncode == 0
    assert sum(l.startswith("docker run -d") for l in arm64.log()) == 1     # created once (50 min on vfs, T60)
    r = arm64(tmp_path / "other", "true")                                   # another work directory: refused
    assert r.returncode == 2 and "mounts another work directory" in r.stdout
    assert arm64(tmp_path / "w", "--stop").returncode == 0
    assert arm64.log()[-1] == "docker rm -f nav-arm64-t"


def test_a_dry_run_prints_the_commands(tmp_path, arm64):
    r = arm64(tmp_path / "w", "bash", "build.sh", NAV_DRY_RUN="1")
    assert r.returncode == 0 and "run -d --name nav-arm64-t" in r.stdout and "sleep infinity" in r.stdout
    assert not arm64.log()
