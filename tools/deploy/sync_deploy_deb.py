"""Keep deploy_deb (the vehicle deployment tree) aligned with deploy/ (docs/DEPLOY_ORIN.md, section 9).

deploy/ is where the code changes and is tested against Python; deploy_deb carries the same library, ROS
conversion and bag replay under the vehicle's names (namespace lidar3d, package lidar_3d_detection, messages
perception_interface/Lidar*). Every mirrored file is its deploy/ source with RENAMES applied, so the two can
only differ in what deploy_deb owns: the node (glog, the vehicle's topics), CMakeLists.txt, package.xml,
launch, packaging and the deploy-only keys of the config.

    python tools/deploy/sync_deploy_deb.py            # check: lists what differs, exit 1 if anything does
    python tools/deploy/sync_deploy_deb.py --write    # rewrite the mirrored files and the config from deploy/

Also checked (not written): the fields of perception_interface's Lidar* messages against dog_perception_msgs,
and deploy_deb's kygi665_msgs messages against the repository's.
"""
import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WS = os.path.join(ROOT, "deploy_deb", "perception_combo_ros")
DEB = os.path.join(WS, "node_src", "lidar_3d_detection")
INTERFACE = os.path.join(WS, "common_interface", "seres_interface", "ros", "perception_interface", "msg")
KYGI = os.path.join(ROOT, "deploy_deb", "slam", "interface", "kygi665_msgs", "msg")

# (source under ROOT, destination under DEB): a directory is mirrored file by file
MIRROR = [
    ("deploy/cpp/include/dogperc", "include/lidar3d"),
    ("deploy/cpp/src", "src"),
    ("deploy/cpp/tests", "tests"),
    ("deploy/cpp/tools/dogperc_replay.cpp", "tools/lidar_3d_replay.cpp"),
    ("deploy/ros2/dogperc_ros/src/ros_convert.hpp", "include/ros_convert.hpp"),
    ("deploy/ros2/dogperc_ros/src/ros_convert.cpp", "src/ros_convert.cpp"),
    ("deploy/ros2/dogperc_ros/src/bag_replay.cpp", "src/bag_replay.cpp"),
    ("deploy/scripts/orin_check.sh", "scripts/orin_check.sh"),
]
# deploy_deb's own files inside the mirrored directories
OWN = {"src/lidar_3d_detection.cpp", "src/main.cpp"}
# in this order: the message types and the executables before the generic names
RENAMES = [
    ("dog_perception_msgs::msg::ObjectArray", "perception_interface::msg::LidarObjectArray"),
    ("dog_perception_msgs::msg::PredictedPath", "perception_interface::msg::LidarPredictedPath"),
    ("dog_perception_msgs::msg::Object", "perception_interface::msg::LidarObject"),
    ("dog_perception_msgs/msg/object_array.hpp", "perception_interface/msg/lidar_object_array.hpp"),
    ("dogperc_bag_replay", "lidar_3d_bag_replay"),
    ("dogperc_replay", "lidar_3d_replay"),
    ("dogperc_node", "lidar_3d_detection"),
    ("dogperc_ros", "lidar_3d_detection"),
    ("dog_perception_msgs", "perception_interface"),
    ("dog_perception", "lidar_3d_detection"),
    ("DOGPERC", "LIDAR3D"),
    ("dogperc", "lidar3d"),
]
CONFIG = "config/lidar_3d_detection.yaml"
# node: keys only the vehicle deployment has (the model next to the config, the vehicle's topic names)
DEPLOY_KEYS = ("model_dir", "objects_topic", "tracks_topic", "predictions_topic", "cloud_topic")
MESSAGES = {"ObjectArray": "LidarObjectArray", "Object": "LidarObject", "PredictedPath": "LidarPredictedPath"}


def rename(text):
    for a, b in RENAMES:
        text = text.replace(a, b)
    return text


def mirrored():
    """[(source, destination)] absolute paths."""
    out = []
    for src, dst in MIRROR:
        s = os.path.join(ROOT, src)
        if os.path.isfile(s):
            out.append((s, os.path.join(DEB, dst)))
            continue
        for d, _, files in os.walk(s):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(d, f), s)
                out.append((os.path.join(d, f), os.path.join(DEB, dst, rename(rel))))
    return out


def extra_files(pairs):
    """Files in the mirrored directories that no source maps to and deploy_deb does not own."""
    targets = {os.path.relpath(d, DEB) for _, d in pairs}
    extra = []
    for _, dst in MIRROR:
        d = os.path.join(DEB, dst)
        if not os.path.isdir(d):
            continue
        for sub, _, files in os.walk(d):
            for f in files:
                rel = os.path.relpath(os.path.join(sub, f), DEB)
                if rel not in targets and rel not in OWN:
                    extra.append(rel)
    return sorted(extra)


def expected_config(current):
    """export_cpp_config.py --robot kygi, plus the deploy keys of the current file (kept as they are)."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
    from export_cpp_config import config_dict, node_section, to_yaml
    from dog_perception.node_config import node_parameters, preprocess_config

    text = to_yaml(config_dict(preprocess_config(node_parameters("kygi")), source=dict(robot="kygi"),
                               node=node_section("kygi"))) + "\n"
    keys = "|".join(DEPLOY_KEYS)
    extra = [line for line in current.splitlines() if re.match(r"^  (%s): " % keys, line)]
    if not text.rstrip("\n").splitlines()[-1].startswith("  "):
        raise SystemExit("export_cpp_config.py no longer ends with the node: section; place the deploy keys by hand")
    return text + "".join(line + "\n" for line in extra)


def fields(path, renames=None):
    """The message's field and constant lines without comments."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for a, b in (renames or {}).items():
                line = re.sub(r"\b%s\b" % a, b, line)
            out.append(" ".join(line.split()))
    return out


def interface_problems():
    out = []
    msgs = os.path.join(ROOT, "deploy", "ros2", "dog_perception_msgs", "msg")
    for ours, theirs in MESSAGES.items():
        if fields(os.path.join(msgs, ours + ".msg"), MESSAGES) != fields(os.path.join(INTERFACE, theirs + ".msg")):
            out.append("perception_interface/msg/%s.msg: fields differ from dog_perception_msgs/msg/%s.msg"
                       % (theirs, ours))
    ref = os.path.join(ROOT, "kygi665_msgs", "msg")
    for f in sorted(set(os.listdir(ref)) | set(os.listdir(KYGI))):
        a, b = os.path.join(ref, f), os.path.join(KYGI, f)
        if not (os.path.exists(a) and os.path.exists(b)) or fields(a) != fields(b):
            out.append("deploy_deb kygi665_msgs/msg/%s differs from kygi665_msgs/msg/%s" % (f, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the mirrored files and the config")
    args = ap.parse_args()

    pairs = mirrored()
    stale = []
    for src, dst in pairs:
        with open(src) as f:
            want = rename(f.read())
        have = open(dst).read() if os.path.exists(dst) else None
        if have != want:
            stale.append(os.path.relpath(dst, DEB))
            if args.write:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    f.write(want)
                if os.access(src, os.X_OK):
                    os.chmod(dst, 0o755)
    cfg = os.path.join(DEB, CONFIG)
    with open(cfg) as f:
        current = f.read()
    want = expected_config(current)
    if want != current:
        stale.append(CONFIG)
        if args.write:
            with open(cfg, "w") as f:
                f.write(want)
    extra = extra_files(pairs)
    problems = interface_problems()

    verb = "rewritten" if args.write else "differs from deploy/"
    for p in stale:
        print("%s: %s" % (p, verb))
    for p in extra:
        print("%s: no source in deploy/ (removed there? delete it, or add it to OWN)" % p)
    for p in problems:
        print(p)
    print("%d mirrored files, %d %s, %d extra, %d interface problems"
          % (len(pairs) + 1, len(stale), "rewritten" if args.write else "stale", len(extra), len(problems)))
    return 1 if extra or problems or (stale and not args.write) else 0


if __name__ == "__main__":
    sys.exit(main())
