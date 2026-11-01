"""What deploy_deb (the C++ pipeline, its ROS 2 node and the vehicle's interfaces; docs/DEPLOY_ORIN.md, section 9)
takes from this repository, checked:

- config/lidar_3d_detection.yaml is ``export_cpp_config.py --robot kygi`` (the Python dataclasses, the single
  source of the parameters) plus the deploy-only keys at the end of its node: section (DEPLOY_KEYS, kept as
  they are);
- deploy_deb's kygi665_msgs messages have the fields of this repository's kygi665_msgs (what the Python reads).

    python tools/deploy/check_deploy_deb.py            # exit 1 when something differs
    python tools/deploy/check_deploy_deb.py --write    # rewrite the config from the dataclasses (after changing them)
"""
import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NODE = os.path.join(ROOT, "deploy_deb", "perception_combo_ros", "node_src", "lidar_3d_detection")
CONFIG = os.path.join(NODE, "config", "lidar_3d_detection.yaml")
KYGI = os.path.join(ROOT, "deploy_deb", "slam", "interface", "kygi665_msgs", "msg")
# node: keys only the vehicle deployment has (the model next to the config, the vehicle's topic names)
DEPLOY_KEYS = ("model_dir", "objects_topic", "tracks_topic", "predictions_topic", "cloud_topic")


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


def fields(path):
    """The message's field and constant lines without comments."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(" ".join(line.split()))
    return out


def message_problems():
    out = []
    ref = os.path.join(ROOT, "kygi665_msgs", "msg")
    for f in sorted(set(os.listdir(ref)) | set(os.listdir(KYGI))):
        a, b = os.path.join(ref, f), os.path.join(KYGI, f)
        if not (os.path.exists(a) and os.path.exists(b)) or fields(a) != fields(b):
            out.append("deploy_deb kygi665_msgs/msg/%s differs from kygi665_msgs/msg/%s" % (f, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the config from the dataclasses")
    args = ap.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        current = f.read()
    want = expected_config(current)
    stale = want != current
    if stale and args.write:
        with open(CONFIG, "w", encoding="utf-8") as f:
            f.write(want)
    problems = message_problems()
    if stale:
        print("%s: %s" % (os.path.relpath(CONFIG, ROOT),
                          "rewritten" if args.write else "differs from export_cpp_config.py --robot kygi"))
    for p in problems:
        print(p)
    print("config %s, %d message problems" % ("stale" if stale and not args.write else "ok", len(problems)))
    return 1 if problems or (stale and not args.write) else 0


if __name__ == "__main__":
    sys.exit(main())
