"""What the C++ node published: a rosbag2 of ``~/objects`` (dog_perception_msgs/ObjectArray, recorded with
``ros2 bag record``), summarized and checked against a replay of the same input (dogperc_bag_replay or
dogperc_replay --messages, their .npz): one message per frame (none dropped), the same stamps, the same
track ids in each frame.

    python tools/deploy/objects_bag.py --bag <recorded bag> --replay <replay.npz> --out objects.json

A bag played in a loop (``ros2 bag play --loop``) gives one pass per loop, split where the stamps jump back;
each pass is checked against the replay, the last one up to its last frame when the recording stopped
during it (more than one pass).

The message definitions are read from deploy/ros2/dog_perception_msgs/msg (rosbags, no ROS install), and the
vehicle node's perception_interface/LidarObjectArray (deploy_deb, the same fields) from its interface package.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MSG_DIR = os.path.join(ROOT, "deploy", "ros2", "dog_perception_msgs", "msg")
NAMES = ("PredictedPath", "Object", "ObjectArray")
DEB_MSG_DIR = os.path.join(ROOT, "deploy_deb", "perception_combo_ros", "common_interface", "seres_interface", "ros",
                           "perception_interface", "msg")
DEB_NAMES = ("LidarPredictedPath", "LidarObject", "LidarObjectArray")
ARRAYS = ("dog_perception_msgs/msg/ObjectArray", "perception_interface/msg/LidarObjectArray")


def typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for package, msg_dir, names in (("dog_perception_msgs", MSG_DIR, NAMES),
                                    ("perception_interface", DEB_MSG_DIR, DEB_NAMES)):
        if not os.path.isdir(msg_dir):
            continue
        for name in names:
            with open(os.path.join(msg_dir, name + ".msg")) as f:
                types.update(get_types_from_msg(f.read(), "%s/msg/%s" % (package, name)))
    ts.register(types)
    return ts


def read_objects(bag, topic=None):
    from rosbags.highlevel import AnyReader

    rows = []
    with AnyReader([Path(bag)], default_typestore=typestore()) as r:
        conns = [c for c in r.connections if c.msgtype in ARRAYS and (topic is None or c.topic == topic)]
        if not conns:
            raise SystemExit("%s: no %s topic" % (bag, " or ".join(ARRAYS)))
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            rows.append(dict(stamp=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, received=t * 1e-9,
                             frame=m.header.frame_id, ids=[int(o.id) for o in m.objects],
                             paths=sum(len(o.predictions) for o in m.objects), msg=m))
    return rows


def split_passes(rows, jump_back=1.0):
    """Rows in recording order -> one list per pass of the bag (a stamp more than ``jump_back`` s before the
    previous one starts a pass: the node starts over there, dogperc/online.hpp)."""
    out = []
    for r in rows:
        if not out or r["stamp"] < out[-1][-1]["stamp"] - jump_back:
            out.append([])
        out[-1].append(r)
    return out


def check_pass(rows, stamps, offsets, ids, partial=False):
    by_stamp = {round(float(s), 6): k for k, s in enumerate(stamps)}
    matched = [(r, by_stamp.get(round(r["stamp"], 6))) for r in rows]
    found = [k for _, k in matched if k is not None]
    expected = (max(found) + 1 if found else 0) if partial else len(stamps)
    gaps = np.diff([r["received"] for r in rows])
    return dict(messages=len(rows), matched_frames=len(set(found)),
                received_gap_max=float(gaps.max()) if len(gaps) else 0.0,
                unmatched_messages=sum(k is None for _, k in matched),
                missing_frames=expected - len({k for k in found if k < expected}),
                same_ids_frames=sum(1 for r, k in matched if k is not None
                                    and r["ids"] == list(ids[offsets[k]:offsets[k + 1]])), partial=partial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="recorded ~/objects")
    ap.add_argument("--topic", default=None, help="default: every (Lidar)ObjectArray topic in the bag")
    ap.add_argument("--replay", default=None, help="replay .npz of the same input to compare with")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = read_objects(args.bag, args.topic)
    report = dict(messages=len(rows), objects=int(sum(len(r["ids"]) for r in rows)),
                  paths=int(sum(r["paths"] for r in rows)), frame_ids=sorted({r["frame"] for r in rows}))
    if rows:
        o = next((x for r in rows for x in r["msg"].objects), None)
        if o is not None:
            report["example"] = dict(id=int(o.id), label=o.label_name, motion=int(o.motion), score=float(o.score),
                                     position=[o.position.x, o.position.y, o.position.z], yaw=float(o.yaw),
                                     size=[o.size.x, o.size.y, o.size.z],
                                     prediction_times=[float(v) for v in o.prediction_times], paths=len(o.predictions))
    if args.replay:
        z = np.load(args.replay)
        stamps, off, ids = z["stamp"], z["trk_offsets"], z["trk_id"]
        parts = split_passes(rows)
        per = [check_pass(p, stamps, off, ids, partial=len(parts) > 1 and i == len(parts) - 1)
               for i, p in enumerate(parts)]
        report.update(replay_frames=int(len(stamps)), passes=len(per))
        for key in ("matched_frames", "unmatched_messages", "missing_frames", "same_ids_frames"):
            report[key] = int(sum(p[key] for p in per))
        report["received_gap_max"] = max(p["received_gap_max"] for p in per)    # [s] between messages of a pass
        if len(per) > 1:
            report["per_pass"] = per
        report["passed"] = report["missing_frames"] == 0 and report["unmatched_messages"] == 0
    for r in rows:
        del r["msg"]
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(report, rows=rows), f, indent=1)
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
