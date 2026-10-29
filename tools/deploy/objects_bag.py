"""What the C++ node published: a rosbag2 of ``~/objects`` (dog_perception_msgs/ObjectArray, recorded with
``ros2 bag record``), summarized and checked against a replay of the same input (dogperc_bag_replay or
dogperc_replay --messages, their .npz): one message per frame (none dropped), the same stamps, the same
track ids in each frame.

    python tools/deploy/objects_bag.py --bag <recorded bag> --replay <replay.npz> --out objects.json

The message definitions are read from deploy/ros2/dog_perception_msgs/msg (rosbags, no ROS install).
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


def typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name in NAMES:
        with open(os.path.join(MSG_DIR, name + ".msg")) as f:
            types.update(get_types_from_msg(f.read(), "dog_perception_msgs/msg/" + name))
    ts.register(types)
    return ts


def read_objects(bag, topic=None):
    from rosbags.highlevel import AnyReader

    rows = []
    with AnyReader([Path(bag)], default_typestore=typestore()) as r:
        conns = [c for c in r.connections if c.msgtype == "dog_perception_msgs/msg/ObjectArray" and
                 (topic is None or c.topic == topic)]
        if not conns:
            raise SystemExit("%s: no dog_perception_msgs/msg/ObjectArray topic" % bag)
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            rows.append(dict(stamp=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, received=t * 1e-9,
                             frame=m.header.frame_id, ids=[int(o.id) for o in m.objects],
                             paths=sum(len(o.predictions) for o in m.objects), msg=m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="recorded ~/objects")
    ap.add_argument("--topic", default=None, help="default: every ObjectArray topic in the bag")
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
        by_stamp = {round(float(s), 6): k for k, s in enumerate(stamps)}
        matched = [(r, by_stamp.get(round(r["stamp"], 6))) for r in rows]
        same_ids = sum(1 for r, k in matched if k is not None and r["ids"] == list(ids[off[k]:off[k + 1]]))
        report.update(replay_frames=int(len(stamps)), matched_frames=sum(k is not None for _, k in matched),
                      unmatched_messages=sum(k is None for _, k in matched),
                      missing_frames=int(len(stamps)) - sum(k is not None for _, k in matched),
                      same_ids_frames=same_ids)
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
