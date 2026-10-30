"""What the C++ node published: a rosbag2 of ``~/objects`` (dog_perception_msgs/ObjectArray, recorded with
``ros2 bag record``), summarized and checked against a replay of the same input (dogperc_bag_replay or
dogperc_replay --messages, their .npz): one message per frame (none dropped), the same stamps, the same
track ids in each frame.

    python tools/deploy/objects_bag.py --bag <recorded bag> --replay <replay.npz> --out objects.json

A bag played in a loop (``ros2 bag play --loop``) gives one pass per loop, split where the stamps jump back;
each pass is checked against the replay, the last one up to its last frame when the recording stopped
during it (more than one pass).

With /tf in the recording (the node's publish_tf), its world -> det frame transforms are checked against the
messages' world_pose at the same stamps (tf_pose_max_diff), and the frames it names are listed.

same_ids_frames compares the id lists, which one extra tentative track early in a pass shifts for the rest of
it (the message order online is not quite the bag's: T31); same_tracks_frames compares the reported tracks
without their ids, by class and world position (within TRACK_TOL).

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
TRACK_TOL = 0.5          # [m]


def typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for package, msg_dir, names in (("dog_perception_msgs", MSG_DIR, NAMES),
                                    ("perception_interface", DEB_MSG_DIR, DEB_NAMES)):
        if not os.path.isdir(msg_dir):
            continue
        for name in names:
            with open(os.path.join(msg_dir, name + ".msg"), encoding="utf-8") as f:
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
                             paths=sum(len(o.predictions) for o in m.objects), msg=m,
                             tracks=world_tracks(m)))
    return rows


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_tracks(m):
    """[(label, x, y)] of the objects in the world frame (the message's world_pose)."""
    q, p = m.world_pose.orientation, m.world_pose.position
    R, t = quat_to_R(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])
    return [(int(o.label),) + tuple((R @ [o.position.x, o.position.y, o.position.z] + t)[:2]) for o in m.objects]


def same_tracks(live, labels, xy, tol=TRACK_TOL):
    """The same reported tracks, ids aside: as many, each matched once to one of its class within tol."""
    if len(live) != len(labels):
        return False
    free = list(range(len(labels)))
    for label, x, y in live:
        d = [(np.hypot(x - xy[j][0], y - xy[j][1]), j) for j in free if labels[j] == label]
        if not d or min(d)[0] > tol:
            return False
        free.remove(min(d)[1])
    return True


def read_tf(bag):
    """{(parent, child): {stamp: 4x4}} of the bag's /tf, empty without it."""
    from rosbags.highlevel import AnyReader

    out = {}
    with AnyReader([Path(bag)], default_typestore=typestore()) as r:
        conns = [c for c in r.connections if c.msgtype == "tf2_msgs/msg/TFMessage" and c.topic == "/tf"]
        if not conns:            # messages(connections=[]) would read every topic
            return out
        for c, _, raw in r.messages(connections=conns):
            for t in r.deserialize(raw, c.msgtype).transforms:
                q, p = t.transform.rotation, t.transform.translation
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = quat_to_R(q.x, q.y, q.z, q.w), [p.x, p.y, p.z]
                stamp = round(t.header.stamp.sec + t.header.stamp.nanosec * 1e-9, 6)
                out.setdefault((t.header.frame_id, t.child_frame_id), {})[stamp] = T
    return out


def tf_report(tf, rows):
    """The world -> det transforms against the messages' world_pose at the same stamps."""
    diffs, found = [], 0
    for r in rows:
        m = r["msg"]
        T = tf.get((m.world_frame, m.header.frame_id), {}).get(round(r["stamp"], 6))
        if T is None:
            continue
        found += 1
        q, p = m.world_pose.orientation, m.world_pose.position
        W = np.eye(4)
        W[:3, :3], W[:3, 3] = quat_to_R(q.x, q.y, q.z, q.w), [p.x, p.y, p.z]
        diffs.append(np.abs(T - W).max())
    return dict(tf_transforms=sorted("%s -> %s (%d)" % (a, b, len(v)) for (a, b), v in tf.items()),
                tf_frames_with_pose=found, tf_pose_max_diff=float(max(diffs)) if diffs else None)


def split_passes(rows, jump_back=1.0):
    """Rows in recording order -> one list per pass of the bag (a stamp more than ``jump_back`` s before the
    previous one starts a pass: the node starts over there, dogperc/online.hpp)."""
    out = []
    for r in rows:
        if not out or r["stamp"] < out[-1][-1]["stamp"] - jump_back:
            out.append([])
        out[-1].append(r)
    return out


def check_pass(rows, stamps, offsets, ids, partial=False, labels=None, positions=None):
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
                                    and r["ids"] == list(ids[offsets[k]:offsets[k + 1]])),
                same_tracks_frames=None if labels is None else sum(
                    1 for r, k in matched if k is not None and "tracks" in r and same_tracks(
                        r["tracks"], labels[offsets[k]:offsets[k + 1]], positions[offsets[k]:offsets[k + 1]])),
                partial=partial)


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
        labels = z["trk_label"] if "trk_label" in z.files else None
        positions = z["trk_position"][:, :2] if labels is not None else None
        parts = split_passes(rows)
        per = [check_pass(p, stamps, off, ids, partial=len(parts) > 1 and i == len(parts) - 1, labels=labels,
                          positions=positions) for i, p in enumerate(parts)]
        report.update(replay_frames=int(len(stamps)), passes=len(per))
        for key in ("matched_frames", "unmatched_messages", "missing_frames", "same_ids_frames"):
            report[key] = int(sum(p[key] for p in per))
        if labels is not None:
            report["same_tracks_frames"] = int(sum(p["same_tracks_frames"] for p in per))
        report["received_gap_max"] = max(p["received_gap_max"] for p in per)    # [s] between messages of a pass
        if len(per) > 1:
            report["per_pass"] = per
        report["passed"] = report["missing_frames"] == 0 and report["unmatched_messages"] == 0
    tf = read_tf(args.bag)
    if tf:
        report.update(tf_report(tf, rows))
    for r in rows:
        del r["msg"]
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(report, rows=rows), f, indent=1)
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
