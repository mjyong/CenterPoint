"""Check the chain perception -> map that the crosswalk gate uses (CrosswalkYield::ObjectsInMap): perception's
objects (/perception/lidar_3d/objects, in its own frame: on the ground under the lidar, x right, y forward) are put
in base_link by a fixed extrinsic (crosswalk_yield_conf det_to_base_*) and in the map by the odometry at the
message's stamp (/lightning/odom). A parked car has to stay where it is in the map while the dog walks and turns;
with a wrong extrinsic yaw it swings around the dog. So, for every vehicle track perception calls static or
stopped, this measures how far its map position wanders, for the configured extrinsic and for the yaw turned by
90, 180 and 270 deg: the configured one should be clearly the smallest.

    python Nav_D20/planning/tools/objects_in_map.py work_dirs/planning_io_t53_1112 [--odom /lightning/odom]

Reads db3 or mcap bags (a replay_planning_check.sh recording, or a bag recorded on the dog with both topics).
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OBJECTS = "/perception/lidar_3d/objects"
# crosswalk_yield.h / mission_conf.proto defaults (D20 A2, T46)
DET_TO_BASE = (0.319, 0.043, -90.0)


def read(path, odom_topic):
    from pathlib import Path

    from bag_on_map import typestore
    from rosbags.highlevel import AnyReader
    odom, frames = [], []
    with AnyReader([Path(path)], default_typestore=typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in (odom_topic, OBJECTS)]
        if not any(c.topic == OBJECTS for c in conns):
            raise SystemExit("%s: no %s" % (path, OBJECTS))
        if not any(c.topic == odom_topic for c in conns):
            raise SystemExit("%s: no %s" % (path, odom_topic))
        for conn, _t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            st = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
            if conn.topic == odom_topic:
                q = m.pose.pose.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                odom.append((st, m.pose.pose.position.x, m.pose.pose.position.y, yaw))
            else:
                frames.append((st, [(int(o.id), int(o.label), int(o.motion), float(o.position.x), float(o.position.y))
                                    for o in m.objects]))
    odom = np.array(sorted(odom), dtype=float)
    return odom, sorted(frames, key=lambda f: f[0])


def pose_at(odom, t, max_gap):
    """(x, y, yaw) of the odometry at t, interpolated; None further than max_gap from the nearest sample (stale)."""
    k = np.searchsorted(odom[:, 0], t)
    near = min(abs(odom[min(k, len(odom) - 1), 0] - t), abs(odom[max(k - 1, 0), 0] - t))
    if near > max_gap:
        return None
    yaw = np.unwrap(odom[:, 3])
    return (float(np.interp(t, odom[:, 0], odom[:, 1])), float(np.interp(t, odom[:, 0], odom[:, 2])),
            float(np.interp(t, odom[:, 0], yaw)))


def to_map(pose, det_xy, det_to_base):
    x, y, yaw = pose
    dx, dy, ydb = det_to_base
    c, s = math.cos(yaw), math.sin(yaw)
    cd, sd = math.cos(math.radians(ydb)), math.sin(math.radians(ydb))
    bx, by = dx + cd * det_xy[0] - sd * det_xy[1], dy + sd * det_xy[0] + cd * det_xy[1]
    return x + c * bx - s * by, y + s * bx + c * by


def check(odom, frames, det_to_base=DET_TO_BASE, max_gap=0.3, min_frames=20, max_range=40.0):
    """-> dict: per candidate yaw (the configured one first) the median wander [m] of static vehicle tracks, the
    tracks used and how much the dog moved and turned while they were seen."""
    yaws = [det_to_base[2] + k * 90.0 for k in range(4)]
    tracks = {}            # id -> [(t, pose, det_xy)]
    skipped = 0
    for t, objs in frames:
        pose = None
        for oid, label, motion, px, py in objs:
            if label != 0 or motion == 1 or math.hypot(px, py) > max_range:
                continue
            if pose is None:
                pose = pose_at(odom, t, max_gap)
                if pose is None:
                    skipped += 1
                    break
            tracks.setdefault(oid, []).append((t, pose, (px, py)))
    used, dog_move, dog_turn = [], [], []
    wander = {y: [] for y in yaws}
    for oid, obs in tracks.items():
        if len(obs) < min_frames:
            continue
        poses = np.array([p for _, p, _ in obs])
        move = float(np.hypot(*(poses[:, :2].max(0) - poses[:, :2].min(0))))
        turn = math.degrees(float(poses[:, 2].max() - poses[:, 2].min()))
        if move < 1.0 and turn < 15.0:      # the dog stood still: any extrinsic keeps a parked car still
            continue
        used.append(oid)
        dog_move.append(move)
        dog_turn.append(turn)
        for y in yaws:
            pts = np.array([to_map(p, d, (det_to_base[0], det_to_base[1], y)) for _, p, d in obs])
            wander[y].append(float(np.median(np.hypot(*(pts - np.median(pts, axis=0)).T))))
    return dict(yaws=yaws, wander={y: (float(np.median(v)) if v else float("nan")) for y, v in wander.items()},
                tracks=used, dog_move=dog_move, dog_turn=dog_turn, frames=len(frames), stale=skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="a recording directory (replay_planning_check.sh: work_dirs/planning_io_<name>)")
    ap.add_argument("--odom", default="/lightning/odom")
    ap.add_argument("--det-to-base", type=float, nargs=3, default=DET_TO_BASE, metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args(argv)
    path = args.bag
    if os.path.isdir(path) and not os.path.isfile(os.path.join(path, "metadata.yaml")):
        subs = sorted(d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d, "metadata.yaml")))
        if subs:                              # a replay directory: the main recording in it
            path = os.path.join(path, subs[0])
    odom, frames = read(path, args.odom)
    r = check(odom, frames, tuple(args.det_to_base), min_frames=args.min_frames)
    print("%s: %d object messages, %d odometry; %d messages with no odometry within 0.3 s (the gate counts them "
          "stale)" % (path, r["frames"], len(odom), r["stale"]))
    if not r["tracks"]:
        print("no static vehicle track seen for %d+ frames while the dog moved >= 1 m or turned >= 15 deg: "
              "nothing to check" % args.min_frames)
        return 1
    print("%d static vehicle tracks (the dog moved median %.1f m, turned median %.0f deg while each was seen)" % (
        len(r["tracks"]), np.median(r["dog_move"]), np.median(r["dog_turn"])))
    best = min(r["yaws"], key=lambda y: r["wander"][y])
    for y in r["yaws"]:
        print("  det -> base_link yaw %6.1f deg: parked cars wander median %.2f m in the map%s%s" % (
            y, r["wander"][y], "  (configured)" if y == r["yaws"][0] else "", "  <- smallest" if y == best else ""))
    ok = best == r["yaws"][0]
    print("extrinsic %s" % ("OK: the configured yaw keeps parked cars still" if ok else
                            "WRONG: another yaw keeps parked cars stiller than the configured one"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
