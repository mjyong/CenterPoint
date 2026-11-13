"""Where a recorded bag runs on the site maps, before replaying it through the planning stack: its INS track
(INSPVAX, turned into the map frame the way lightning does it), how much of it is on the route network, which
crosswalks it passes or crosses, and a goal for replay_planning_check.sh. Also counts perception's objects in the
bag (/perception/lidar_3d/objects, recorded on the dog) and the vehicles in them.

    python Nav_D20/planning/tools/bag_on_map.py --bag /public/zjst/robot/cp_data/1112_t40 \
        --map data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13 --map Nav_D20/map/a70c6db6-....zip

Reads db3 and mcap bags (rosbags); the INSPVAX and LidarObjectArray types come from this repository's message
files. The map origin is system.ins_lat / ins_lon / ins_height of the lightning config (--loc-config).
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

INSPVAX = "/rte/rdap/rtk/inspvax"
OBJECTS = "/perception/lidar_3d/objects"
KYGI_MSGS = os.path.join(ROOT, "kygi665_msgs", "msg")
PERCEPTION_MSGS = os.path.join(ROOT, "Nav_D20", "seres_interface", "ros", "perception_interface", "msg")
LOC_CONFIG = os.path.join(ROOT, "Nav_D20", "slam", "lightning-lm-master", "config", "default_d20.yaml")


def typestore():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for folder, pkg in ((KYGI_MSGS, "kygi665_msgs"), (PERCEPTION_MSGS, "perception_interface")):
        for f in sorted(glob.glob(os.path.join(folder, "*.msg"))):
            name = "%s/msg/%s" % (pkg, os.path.splitext(os.path.basename(f))[0])
            text = open(f, encoding="utf-8").read()
            text = text.replace("kygi665_msgs/", "kygi665_msgs/msg/") if pkg == "kygi665_msgs" else text
            try:
                types.update(get_types_from_msg(text, name))
            except Exception as e:   # noqa: BLE001  (a message this tool does not need)
                print("  (type %s not registered: %s)" % (name, e))
    ts.register(types)
    return ts


def read_bag(path, ts):
    from rosbags.highlevel import AnyReader
    ins, objs = [], []
    from pathlib import Path
    with AnyReader([Path(path)], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic in (INSPVAX, OBJECTS)]
        for conn, t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if conn.topic == INSPVAX:
                ins.append((t * 1e-9, m.latitude, m.longitude, m.height, int(m.ins_status.data),
                            int(m.pos_type.data)))
            else:
                vehicles = sum(1 for o in m.objects if o.label == 0)
                moving = sum(1 for o in m.objects if o.label == 0 and o.motion == 1)
                near = min([math.hypot(o.position.x, o.position.y) for o in m.objects if o.label == 0] or [np.inf])
                objs.append((t * 1e-9, len(m.objects), vehicles, moving, near))
    return np.array(ins, dtype=float), objs


def along(xy):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def polygon_distance(xy, poly):
    """(N,) distance from each point to the polygon's outline (its edges, not just the corners: a 32 m crosswalk
    passed at its middle is 16 m from every corner)."""
    a = np.asarray(poly, float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = np.einsum("nkd,kd->nk", xy[:, None, :] - a[None], ab) / np.maximum(np.einsum("kd,kd->k", ab, ab), 1e-12)
    foot = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.min(np.hypot(*(xy[:, None, :] - foot).transpose(2, 0, 1)), axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--map", action="append", default=[], help="map directory / zip / map.yaml (repeat)")
    ap.add_argument("--loc-config", default=LOC_CONFIG)
    ap.add_argument("--near", type=float, default=15.0, help="[m] crosswalks this close to the track are listed")
    args = ap.parse_args(argv)

    from loc_viz import lla_to_enu, read_loc_origin
    from planning_mcap_viz import load_map, read_crosswalks

    origin = read_loc_origin(open(args.loc_config, encoding="utf-8", errors="replace").read())
    if origin is None:
        raise SystemExit("%s: no system.ins_lat / ins_lon / ins_height" % args.loc_config)
    ts = typestore()
    ins, objs = read_bag(args.bag, ts)
    if len(ins) == 0:
        raise SystemExit("%s: no %s" % (args.bag, INSPVAX))
    good = (ins[:, 4] == 3)
    enu = lla_to_enu(ins[:, 1], ins[:, 2], ins[:, 3], origin)
    xy = enu[good][:, :2][::10]              # 10 Hz
    s = along(xy)
    print("%s: %d INSPVAX (%.0f%% good, %.0f%% RTK fixed), %.0f s; track in the map frame (origin %.6f, %.6f): "
          "start (%.1f, %.1f), end (%.1f, %.1f), x %.0f..%.0f, y %.0f..%.0f, %.0f m long" % (
              args.bag, len(ins), 100 * good.mean(), 100 * np.mean(ins[:, 5] == 56), ins[-1, 0] - ins[0, 0],
              origin[0], origin[1], *xy[0], *xy[-1], xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max(),
              s[-1]))
    if objs:
        o = np.array(objs, dtype=float)
        print("%s: %d messages; vehicles per message: max %d, frames with a vehicle %.0f%%, with a moving one %.0f%%;"
              " nearest vehicle: median %.1f m, min %.1f m" % (
                  OBJECTS, len(o), o[:, 2].max(), 100 * np.mean(o[:, 2] > 0), 100 * np.mean(o[:, 3] > 0),
                  np.median(o[np.isfinite(o[:, 4]), 4]) if np.isfinite(o[:, 4]).any() else float("nan"),
                  np.nanmin(np.where(np.isfinite(o[:, 4]), o[:, 4], np.nan)) if np.isfinite(o[:, 4]).any()
                  else float("nan")))
    else:
        print("%s: not in the bag (the yield gate will see no vehicles: it waits at every crosswalk)" % OBJECTS)

    for mp in args.map:
        meta = load_map(mp)
        img, res = meta["img"], meta["resolution"]
        ox, oy = meta["origin"][0], meta["origin"][1]
        h, w = img.shape
        occ = img / 255.0 if meta["negate"] else (255.0 - img.astype(float)) / 255.0
        free = occ < meta["free_thresh"]
        cols = np.floor((xy[:, 0] - ox) / res).astype(int)
        rows = h - 1 - np.floor((xy[:, 1] - oy) / res).astype(int)
        inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        on = np.zeros(len(xy), bool)
        for i in np.nonzero(inside)[0]:          # within 0.5 m of a route cell
            r0, c0 = rows[i], cols[i]
            win = free[max(0, r0 - 5):r0 + 6, max(0, c0 - 5):c0 + 6]
            on[i] = win.any()
        print("\nmap %s: %.0f%% of the track inside the map, %.0f%% within 0.5 m of a route cell" % (
            meta["name"], 100 * inside.mean(), 100 * on.mean()))
        cws = read_crosswalks(meta["osm"]) if meta.get("osm") else []
        from dog_perception.yield_gate import inside_polygon
        crossed = []
        for rid, poly in cws:
            d = polygon_distance(xy, poly)
            ins_poly = inside_polygon(xy, poly)
            if d.min() > args.near and not ins_poly.any():
                continue
            k = int(np.argmin(d))
            print("  crosswalk %s (centre %.1f, %.1f): track %s, closest to its edge %.1f m at %.0f m along it" % (
                rid, *poly.mean(0), "CROSSES it (%.0f..%.0f m along)" % (s[ins_poly].min(), s[ins_poly].max())
                if ins_poly.any() else "passes by", d.min(), s[k]))
            if ins_poly.any():
                crossed.append((rid, s[ins_poly].max()))
        if not cws:
            print("  no crosswalks in this map (no semantic.osm)")
        # a goal: on the route, past the last crosswalk crossed (else the farthest on-route point along the track)
        cand = np.nonzero(on)[0]
        if len(cand):
            if crossed:
                past = max(sc for _, sc in crossed)
                after = [i for i in cand if s[i] > past + 3.0]
                gi = after[min(len(after) - 1, 20)] if after else cand[-1]
            else:
                gi = cand[np.argmax(s[cand])]
            print("  goal for replay_planning_check.sh: %.1f %.1f (%.0f m along the track%s)" % (
                xy[gi, 0], xy[gi, 1], s[gi], ", past crosswalk %s" % crossed[-1][0] if crossed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
