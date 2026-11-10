"""Draw the T44 base yield-gate result: robot fixed, 8 crossing paths, labels and tracks.

Paths turn red while an unsafe crossing is underway and orange while a risky one is.
Time is concatenated so the 32 scenes play one after another. Frame is the detection
frame of each scene's first pose: x right, y forward.
"""
import csv
import os
import pickle
from collections import defaultdict

import numpy as np

from dog_perception import foxglove as fx
from dog_perception.yield_eval import gt_sample, gt_tracks, track_objects

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
CSV = os.path.join(REPO, "data/reports/T44/episodes_base.csv")
OUT = os.path.join(ROOT, "yield_base.mcap")
INDEX = os.path.join(ROOT, "scenes.txt")
CROSS = 8.5          # s, 8 m path + 0.5 m of the robot at 1 m/s
DT = 0.5


def to_det(xy, R, origin):
    return (R.T @ (np.asarray(xy, float) - origin).T).T


def yaw_det(yaw, dyaw):
    return float(yaw - dyaw)


def windows(rows):
    """path -> list of (t_rel, t_rel_end, kind) with kind unsafe overriding risky."""
    out = defaultdict(list)
    for r in rows:
        if r["go"] == "" or r["known"] != "1":
            continue
        if r["unsafe"] != "1" and r["risky"] != "1":
            continue
        t = float(r["arrive"]) + float(r["go"])
        out[r["path"]].append((t, t + CROSS, "unsafe" if r["unsafe"] == "1" else "risky"))
    return out


def active(spans, t):
    kind = None
    for a, b, k in spans:
        if a <= t <= b:
            if k == "unsafe":
                return "unsafe"
            kind = "risky"
    return kind


def main():
    by_scene = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_scene[r["scene"]].append(r)
    scenes = sorted(by_scene)
    paths = {}
    # names in the same order the eval used
    from dog_perception.yield_eval import fan_paths
    paths = fan_paths(8, 8.0, 0.0)

    w = fx.McapWriter(OUT)
    cursor = 1_700_000_000.0
    index = []
    for scene in scenes:
        labels = pickle.load(open(os.path.join(REPO, "data/labels", scene, "labels_sustech2.pkl"), "rb"))
        results = pickle.load(open(os.path.join(REPO, "data/runs_online", scene, "pillar/results.pkl"), "rb"))
        good = [r for r in results if not r.get("pose_bad")]
        if not good or not labels["frames"]:
            continue
        T0 = np.asarray(good[0]["T_world_det"])
        R, origin = T0[:2, :2], T0[:2, 3]
        dyaw = float(np.arctan2(R[1, 0], R[0, 0]))
        frames = track_objects(results, 3.0)
        tracks = gt_tracks(labels, ("vehicle", "cyclist"))
        t0 = frames[0][0]
        t1 = frames[-1][0]
        vis0 = cursor
        index.append((scene, vis0, vis0 + (t1 - t0)))
        spans = windows(by_scene[scene])
        w.write("/log", "foxglove.Log", vis0, fx.log(vis0, scene, name="scene"))

        stamps = np.array([st for st, _ in frames])
        times = np.arange(t0, t1 + 1e-6, DT)
        for stamp in times:
            k = int(np.argmin(np.abs(stamps - stamp)))
            rel = float(stamp - t0)
            vis = vis0 + rel
            ents = []
            n_bad = n_risk = 0
            lines, texts = [], []
            for name, seg in paths.items():
                kind = active(spans[name], rel)
                if kind == "unsafe":
                    rgb, n_bad = (0.9, 0.15, 0.15, 1), n_bad + 1
                elif kind == "risky":
                    rgb, n_risk = (0.95, 0.55, 0.1, 1), n_risk + 1
                else:
                    rgb = (0.25, 0.75, 0.35, 0.85)
                p = np.column_stack([seg, np.zeros(len(seg))])
                lines.append(fx.line(p, rgb, 0.18 if kind else 0.06))
                texts.append(fx.text((*seg[-1], 0.4), name[3:], rgb, 12))
            # blind sector, 142-218 deg from forward
            wedge = [np.zeros(3)]
            for deg in (142, 218):
                a = np.radians(deg)
                wedge.append([-8 * np.sin(a), 8 * np.cos(a), 0])
            lines.append(fx.line(wedge, (0.45, 0.45, 0.5, 0.5), 0.04))
            ents.append(fx.entity("paths", vis, "det", 0.6, lines=lines, texts=texts))
            dog = np.array([0, 0, 0.4, 1.0, 0.4, 0.7, np.pi / 2])
            ents.append(fx.entity("dog", vis, "det", 0.6, lines=[fx.box_wire(dog, (0.2, 0.9, 0.4, 1), 0.05)],
                                  texts=[fx.text((0, -1.2, 1.2),
                                                 "%s  红 %d 条正在不安全通过  橙 %d 条偏冒险" % (scene, n_bad, n_risk),
                                                 (1, 1, 1, 1), 16)]))
            w.write("/yield/paths", "foxglove.SceneUpdate", vis, fx.scene_update(ents))

            boxes = []
            for tid, tr in tracks.items():
                xy, yaw, speed, valid = gt_sample(tr, [stamp], 1.1)
                if not valid[0]:
                    continue
                p = to_det(xy[0], R, origin)
                box = np.array([p[0], p[1], 0.8, tr["l"], tr["w"], 1.5, yaw_det(yaw[0], dyaw)])
                boxes.append(fx.entity("gt-%d" % tid, vis, "det", 0.6,
                                       lines=[fx.box_wire(box, (0.3, 0.55, 1, 1), 0.04)],
                                       texts=[fx.text((p[0], p[1], 2.2), "GT %d %.0fkm/h" % (tid, speed[0] * 3.6),
                                                      (0.6, 0.75, 1, 1), 11)]))
            w.write("/yield/labels", "foxglove.SceneUpdate", vis, fx.scene_update(boxes))

            trk = []
            _, objs = frames[k]
            for o in objs:
                if o["label"] not in ("vehicle", "cyclist"):
                    continue
                p = to_det(o["position"][:2], R, origin)
                v = R.T @ np.asarray(o["velocity"][:2], float)
                spd = float(np.hypot(*v))
                yaw = float(np.arctan2(v[1], v[0])) if spd > 0.4 else yaw_det(o["yaw"], dyaw)
                box = np.array([p[0], p[1], 0.9, o["size"][0], o["size"][1], 1.5, yaw])
                trk.append(fx.entity("tr-%d" % o["id"], vis, "det", 0.6,
                                     lines=[fx.box_wire(box, (1, 0.65, 0.15, 1), 0.05)],
                                     arrows=[fx.arrow((p[0], p[1], 1.6), (v[0], v[1], 0), (1, 0.65, 0.15, 1), 0.08)],
                                     texts=[fx.text((p[0], p[1], 2.6), "trk %d %s %.0fkm/h" % (
                                         o["id"], o["motion"], spd * 3.6), (1, 0.8, 0.4, 1), 11)]))
            w.write("/yield/tracks", "foxglove.SceneUpdate", vis, fx.scene_update(trk))
            w.write("/yield/counts", "dog.YieldCounts", vis,
                    {"unsafe_paths": n_bad, "risky_paths": n_risk, "labels": len(boxes), "tracks": len(trk)})
        cursor = vis0 + (t1 - t0) + 2.0
        print(scene, "vis", vis0, "dur", round(t1 - t0, 1))

    w.close()
    with open(INDEX, "w") as f:
        f.write("scene start_s end_s   (Foxglove time is unix; subtract %d)\n" % 1700000000)
        for name, a, b in index:
            f.write("%s %.1f %.1f\n" % (name, a - 1700000000, b - 1700000000))
    print("wrote", OUT, os.path.getsize(OUT), "messages", w.messages)


if __name__ == "__main__":
    main()
