"""Draw a yield-gate result so each labelled car keeps its own colour, and the car
the eval says the robot would meet is marked.

A path is red while an unsafe crossing is underway and orange while a risky one is.
The labelled track in ``gap_track`` for that crossing is drawn thicker, with a line
to the path and a label naming the track id, the path and the cause. Other cars stay
their own colour. Tracker boxes use a second palette and say which label they sit on.
Time is concatenated so the 32 scenes play one after another. Frame is the detection
frame of each scene's first pose: x right, y forward.
"""
import csv
import os
from collections import defaultdict

import numpy as np

from dog_perception import foxglove as fx
from dog_perception.yield_eval import fan_paths, gt_sample, gt_tracks, track_objects

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT_DIR = os.path.join(REPO, "work_dirs/yield_t45")
CSV = os.path.join(REPO, "data/reports/T45/episodes_base.csv")
OUT = os.path.join(OUT_DIR, "yield_base.mcap")
INDEX = os.path.join(OUT_DIR, "scenes.txt")
CROSS = 8.5          # s, 8 m path + 0.5 m of the robot at 1 m/s
DT = 0.5
CAUSE = {"blind": "盲区", "far": "远处", "near": "近处", "occluded": "挡住",
         "detection": "没跟上", "prediction": "没预测到"}


def to_det(xy, R, origin):
    return (R.T @ (np.asarray(xy, float) - origin).T).T


def yaw_det(yaw, dyaw):
    return float(yaw - dyaw)


def hue_rgb(i, sat=0.72, val=0.95, alpha=1.0):
    """Stable colour for an integer id. Neighbouring ids land far apart on the wheel."""
    h = (int(i) * 0.618033988749895) % 1.0
    i6 = int(h * 6.0)
    f = h * 6.0 - i6
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    r, g, b = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)][i6 % 6]
    return (r, g, b, alpha)


def closest_on_seg(p, seg):
    a, b = np.asarray(seg[0], float), np.asarray(seg[-1], float)
    ab = b - a
    den = float(np.dot(ab, ab))
    u = 0.0 if den < 1e-9 else float(np.clip(np.dot(np.asarray(p[:2]) - a, ab) / den, 0.0, 1.0))
    return a + u * ab


def threats_at(rows, rel):
    """gap_track id -> {kind, paths, causes} for crossings underway at ``rel`` (scene time)."""
    found = {}
    for r in rows:
        if r["go"] == "" or r["known"] != "1" or (r["unsafe"] != "1" and r["risky"] != "1"):
            continue
        t = float(r["arrive"]) + float(r["go"])
        if not (t <= rel <= t + CROSS):
            continue
        tid = int(float(r["gap_track"])) if r["gap_track"] not in ("", "-1") else -1
        slot = found.setdefault(tid, {"kind": "risky", "paths": [], "causes": []})
        if r["unsafe"] == "1":
            slot["kind"] = "unsafe"
        if r["path"] not in slot["paths"]:
            slot["paths"].append(r["path"])
        cause = r.get("cause") or ""
        if cause and cause not in slot["causes"]:
            slot["causes"].append(cause)
    return found


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_scene = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_scene[r["scene"]].append(r)
    scenes = sorted(by_scene)
    paths = fan_paths(8, 8.0, 0.0)

    w = fx.McapWriter(OUT)
    cursor = 1_700_000_000.0
    index = []
    for scene in scenes:
        labels = pickle_load(os.path.join(REPO, "data/labels", scene, "labels_sustech2.pkl"))
        results = pickle_load(os.path.join(REPO, "data/runs_online", scene, "pillar/results.pkl"))
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
        rows = by_scene[scene]
        w.write("/log", "foxglove.Log", vis0, fx.log(vis0, scene, name="scene"))

        stamps = np.array([st for st, _ in frames])
        for stamp in np.arange(t0, t1 + 1e-6, DT):
            k = int(np.argmin(np.abs(stamps - stamp)))
            rel = float(stamp - t0)
            vis = vis0 + rel
            hot = threats_at(rows, rel)
            ents, lines, texts = [], [], []
            for name, seg in paths.items():
                owners = [(tid, h) for tid, h in hot.items() if name in h["paths"]]
                kind = "unsafe" if any(h["kind"] == "unsafe" for _, h in owners) else (
                    "risky" if owners else None)
                rgb = (0.92, 0.15, 0.15, 1) if kind == "unsafe" else (
                    (0.95, 0.55, 0.1, 1) if kind == "risky" else (0.25, 0.75, 0.35, 0.7))
                p = np.column_stack([seg, np.zeros(len(seg))])
                lines.append(fx.line(p, rgb, 0.18 if kind else 0.06))
                who = " ".join("GT%d" % tid for tid, _ in owners if tid >= 0)
                texts.append(fx.text((*seg[-1], 0.4), (name[3:] + (" " + who if who else "")), rgb, 12))
            wedge = [np.zeros(3)]
            for deg in (142, 218):
                a = np.radians(deg)
                wedge.append([-8 * np.sin(a), 8 * np.cos(a), 0])
            lines.append(fx.line(wedge, (0.45, 0.45, 0.5, 0.5), 0.04))
            ents.append(fx.entity("paths", vis, "det", 0.6, lines=lines, texts=texts))

            banner = scene
            for tid, h in sorted(hot.items()):
                if tid < 0:
                    continue
                word = "碰撞" if h["kind"] == "unsafe" else "冒险"
                why = "、".join(CAUSE.get(c, c) for c in h["causes"])
                fans = "/".join(n[3:] for n in h["paths"])
                banner += "   %s GT%d→%s%s" % (word, tid, fans, (" " + why) if why else "")
            if not any(tid >= 0 for tid in hot):
                banner += "   这一帧没有正在通过的风险车"
            dog = np.array([0, 0, 0.4, 1.0, 0.4, 0.7, np.pi / 2])
            ents.append(fx.entity("dog", vis, "det", 0.6, lines=[fx.box_wire(dog, (0.2, 0.9, 0.4, 1), 0.05)],
                                  texts=[fx.text((0, -1.6, 1.4), banner, (1, 1, 1, 1), 16)]))
            w.write("/yield/paths", "foxglove.SceneUpdate", vis, fx.scene_update(ents))

            boxes, gt_pos = [], {}
            for tid, tr in tracks.items():
                xy, yaw, speed, valid = gt_sample(tr, [stamp], 1.1)
                if not valid[0]:
                    continue
                p = to_det(xy[0], R, origin)
                gt_pos[tid] = p
                own = hue_rgb(tid)
                h = hot.get(tid)
                kind = h["kind"] if h else None
                mark = (0.95, 0.12, 0.12, 1) if kind == "unsafe" else (
                    (1.0, 0.5, 0.05, 1) if kind == "risky" else None)
                box = np.array([p[0], p[1], 0.8, tr["l"], tr["w"], 1.5, yaw_det(yaw[0], dyaw)])
                wires = [fx.box_wire(box, own, 0.09 if kind else 0.035)]
                if mark is not None:
                    fat = box.copy()
                    fat[3] += 0.8
                    fat[4] += 0.8
                    wires.append(fx.box_wire(fat, mark, 0.07))
                    for name in h["paths"]:
                        q = closest_on_seg(p, paths[name])
                        wires.append(fx.line([(p[0], p[1], 0.5), (q[0], q[1], 0.3)], mark, 0.05))
                if kind == "unsafe":
                    label = "碰撞 GT%d %s %s" % (
                        tid, "/".join(n[3:] for n in h["paths"]),
                        "、".join(CAUSE.get(c, c) for c in h["causes"]))
                    ink = mark
                elif kind == "risky":
                    label = "冒险 GT%d %s" % (tid, "/".join(n[3:] for n in h["paths"]))
                    ink = mark
                else:
                    label = "GT%d" % tid
                    ink = own
                label += " %.0fkm/h" % (speed[0] * 3.6)
                boxes.append(fx.entity("gt-%d" % tid, vis, "det", 0.6, lines=wires,
                                       texts=[fx.text((p[0], p[1], 2.4 if kind else 2.0), label, ink, 14 if kind else 11)]))
            w.write("/yield/labels", "foxglove.SceneUpdate", vis, fx.scene_update(boxes))

            trk = []
            for o in frames[k][1]:
                if o["label"] not in ("vehicle", "cyclist"):
                    continue
                p = to_det(o["position"][:2], R, origin)
                v = R.T @ np.asarray(o["velocity"][:2], float)
                spd = float(np.hypot(*v))
                yaw = float(np.arctan2(v[1], v[0])) if spd > 0.4 else yaw_det(o["yaw"], dyaw)
                near = [tid for tid, g in gt_pos.items() if np.hypot(p[0] - g[0], p[1] - g[1]) <= 2.5]
                rgb = hue_rgb(o["id"] + 3, sat=0.45, val=1.0)
                note = "trk%d %s %.0fkm/h" % (o["id"], o["motion"], spd * 3.6)
                if near:
                    note += " = " + ",".join("GT%d" % tid for tid in near)
                    if any(tid in hot for tid in near):
                        rgb = (1.0, 0.85, 0.2, 1)
                box = np.array([p[0], p[1], 1.15, o["size"][0], o["size"][1], 1.2, yaw])
                trk.append(fx.entity("tr-%d" % o["id"], vis, "det", 0.6,
                                     lines=[fx.box_wire(box, rgb, 0.045)],
                                     arrows=[fx.arrow((p[0], p[1], 1.7), (v[0], v[1], 0), rgb, 0.07)],
                                     texts=[fx.text((p[0], p[1], 3.1), note, rgb, 11)]))
            w.write("/yield/tracks", "foxglove.SceneUpdate", vis, fx.scene_update(trk))
        cursor = vis0 + (t1 - t0) + 2.0
        print(scene, "vis", round(vis0 - 1700000000, 1))

    w.close()
    with open(INDEX, "w") as f:
        f.write("scene start_s end_s   (Foxglove time is unix; subtract %d)\n" % 1700000000)
        for name, a, b in index:
            f.write("%s %.1f %.1f\n" % (name, a - 1700000000, b - 1700000000))
    print("wrote", OUT, os.path.getsize(OUT), "messages", w.messages)


def pickle_load(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
