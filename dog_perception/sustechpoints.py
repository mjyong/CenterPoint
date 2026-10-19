"""Convert auto-labels to / from a SUSTechPOINTS scene (https://github.com/naurril/SUSTechPOINTS).

SUSTechPOINTS reads ``data/<scene>/lidar/<frame>.pcd`` and
``data/<scene>/label/<frame>.json`` (a list of
``{"obj_id", "obj_type", "obj_attr", "psr": {"position", "scale", "rotation"}}``,
position = box center, scale = (length, width, height), rotation.z = yaw;
see its ``scene_reader.py`` and ``public/js/annotation.js``). Frames are
sorted by file name.

Export writes each labelled frame in the detection frame (gravity aligned,
ground at z ~ 0), point cloud and boxes alike, so no calibration or ego
pose is needed, shifted down by ``LIDAR_HEIGHT`` (1.8 m): SUSTechPOINTS
assumes a lidar 1.8 m above the ground and puts a new box's bottom at
z = -1.8 (``public/js/editor.js``, ``add_box_on_mouse_pos``); without the
shift new boxes land under our ground and cannot snap to the points. Import
shifts back (``z_offset`` in ``dog_meta.json``). Frame names are the stamp
in microseconds (sortable). The track id becomes ``obj_id``; flagged tracks get ``obj_attr = "review"`` so
they stand out in the tool. ``dog_meta.json`` keeps what SUSTechPOINTS does
not: stamps, ``T_world_det`` per frame and the auto-label scores.

Import reads the (corrected) labels back. Classes and vehicle subclasses map
from the SUSTechPOINTS object types (``TYPE_TO_SUBCLASS``: Car / Van -> car,
Truck and construction machines -> truck, Bus / Tram / LongVehicle -> bus; other
types are skipped and counted). Export writes a vehicle's subclass as its type
(``SUBCLASS_TO_TYPE``), Car when it has none. A box
the annotator did not move (center within ``moved_tol``) keeps the exported
auto-label velocity, which already handles static objects (see
dog_perception/autolabel.py); moved or new boxes get a velocity from their
``obj_id``'s annotated centers in the world frame (difference over +-0.5 s,
zero below ``static_speed``). The result has the ``labels.pkl`` layout of
tools/dog/auto_label.py and goes straight into export_dataset.py.
"""
import json
import os
from collections import defaultdict

import numpy as np

from .autolabel import _window_velocity
from .detection.boxes import CLASSES, SUBCLASS_TO_CLASS, Detections
from .geometry import inv_T, wrap_angle

CLASS_TO_TYPE = {"vehicle": "Car", "pedestrian": "Pedestrian", "cyclist": "BicycleRider"}
SUBCLASS_TO_TYPE = {"car": "Car", "truck": "Truck", "bus": "Bus", "pedestrian": "Pedestrian", "cyclist": "BicycleRider"}

# SUSTechPOINTS object types (public/js/obj_cfg.js) -> subclasses, grouped by size the way nuScenes'
# heads are (construction machines with trucks; trams and long vehicles with buses, like trailers).
# Parked bicycles / motorcycles are cyclists, as in the nuScenes mapping (NUSC_TO_SUBCLASS); the
# tracker tells them apart by speed.
TYPE_TO_SUBCLASS = {
    **{t: "car" for t in ("Car", "Van", "PoliceCar", "TourCar")},
    **{t: "truck" for t in ("Truck", "ConcreteTruck", "Excavator", "ForkLift", "Crane", "RoadRoller", "Bulldozer")},
    **{t: "bus" for t in ("Bus", "Tram", "LongVehicle")},
    **{t: "pedestrian" for t in ("Pedestrian", "RoadWorker", "Child")},
    **{t: "cyclist" for t in ("BicycleRider", "MotorcyleRider", "ScooterRider", "Bicycle", "Motorcycle", "Scooter")},
}
TYPE_TO_CLASS = {t: SUBCLASS_TO_CLASS[s] for t, s in TYPE_TO_SUBCLASS.items()}

META_FILE = "dog_meta.json"
LIDAR_HEIGHT = 1.8          # [m] SUSTechPOINTS' assumed lidar height above the ground


def frame_name(stamp):
    return "%016d" % int(round(float(stamp) * 1e6))


def write_pcd(path, xyz, intensity):
    """Binary PCD v0.7 with float fields x y z intensity (what SUSTechPOINTS' PCDLoader reads)."""
    data = np.column_stack([np.asarray(xyz, np.float32).reshape(-1, 3),
                            np.asarray(intensity, np.float32).reshape(-1, 1)]).astype("<f4")
    n = len(data)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n"
              % (n, n))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_pcd(path):
    """(N, 4) x y z intensity from a PCD written by ``write_pcd``."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    head = raw[:end].decode("ascii")
    n = int([line for line in head.splitlines() if line.startswith("POINTS")][0].split()[1])
    return np.frombuffer(raw[end:end + 16 * n], "<f4").reshape(n, 4)


def box_to_psr(box):
    x, y, z, l, w, h, yaw = (float(v) for v in box[:7])
    return {"position": {"x": x, "y": y, "z": z}, "scale": {"x": l, "y": w, "z": h},
            "rotation": {"x": 0.0, "y": 0.0, "z": float(wrap_angle(yaw))}}


def psr_to_box(psr):
    p, s, r = psr["position"], psr["scale"], psr["rotation"]
    return np.array([p["x"], p["y"], p["z"], abs(s["x"]), abs(s["y"]), abs(s["z"]), wrap_angle(r["z"])], np.float64)


def export_scene(labels, frames_dir, out_root, scene, every=1, t_start=None, t_end=None, current_sweep_only=False):
    """labels.pkl + run_sequence.py frames -> ``out_root/scene`` (point ``out_root`` at SUSTechPOINTS' data/).

    ``t_start`` / ``t_end``: seconds since the recording start. Returns the number of frames written."""
    root = os.path.join(out_root, scene)
    os.makedirs(os.path.join(root, "lidar"), exist_ok=True)
    os.makedirs(os.path.join(root, "label"), exist_ok=True)
    t0 = labels["frames"][0]["stamp"] if labels["frames"] else 0.0
    scores = defaultdict(list)
    meta_frames = []
    for k, f in enumerate(labels["frames"]):
        rel = f["stamp"] - t0
        if k % every or (t_start is not None and rel < t_start - 1e-6) or (t_end is not None and rel > t_end + 1e-6):
            continue
        if f.get("pose_bad"):                     # pose not trusted: not in the dataset, nothing to annotate
            continue
        src = os.path.join(frames_dir, "%.6f.npy" % f["stamp"])
        if not os.path.exists(src):
            continue
        pts = np.load(src)
        if current_sweep_only and pts.shape[1] > 4:
            pts = pts[pts[:, 4] <= 1e-6]
        name = frame_name(f["stamp"])
        xyz = pts[:, :3] - [0.0, 0.0, LIDAR_HEIGHT]
        write_pcd(os.path.join(root, "lidar", name + ".pcd"), xyz, np.clip(pts[:, 3] / 255.0, 0, 1))
        objs, exported = [], {}
        for o in f["objects"]:
            box = np.asarray(o["box"], float).copy()
            box[2] -= LIDAR_HEIGHT
            otype = SUBCLASS_TO_TYPE.get(o.get("subclass")) or CLASS_TO_TYPE[o["label"]]
            objs.append({"obj_id": str(o["track_id"]), "obj_type": otype,
                         "obj_attr": "review" if o.get("review") else "", "psr": box_to_psr(box)})
            scores[o["track_id"]].append(float(o["score"]))
            exported[str(o["track_id"])] = dict(center=np.asarray(o["box"][:3], float).tolist(),
                                                velocity=np.asarray(o["velocity"], float).tolist())
        with open(os.path.join(root, "label", name + ".json"), "w") as fh:
            json.dump(objs, fh)
        meta_frames.append(dict(name=name, stamp=float(f["stamp"]),
                                T_world_det=np.asarray(f["T_world_det"], float).tolist(), objects=exported))
    meta = dict(sequence=labels.get("sequence"), frames=meta_frames, class_to_type=CLASS_TO_TYPE,
                current_sweep_only=bool(current_sweep_only), z_offset=-LIDAR_HEIGHT,
                track_scores={str(k): float(np.mean(v)) for k, v in scores.items()})
    with open(os.path.join(root, META_FILE), "w") as fh:
        json.dump(meta, fh, indent=1)
    return len(meta_frames)


def import_scene(scene_dir, type_map=None, static_speed=0.2, velocity_window=0.5, moved_tol=0.05):
    """SUSTechPOINTS scene (exported by ``export_scene``, then corrected) -> (labels dict, counts).

    ``type_map`` adds or overrides object types: type -> subclass (car, truck, ...) or class."""
    type_map = dict(TYPE_TO_SUBCLASS, **(type_map or {}))
    with open(os.path.join(scene_dir, META_FILE)) as fh:
        meta = json.load(fh)
    z_offset = float(meta.get("z_offset", 0.0))
    counts = defaultdict(int)
    frames, by_id = [], defaultdict(list)          # obj_id -> [(frame index, object index)]
    for fm in meta["frames"]:
        path = os.path.join(scene_dir, "label", fm["name"] + ".json")
        anns = []
        if os.path.exists(path):
            with open(path) as fh:
                anns = json.load(fh) or []
        T = np.asarray(fm["T_world_det"], float)
        objs = []
        for a in anns:
            name = type_map.get(a.get("obj_type"))
            if name in SUBCLASS_TO_CLASS:
                cls, sub = SUBCLASS_TO_CLASS[name], name
            elif name in CLASSES:
                cls, sub = name, None if name == "vehicle" else name
            else:
                counts["skipped type %s" % a.get("obj_type")] += 1
                continue
            oid = str(a.get("obj_id", "")).strip()
            if not oid:
                oid = "anon-%s-%d" % (fm["name"], len(objs))
            box = psr_to_box(a["psr"])
            box[2] -= z_offset
            ex = fm.get("objects", {}).get(oid)
            kept = ex is not None and np.hypot(*(box[:2] - ex["center"][:2])) <= moved_tol
            by_id[oid].append((len(frames), len(objs)))
            objs.append(dict(obj_id=oid, label=cls, subclass=sub, box=box,
                             velocity=np.asarray(ex["velocity"]) if kept else None))
            counts[cls] += 1
            counts["moved or new"] += not kept
        frames.append(dict(stamp=fm["stamp"], T_world_det=T, objects=objs))

    # track ids: numeric obj_ids (the auto-label tracks) are kept, others get new numbers
    numeric = [int(k) for k in by_id if k.isdigit()]
    next_id = max(numeric, default=0) + 1
    track_id = {}
    for k in sorted(by_id):
        if k.isdigit():
            track_id[k] = int(k)
        else:
            track_id[k], next_id = next_id, next_id + 1
    scores = meta.get("track_scores", {})

    # moved / new boxes: velocity from the annotated centers, in the world frame
    for oid, refs in by_id.items():
        refs = sorted(refs)
        world = [Detections(frames[fi]["objects"][oi]["box"][None], np.zeros((1, 2)), np.ones(1),
                            np.zeros(1, np.int64)).transform(frames[fi]["T_world_det"], "world") for fi, oi in refs]
        t = np.array([frames[fi]["stamp"] for fi, _ in refs])
        xy = np.stack([w.boxes[0, :2] for w in world])
        v = _window_velocity(t, xy[:, 0], xy[:, 1], velocity_window)
        v[np.hypot(v[:, 0], v[:, 1]) < static_speed] = 0.0
        for (fi, oi), vw in zip(refs, v):
            f = frames[fi]
            o = f["objects"][oi]
            if o["velocity"] is None:
                o["velocity"] = (inv_T(f["T_world_det"])[:3, :3] @ np.array([vw[0], vw[1], 0.0]))[:2]
            o.update(track_id=track_id[oid], score=float(scores.get(oid, 1.0)), review=False, reviewed=True,
                     interpolated=False, moving=bool(np.hypot(*o["velocity"]) > 0))
            del o["obj_id"]
    labels = dict(sequence=meta.get("sequence"), frames=frames, source="sustechpoints:" + os.path.abspath(scene_dir))
    return labels, dict(counts)
