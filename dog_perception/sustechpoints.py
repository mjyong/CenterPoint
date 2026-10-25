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
they stand out in the tool. An object whose attribute contains ``ignore`` (set it in the tool for
something that cannot be told apart) comes back as an ignore box for training (``export_dataset.py``). ``dog_meta.json`` keeps what SUSTechPOINTS does
not: stamps, ``T_world_det`` per frame and the auto-label scores.

Import reads the (corrected) labels back. Classes and vehicle subclasses map
from the SUSTechPOINTS object types (``TYPE_TO_SUBCLASS``: Car / Van -> car,
Truck and construction machines -> truck, Bus / Tram / LongVehicle -> bus; other
types are skipped and counted). Export writes a vehicle's subclass as its type
(``SUBCLASS_TO_TYPE``), Car when it has none. Attributes (``obj_attr``, comma
separated): ``ignore`` makes an ignore box, ``static`` zero velocity, ``moving`` a
velocity from the reviewed boxes; ``review`` (exported on the boxes to check, drawn
red by the vendored tool, cleared when the reviewer confirms the object) keeps the
box flagged, so a box nobody confirmed becomes an ignore box, not a vehicle. A box
the annotator did not move (center within ``moved_tol``) keeps the exported
auto-label velocity, which already handles static objects (see
dog_perception/autolabel.py); moved or new boxes get a velocity from their
``obj_id``'s annotated centers in the world frame (difference over +-0.5 s,
zero below ``static_speed``). The result has the ``labels.pkl`` layout of
tools/dog/auto_label.py and goes straight into export_dataset.py.
"""
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict

import numpy as np

from .autolabel import _window_velocity
from .detection.boxes import CLASSES, SUBCLASS_TO_CLASS, SUBCLASSES, Detections
from .geometry import inv_T, wrap_angle
from .label_review import train_label
from .visibility import annotate_points, box_point_counts, visibility

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
    # small tricycles and mobility scooters are not vehicles (2026-10): cyclist
    **{t: "cyclist" for t in ("BicycleRider", "MotorcyleRider", "ScooterRider", "Bicycle", "Motorcycle", "Scooter",
                              "Trimotorcycle", "FreightTricycle")},
}
TYPE_TO_CLASS = {t: SUBCLASS_TO_CLASS[s] for t, s in TYPE_TO_SUBCLASS.items()}

META_FILE = "dog_meta.json"
LIDAR_HEIGHT = 1.8          # [m] SUSTechPOINTS' assumed lidar height above the ground


def frame_name(stamp):
    return "%016d" % int(round(float(stamp) * 1e6))


def write_pcd(path, xyz, intensity, age=None):
    """Binary PCD v0.7 with float fields x y z intensity (what SUSTechPOINTS' PCDLoader reads) and, with
    ``age``, t: the age of each point's sweep in seconds (0 for the current sweep). The vendored
    SUSTechPOINTS draws points with t > 0 dimmer (public/js/lidar.js)."""
    cols = [np.asarray(xyz, np.float32).reshape(-1, 3), np.asarray(intensity, np.float32).reshape(-1, 1)]
    fields = "x y z intensity"
    if age is not None:
        cols.append(np.asarray(age, np.float32).reshape(-1, 1))
        fields += " t"
    data = np.column_stack(cols).astype("<f4")
    n, k = data.shape
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS %s\nSIZE %s\nTYPE %s\nCOUNT %s\n"
              "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n"
              % (fields, " ".join(["4"] * k), " ".join(["F"] * k), " ".join(["1"] * k), n, n))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_pcd(path, with_age=False):
    """(N, 4) x y z intensity from a PCD written by ``write_pcd``; ``with_age``: (N, 5) with the sweep
    age (0 when the file has none)."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    head = raw[:end].decode("ascii")
    field = lambda key: [line for line in head.splitlines() if line.startswith(key)][0].split()[1:]   # noqa: E731
    n, names = int(field("POINTS")[0]), field("FIELDS")
    data = np.frombuffer(raw[end:end + 4 * len(names) * n], "<f4").reshape(n, len(names))
    out = data[:, :4]
    if with_age:
        out = np.column_stack([out, data[:, names.index("t")] if "t" in names else np.zeros(n, np.float32)])
    return out


def box_to_psr(box):
    x, y, z, l, w, h, yaw = (float(v) for v in box[:7])
    return {"position": {"x": x, "y": y, "z": z}, "scale": {"x": l, "y": w, "z": h},
            "rotation": {"x": 0.0, "y": 0.0, "z": float(wrap_angle(yaw))}}


def psr_to_box(psr):
    p, s, r = psr["position"], psr["scale"], psr["rotation"]
    return np.array([p["x"], p["y"], p["z"], abs(s["x"]), abs(s["y"]), abs(s["z"]), wrap_angle(r["z"])], np.float64)


SPLIT_TITLES = {"train": "训练集", "val": "验证集", "test": "测试集"}


def object_info(o, classes="coarse"):
    """What the vendored SUSTechPOINTS shows under a box (scene_reader.add_dog_info): the auto-label's
    track score, this frame's teacher score, speed, motion state, points in the box and what training
    makes of the box without review (``train_as``: class, 'ignore' or '' when left out)."""
    name, ign = train_label(o, classes)
    vis = visibility(o)
    return dict(score=round(float(o.get("score", 0.0)), 3), det_score=round(float(o.get("det_score", 0.0)), 3),
                interpolated=bool(o.get("interpolated", False)), speed=round(float(np.hypot(*o["velocity"])), 2),
                moving=bool(o.get("moving", False)), review=bool(o.get("review", False)),
                points=int(o["points"]), points_all=int(o["points_all"]), few_points=vis in ("none", "few"),
                train_as="ignore" if ign else name or "")


def needs_review(o, classes="coarse"):
    """A box the reviewer should look at: training would make it an ignore box for a track-level reason
    (flagged track, or with ``classes='vehicle'`` a teacher subclass that disagrees with the size), not
    for being hidden in this frame, and nobody marked it ignore yet."""
    if o.get("ignore"):
        return False
    track = {k: v for k, v in o.items() if k not in ("points", "points_all")}
    return train_label(track, classes)[1]


def label_digest(anns):
    """Content of one SUSTechPOINTS label file, independent of how it was written: SUSTechPOINTS saves
    every loaded frame on Ctrl+S, re-indented, numbers as JavaScript prints them, objects sorted by id."""
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return round(float(v), 5)
        return v
    items = sorted((norm(a) for a in anns or []), key=lambda a: str(a.get("obj_id")))
    return hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()


def edited_frames(root):
    """Frames of an exported scene whose labels someone changed in SUSTechPOINTS since the export (content,
    not file times: a save without changes does not count). Exports made before the digests were kept
    count a label file written after dog_meta.json as changed."""
    meta_path = os.path.join(root, META_FILE)
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as fh:
        meta = json.load(fh)
    out = []
    for fm in meta.get("frames", []):
        path = os.path.join(root, "label", fm["name"] + ".json")
        if not os.path.exists(path):
            out.append(fm["name"])
        elif "label_sha1" in fm:
            with open(path) as fh:
                if label_digest(json.load(fh)) != fm["label_sha1"]:
                    out.append(fm["name"])
        elif os.path.getmtime(path) > os.path.getmtime(meta_path):
            out.append(fm["name"])
    return out


def export_scene(labels, frames_dir, out_root, scene, every=1, t_start=None, t_end=None, current_sweep_only=False,
                 split=None, classes="coarse", overwrite=False, label_classes=None):
    """labels.pkl + run_sequence.py frames -> ``out_root/scene`` (point ``out_root`` at SUSTechPOINTS' data/).

    ``t_start`` / ``t_end``: seconds since the recording start. ``split`` (train / val / test) goes into
    the scene's desc.json, which the vendored SUSTechPOINTS uses to group its scene list. ``classes``: the
    training classes (``label_review.train_label``) that decide which boxes are marked ``review`` and
    the ``train_as`` shown in the tool. ``label_classes``: export only objects of these label classes
    (default all); the meta records them, so the import and ``propagate_keyframes`` know the review
    covered nothing else. A scene whose labels were changed in SUSTechPOINTS since its last export
    (``edited_frames``) is not written over unless ``overwrite``: that would throw the review away.
    Returns the number of frames written."""
    label_classes = list(label_classes or CLASSES)
    root = os.path.join(out_root, scene)
    edited = edited_frames(root)
    if edited and not overwrite:
        raise RuntimeError("%s: labels of %d frames were changed in SUSTechPOINTS since the export (%s ...); "
                           "import them first, or overwrite=True / --force to throw them away"
                           % (scene, len(edited), ", ".join(edited[:3])))
    os.makedirs(os.path.join(root, "lidar"), exist_ok=True)
    os.makedirs(os.path.join(root, "label"), exist_ok=True)
    os.makedirs(os.path.join(root, "dog_info"), exist_ok=True)
    if split:
        with open(os.path.join(root, "desc.json"), "w") as fh:
            json.dump({"scene": SPLIT_TITLES.get(split, split), "split": split}, fh, ensure_ascii=False)
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
        cloud = pts = np.load(src)
        if current_sweep_only and pts.shape[1] > 4:
            pts = pts[pts[:, 4] <= 1e-6]
        name = frame_name(f["stamp"])
        xyz = pts[:, :3] - [0.0, 0.0, LIDAR_HEIGHT]
        write_pcd(os.path.join(root, "lidar", name + ".pcd"), xyz, np.clip(pts[:, 3] / 255.0, 0, 1),
                  pts[:, 4] if pts.shape[1] > 4 else None)
        objs, exported, info = [], {}, {}
        # points counted on the detector's input, as export_dataset.py does (all sweeps, also with current_sweep_only)
        for o in annotate_points([o for o in f["objects"] if o["label"] in label_classes], cloud):
            box = np.asarray(o["box"], float).copy()
            box[2] -= LIDAR_HEIGHT
            otype = SUBCLASS_TO_TYPE.get(o.get("subclass")) or CLASS_TO_TYPE[o["label"]]
            objs.append({"obj_id": str(o["track_id"]), "obj_type": otype,
                         "obj_attr": "ignore" if o.get("ignore") else "review" if needs_review(o, classes) else "",
                         "psr": box_to_psr(box)})
            scores[o["track_id"]].append(float(o["score"]))
            exported[str(o["track_id"])] = dict(center=np.asarray(o["box"][:3], float).tolist(),
                                                velocity=np.asarray(o["velocity"], float).tolist())
            info[str(o["track_id"])] = object_info(o, classes)
        with open(os.path.join(root, "label", name + ".json"), "w") as fh:
            json.dump(objs, fh)
        with open(os.path.join(root, "dog_info", name + ".json"), "w") as fh:
            json.dump(info, fh)
        meta_frames.append(dict(name=name, stamp=float(f["stamp"]), label_sha1=label_digest(objs),
                                T_world_det=np.asarray(f["T_world_det"], float).tolist(), objects=exported))
    # files of frames this export does not write (an earlier export with other frames) would still be listed
    names = {fm["name"] for fm in meta_frames}
    for sub, ext in (("lidar", ".pcd"), ("label", ".json"), ("dog_info", ".json")):
        for fn in os.listdir(os.path.join(root, sub)):
            if fn.endswith(ext) and fn[:-len(ext)] not in names:
                os.remove(os.path.join(root, sub, fn))
    meta = dict(sequence=labels.get("sequence"), frames=meta_frames, class_to_type=CLASS_TO_TYPE,
                label_classes=label_classes, current_sweep_only=bool(current_sweep_only), z_offset=-LIDAR_HEIGHT,
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
    exported_ids = {oid for fm in meta["frames"] for oid in fm.get("objects", {})}
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
            attrs = {t.strip() for t in str(a.get("obj_attr") or "").lower().split(",")}
            vel = np.asarray(ex["velocity"]) if kept else None
            if "static" in attrs:                 # the reviewer says it stands still
                vel = np.zeros(2)
            elif "moving" in attrs:               # the reviewer says it moves: speed from the reviewed boxes
                vel = None
            counts["still marked review"] += "review" in attrs
            by_id[oid].append((len(frames), len(objs)))
            objs.append(dict(obj_id=oid, label=cls, subclass=sub, box=box, velocity=vel,
                             ignore="ignore" in str(a.get("obj_attr") or "").lower(),
                             still_review="review" in attrs,
                             added=oid not in exported_ids))      # drawn by the reviewer (propagate_keyframes)
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
            # a box still marked review (red, never confirmed) stays flagged: an ignore box in export_dataset.py
            o.update(track_id=track_id[oid], score=float(scores.get(oid, 1.0)), review=o.pop("still_review"),
                     reviewed=True,
                     interpolated=False, moving=bool(np.hypot(*o["velocity"]) > 0))
            del o["obj_id"]
    labels = dict(sequence=meta.get("sequence"), frames=frames, source="sustechpoints:" + os.path.abspath(scene_dir),
                  label_classes=meta.get("label_classes"))
    return labels, dict(counts)


CANDIDATE_ID = re.compile(r"c\d+$")       # obj_id of a candidate added by ``add_candidates``


def _det_type(label, sub):
    return SUBCLASS_TO_TYPE.get(SUBCLASSES[sub] if sub >= 0 else None) or CLASS_TO_TYPE[CLASSES[label]]


def write_detections(scene_dir, results, min_score=0.15):
    """The models' detections in each frame of an exported scene, for drawing boxes: the vendored
    SUSTechPOINTS turns the points selected with Ctrl + drag into the best-scoring of these boxes that
    holds most of them (third_party/SUSTechPOINTS/dog_fit.py). ``results``: {model name: run_sequence.py
    results.pkl rows}. Writes ``dog_dets/<frame>.json`` (detections of the scene's label classes with a
    score >= ``min_score``, in the tool's frame); returns the number of boxes written."""
    with open(os.path.join(scene_dir, META_FILE)) as fh:
        meta = json.load(fh)
    z_off = float(meta.get("z_offset", 0.0))
    want = [CLASSES.index(c) for c in (meta.get("label_classes") or CLASSES)]
    rows_at = {m: {round(float(r["stamp"]), 6): r for r in rows} for m, rows in results.items()}
    os.makedirs(os.path.join(scene_dir, "dog_dets"), exist_ok=True)
    total = 0
    for fm in meta["frames"]:
        out = []
        for m, at in rows_at.items():
            r = at.get(round(float(fm["stamp"]), 6))
            if r is None:
                continue
            d = r["detections"]
            for k in np.nonzero((d.scores >= min_score) & np.isin(d.labels, want))[0]:
                b = np.asarray(d.boxes[k], float).copy()
                b[2] += z_off
                psr = {key: {ax: round(v, 3) for ax, v in val.items()} for key, val in box_to_psr(b).items()}
                out.append(dict(model=m, score=round(float(d.scores[k]), 3),
                                obj_type=_det_type(int(d.labels[k]), int(d.subclasses[k])), psr=psr))
        out.sort(key=lambda x: -x["score"])
        with open(os.path.join(scene_dir, "dog_dets", fm["name"] + ".json"), "w") as fh:
            json.dump(out, fh)
        total += len(out)
    return total


def _attrs(a):
    return {t.strip() for t in str(a.get("obj_attr") or "").lower().split(",")}


def _bev_hit(xy, boxes, dist, margin):
    """Whether the BEV point ``xy`` is within ``dist`` of a box center or inside a box grown by ``margin``."""
    b = np.asarray(boxes, float).reshape(-1, 7)
    if not len(b):
        return False
    d = np.asarray(xy, float)[:2] - b[:, :2]
    c, s = np.cos(b[:, 6]), np.sin(b[:, 6])
    inside = (np.abs(c * d[:, 0] + s * d[:, 1]) <= b[:, 3] / 2 + margin) & \
        (np.abs(-s * d[:, 0] + c * d[:, 1]) <= b[:, 4] / 2 + margin)
    return bool((np.hypot(d[:, 0], d[:, 1]) <= dist).any() or inside.any())


def _same_object(box, boxes, dist, margin):
    """``box`` and one of ``boxes`` cover the same object: centers within ``dist``, or either center inside
    the other box grown by ``margin`` (a partial box of a long truck and a full one)."""
    boxes = np.asarray(boxes, float).reshape(-1, 7)
    return _bev_hit(box[:2], boxes, dist, margin) or any(_bev_hit(b[:2], box, dist, margin) for b in boxes)


def add_candidates(scene_dir, results, min_score=0.3, match_dist=2.0, margin=0.5, link_dist=2.5, max_gap=1.5,
                   min_points=1, backup=True, object_min_score=None, min_keyframes=1, max_range=None,
                   same_place=1.5):
    """Detections of other models as boxes to check ("candidates") in an exported scene, reviewed or not.

    Labels made by reviewing the teacher's boxes favour the teacher: what it missed is not labelled, so
    another model that finds it scores a false positive. This adds to every frame of the scene the
    detections in ``results`` ({model name: run_sequence.py results.pkl rows}) of the scene's label
    classes with a score >= ``min_score`` that nothing in the frame accounts for. A detection is left out
    when it and a box in the label file cover the same object (centers within ``match_dist``, or one
    center inside the other box grown by ``margin``: already labelled), when its center is within
    ``match_dist`` of a box the reviewer deleted (an exported object or an earlier candidate missing
    from the file: judged not to be one), or when the frame's cloud has fewer than ``min_points`` points
    in it (nothing to judge). Detections of several models on one object become one candidate, the
    best-scoring box. Candidates of frames up to ``max_gap`` seconds apart whose world centers (carried
    forward by the detection's velocity) are within ``link_dist`` share an ``obj_id`` (``c<n>``), so one
    confirm or delete in the tool decides the whole object.

    Fewer to review: ``max_range`` leaves out detections farther from the robot (score the models within
    the same range, compare_detectors.py --max-range); an object is kept only when it is a candidate in
    at least ``min_keyframes`` keyframes (one-frame flicker is mostly clutter) and its best box scores
    >= ``object_min_score`` (default ``min_score``; its other boxes down to ``min_score`` stay, so a
    vehicle keeps all its frames). A candidate within ``same_place`` m (world frame) of a label in
    another frame of the scene is marked with that label's id (``labelled_as``): mostly a parked vehicle
    the auto-label's track does not cover at this frame.

    Candidates are red (``obj_attr`` review). On import a confirmed one becomes a label, with a velocity
    from its annotated centers like a box the reviewer drew; one still red becomes an ignore box; a
    deleted one nothing. Running again replaces the candidates still red and keeps the confirmed ones;
    the ones the reviewer deleted are remembered in dog_meta.json and not proposed again. The label files
    are copied to ``label_backup/<time>/`` first. Close the scene in the browser before running and load
    it again afterwards: a save from a page loaded before would write the old labels back.
    Returns counts.
    """
    with open(os.path.join(scene_dir, META_FILE)) as fh:
        meta = json.load(fh)
    z_off = float(meta.get("z_offset", 0.0))
    want = [CLASSES.index(c) for c in (meta.get("label_classes") or CLASSES)]
    record = {k: dict(v) for k, v in meta.get("candidates", {}).get("frames", {}).items()}  # frame -> {id: xy}
    rows_at = {m: {round(float(r["stamp"]), 6): r for r in rows} for m, rows in results.items()}
    counts = defaultdict(int)
    if backup and os.path.isdir(os.path.join(scene_dir, "label")):
        dst = base = os.path.join(scene_dir, "label_backup", time.strftime("%Y%m%d-%H%M%S"))
        k = 1
        while os.path.exists(dst):
            dst, k = "%s-%d" % (base, k), k + 1
        shutil.copytree(os.path.join(scene_dir, "label"), dst)

    work = []
    for fm in sorted(meta["frames"], key=lambda f: f["stamp"]):
        name = fm["name"]
        path = os.path.join(scene_dir, "label", name + ".json")
        anns = []
        if os.path.exists(path):
            with open(path) as fh:
                anns = json.load(fh) or []
        ids = {str(a.get("obj_id")) for a in anns}
        old = {str(a.get("obj_id")) for a in anns if CANDIDATE_ID.match(str(a.get("obj_id"))) and "review" in _attrs(a)}
        anns = [a for a in anns if str(a.get("obj_id")) not in old]
        counts["earlier candidates replaced"] += len(old)
        rec = record.get(name, {})
        gone = [ex["center"][:2] for oid, ex in fm.get("objects", {}).items() if oid not in ids]
        gone += [xy for cid, xy in rec.items() if cid not in ids]
        record[name] = {cid: xy for cid, xy in rec.items() if cid not in old}
        labelled = np.array([psr_to_box(a["psr"]) for a in anns]).reshape(-1, 7)

        dets = []                                 # (score, model, box, velocity, label, subclass)
        for m, at in rows_at.items():
            r = at.get(round(float(fm["stamp"]), 6))
            if r is None:
                counts["frames without results: %s" % m] += 1
                continue
            d = r["detections"]
            near = np.hypot(d.boxes[:, 0], d.boxes[:, 1]) <= max_range if max_range else np.ones(len(d), bool)
            counts["beyond max range"] += int(((d.scores >= min_score) & np.isin(d.labels, want) & ~near).sum())
            for k in np.nonzero((d.scores >= min_score) & np.isin(d.labels, want) & near)[0]:
                dets.append((float(d.scores[k]), m, np.asarray(d.boxes[k], float), np.asarray(d.velocities[k], float),
                             int(d.labels[k]), int(d.subclasses[k])))
        clusters = []                             # [best detection, {model: score}]
        for det in sorted(dets, key=lambda x: -x[0]):
            hit = next((c for c in clusters if _same_object(det[2], c[0][2], match_dist, margin)), None)
            if hit is None:
                clusters.append([det, {det[1]: det[0]}])
            else:
                hit[1][det[1]] = max(hit[1].get(det[1], 0.0), det[0])
        cands = []
        for det, models in clusters:
            if _same_object(det[2], labelled, match_dist, margin):
                counts["already labelled"] += 1
            elif any(np.hypot(*(det[2][:2] - np.asarray(xy[:2]))) <= match_dist for xy in gone):
                counts["deleted by the reviewer"] += 1
            else:
                cands.append(dict(det=det, models=models))
        if cands:
            pts = read_pcd(os.path.join(scene_dir, "lidar", name + ".pcd"), with_age=True)
            boxes = np.stack([c["det"][2] for c in cands]) + [0, 0, z_off, 0, 0, 0, 0]
            cur, tot = box_point_counts(pts, boxes)
            for c, n, n_all in zip(cands, cur, tot):
                c.update(points=int(n), points_all=int(n_all))
            counts["no points"] += sum(c["points_all"] < min_points for c in cands)
            cands = [c for c in cands if c["points_all"] >= min_points]
        T = np.asarray(fm["T_world_det"], float)
        for c in cands:
            c["world"], c["world_vel"] = _to_world(c["det"][2], c["det"][3], T)
        work.append((fm, anns, cands))

    # labels of every frame in the world frame, for "labelled at this place in another frame"
    placed = [(fm["name"], str(a.get("obj_id")), _to_world(psr_to_box(a["psr"]), np.zeros(2),
                                                           np.asarray(fm["T_world_det"], float))[0][:2])
              for fm, anns, _ in work for a in anns if not CANDIDATE_ID.match(str(a.get("obj_id")))]
    placed_frame = np.array([p[0] for p in placed])
    placed_id = [p[1] for p in placed]
    placed_xy = np.array([p[2] for p in placed]).reshape(-1, 2)

    # ids: above every candidate number still in the files or remembered
    taken = [str(a.get("obj_id")) for _, anns, _ in work for a in anns]
    taken += [cid for rec in record.values() for cid in rec]
    taken = [int(cid[1:]) for cid in taken if CANDIDATE_ID.match(cid)]
    next_id = max(taken, default=0) + 1
    tracks = []                                   # [id, label, stamp, world xy, world velocity]
    for fm, anns, cands in work:
        t = float(fm["stamp"])
        pairs = sorted((float(np.hypot(*(c["world"][:2] - (tr[3] + tr[4] * (t - tr[2]))))), i, j)
                       for i, c in enumerate(cands) for j, tr in enumerate(tracks)
                       if tr[1] == c["det"][4] and 0 < t - tr[2] <= max_gap + 1e-6)
        got, used = {}, set()
        for d, i, j in pairs:
            if d <= link_dist and i not in got and j not in used:
                got[i] = j
                used.add(j)
        for i, c in enumerate(cands):
            if i in got:
                tr = tracks[got[i]]
            else:
                tr = ["c%d" % next_id, c["det"][4], t, None, None]
                next_id += 1
                tracks.append(tr)
            tr[2:] = [t, c["world"][:2], c["world_vel"]]
            c["id"] = tr[0]

    # objects seen in too few keyframes or never scoring high enough: left out
    boxes_of, best_of = defaultdict(int), defaultdict(float)
    for _, _, cands in work:
        for c in cands:
            boxes_of[c["id"]] += 1
            best_of[c["id"]] = max(best_of[c["id"]], c["det"][0])
    floor = min_score if object_min_score is None else object_min_score
    for k, (fm, anns, cands) in enumerate(work):
        few = [c for c in cands if boxes_of[c["id"]] < min_keyframes]
        low = [c for c in cands if boxes_of[c["id"]] >= min_keyframes and best_of[c["id"]] < floor]
        counts["left out: in fewer than %d keyframes" % min_keyframes] += len(few)
        counts["left out: best box below %.2f" % floor] += len(low)
        work[k] = (fm, anns, [c for c in cands if c not in few and c not in low])
    for k in [k for k in counts if k.startswith("left out") and not counts[k]]:
        del counts[k]
    if not counts.get("beyond max range"):
        counts.pop("beyond max range", None)

    for fm, anns, cands in work:
        name = fm["name"]
        for c in cands:
            d = np.where(placed_frame != name, np.hypot(*(placed_xy - c["world"][:2]).T), np.inf)
            c["labelled_as"] = placed_id[int(np.argmin(d))] if len(d) and d.min() <= same_place else None
            counts["at a place labelled in another frame"] += c["labelled_as"] is not None
        for c in cands:
            score, _, box, vel, label, sub = c["det"]
            b = box.copy()
            b[2] += z_off
            anns.append({"obj_id": c["id"], "obj_type": _det_type(label, sub), "obj_attr": "review",
                         "psr": box_to_psr(b)})
            record[name][c["id"]] = [float(box[0]), float(box[1])]
            for m in c["models"]:
                counts["found by %s" % m] += 1
            if len(c["models"]) == 1:
                counts["found by %s only" % next(iter(c["models"]))] += 1
        with open(os.path.join(scene_dir, "label", name + ".json"), "w") as fh:
            json.dump(anns, fh)
        info_path = os.path.join(scene_dir, "dog_info", name + ".json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path) as fh:
                info = json.load(fh)
        here = {str(a.get("obj_id")) for a in anns}
        info = {k: v for k, v in info.items() if not CANDIDATE_ID.match(k) or k in here}
        for c in cands:
            score, _, box, vel, label, sub = c["det"]
            o = dict(label=CLASSES[label], points=c["points"], points_all=c["points_all"])
            info[c["id"]] = dict(candidate=True, models=sorted(([m, round(s, 3)] for m, s in c["models"].items()),
                                                               key=lambda x: -x[1]),
                                 score=round(score, 3), det_score=round(score, 3), interpolated=False,
                                 speed=round(float(np.hypot(*vel)), 2), moving=bool(np.hypot(*vel) > 1.0),
                                 review=True, points=c["points"], points_all=c["points_all"],
                                 few_points=visibility(o) in ("none", "few"), train_as="ignore",
                                 labelled_as=c["labelled_as"])
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, "w") as fh:
            json.dump(info, fh)
        counts["candidate boxes"] += len(cands)
    counts["candidate objects"] = len({c["id"] for _, _, cands in work for c in cands})
    runs = meta.get("candidates", {}).get("runs", [])
    runs.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), models=sorted(results), min_score=min_score,
                     boxes=counts["candidate boxes"], objects=counts["candidate objects"]))
    meta["candidates"] = dict(runs=runs, frames={k: v for k, v in record.items() if v})
    with open(os.path.join(scene_dir, META_FILE), "w") as fh:
        json.dump(meta, fh, indent=1)
    return dict(counts)


def _to_world(box, vel, T):
    d = Detections(np.asarray(box, float)[None], np.asarray(vel, float).reshape(1, 2), np.ones(1),
                   np.zeros(1, np.int64)).transform(np.asarray(T, float), "world")
    return d.boxes[0], d.velocities[0]


def _to_det(box, vel, T):
    d = Detections(np.asarray(box, float)[None], np.asarray(vel, float).reshape(1, 2), np.ones(1),
                   np.zeros(1, np.int64)).transform(inv_T(np.asarray(T, float)), "det")
    return d.boxes[0], d.velocities[0]


def _link(a, b, link_dist):
    """Pairs of track ids between two keyframes' {id: (object, world box, world velocity)}: the same id,
    then, for objects the reviewer drew anew in each keyframe, the nearest of the same class."""
    pairs = {i: i for i in a if i in b}
    rest_a = [i for i in a if i not in pairs]
    rest_b = {i for i in b if i not in pairs.values()}
    cand = sorted((float(np.hypot(*(a[i][1][:2] - b[j][1][:2]))), i, j) for i in rest_a for j in rest_b
                  if a[i][0]["label"] == b[j][0]["label"])
    for d, i, j in cand:
        if d <= link_dist and i not in pairs and j in rest_b:
            pairs[i] = j
            rest_b.discard(j)
    return pairs


def propagate_keyframes(full, keyed, max_gap=1.0, link_dist=2.0, classes=None):
    """Labels at every frame from keyframes reviewed box by box (SUSTechPOINTS shows every n-th frame).

    ``full``: the scene's labels at every frame (labels.pkl, or labels_reviewed.pkl after the track review),
    ``keyed``: ``import_scene`` of its reviewed keyframes. ``classes``: the label classes the review
    covered; default the export's ``label_classes``, or for an export older than that, the classes in the
    keyframes. ``full`` objects of other classes nobody reviewed (a vehicles-only export) are copied as
    they are. Returns (labels, counts), for the reviewed classes:

    * keyframes: the reviewed objects as they are;
    * a frame between two keyframes at most ``max_gap`` seconds apart:
      - an object reviewed in both (same track id, or for boxes the reviewer drew anew in each keyframe, the
        nearest of the same class within ``link_dist`` in the world frame): center, size, yaw and velocity
        interpolated in the world frame; an ignore box if it is one, or still marked review, at either end;
      - a track that was at a neighbouring keyframe in ``full`` but not after review: removed by the
        reviewer, left out here too;
      - a ``full`` label of an object reviewed at one end only, or of a track no keyframe shows (shorter
        than the keyframe spacing): an ignore box, nobody looked at it;
    * frames before the first or after the last keyframe, or in a longer gap: left out.
    """
    counts = defaultdict(int)
    full_frames = [f for f in full["frames"] if not f.get("pose_bad")]
    keys = sorted(keyed["frames"], key=lambda f: f["stamp"])
    if not keys:
        return dict(full, frames=[]), dict(counts)
    key_t = np.array([f["stamp"] for f in keys])
    classes = set(classes or keyed.get("label_classes") or {o["label"] for f in keys for o in f["objects"]})

    def others(f):
        objs = [dict(o) for o in f["objects"] if o["label"] not in classes]
        counts["other classes, copied as they are"] += len(objs)
        return objs
    # boxes the reviewer added get ids above every track id of the full labels
    next_id = 1 + max([o["track_id"] for f in full["frames"] for o in f["objects"]] +
                      [o["track_id"] for f in keys for o in f["objects"]], default=0)
    remap = {}
    for f in keys:
        for o in f["objects"]:
            if o.get("added"):
                if o["track_id"] not in remap:
                    remap[o["track_id"]], next_id = next_id, next_id + 1
                o["track_id"] = remap[o["track_id"]]
    full_at = {round(f["stamp"], 4): {o["track_id"] for o in f["objects"]} for f in full["frames"]}

    def world(kf):
        return {o["track_id"]: (o, *_to_world(o["box"], o["velocity"], kf["T_world_det"])) for o in kf["objects"]}

    out = []
    for f in full_frames:
        t = f["stamp"]
        k = int(np.searchsorted(key_t, t - 1e-4))
        if k < len(keys) and abs(key_t[k] - t) < 1e-4:
            out.append(dict(f, objects=[dict(o) for o in keys[k]["objects"]] + others(f)))
            counts["keyframe boxes"] += len(keys[k]["objects"])
            continue
        if k == 0 or k == len(keys) or key_t[k] - key_t[k - 1] > max_gap + 1e-6:
            counts["frames left out (outside the keyframes)"] += 1
            continue
        k0, k1 = keys[k - 1], keys[k]
        a = (t - k0["stamp"]) / (k1["stamp"] - k0["stamp"])
        w0, w1 = world(k0), world(k1)
        pairs = _link(w0, w1, link_dist)
        objs = []
        for i, j in pairs.items():
            o0, b0, v0 = w0[i]
            o1, b1, v1 = w1[j]
            b = b0 + a * (b1 - b0)
            b[6] = wrap_angle(b0[6] + a * wrap_angle(b1[6] - b0[6]))
            box, vel = _to_det(b, v0 + a * (v1 - v0), f["T_world_det"])
            objs.append(dict(track_id=i, label=o0["label"], subclass=o0.get("subclass"), box=box, velocity=vel,
                             score=o0.get("score", 1.0), review=False, reviewed=True, interpolated=True,
                             moving=bool(np.hypot(*vel) > 0),
                             ignore=bool(o0.get("ignore") or o1.get("ignore") or o0.get("review") or o1.get("review"))))
        counts["interpolated boxes"] += len(objs)
        done = set(pairs) | set(pairs.values())
        reviewed = set(w0) | set(w1)
        auto_at_keys = full_at.get(round(k0["stamp"], 4), set()) | full_at.get(round(k1["stamp"], 4), set())
        for o in f["objects"]:
            tid = o["track_id"]
            if tid in done or o["label"] not in classes:
                continue
            if tid in reviewed:
                counts["ignore: reviewed at one end"] += 1
            elif tid in auto_at_keys:
                counts["left out: removed by the reviewer"] += 1
                continue
            else:
                counts["ignore: between keyframes only"] += 1
            objs.append(dict(o, ignore=True))
        out.append(dict(f, objects=objs + others(f)))
    return dict(full, frames=out, source="propagated from keyframes: " + str(keyed.get("source"))), dict(counts)
