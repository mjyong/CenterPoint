import json
import os

import numpy as np
import pytest

from dog_perception.geometry import make_T, rot_z
from dog_perception.sustechpoints import (box_to_psr, export_scene, frame_name, import_scene, psr_to_box, read_pcd,
                                          write_pcd)


def _labels():
    """A parked car (track 1, flagged) and a walking pedestrian (track 2); the robot drives and turns."""
    frames = []
    for k in range(10):
        T = make_T(rot_z(0.1 * k), [0.5 * k, 0.2 * k, 0])
        Ti = np.linalg.inv(T)
        car_w = np.array([12.0, 3.0, 0.8, 1])
        ped_w = np.array([6.0 + 1.2 * 0.1 * k, -2.0, 0.85, 1])
        objs = [dict(track_id=1, label="vehicle", box=np.r_[(Ti @ car_w)[:3], 4.5, 1.9, 1.6, 0.3 - 0.1 * k],
                     velocity=np.zeros(2), score=0.4, review=True, interpolated=False, moving=False),
                dict(track_id=2, label="pedestrian", box=np.r_[(Ti @ ped_w)[:3], 0.6, 0.6, 1.7, -0.1 * k],
                     velocity=(Ti[:3, :3] @ [1.2, 0, 0])[:2], score=0.8, review=False, interpolated=False,
                     moving=True)]
        frames.append(dict(stamp=1000.0 + 0.1 * k, T_world_det=T, objects=objs))
    return dict(sequence="test", frames=frames)


def _frames_dir(tmp_path, labels):
    d = tmp_path / "frames"
    d.mkdir()
    rng = np.random.default_rng(0)
    for f in labels["frames"]:
        pts = np.c_[rng.normal(size=(100, 3)), rng.uniform(0, 255, 100), np.repeat([0.0, 0.1], 50)]
        np.save(d / ("%.6f.npy" % f["stamp"]), pts.astype(np.float32))
    return str(d)


def test_pcd_and_psr_round_trip(tmp_path):
    xyz = np.random.default_rng(1).normal(size=(37, 3))
    write_pcd(str(tmp_path / "a.pcd"), xyz, np.linspace(0, 1, 37))
    with open(tmp_path / "a.pcd", "rb") as f:
        head = f.read(200).decode("ascii", "replace")
    assert "FIELDS x y z intensity" in head and "DATA binary" in head
    p = read_pcd(str(tmp_path / "a.pcd"))
    assert p.shape == (37, 4) and np.allclose(p[:, :3], xyz, atol=1e-6)
    box = np.array([1.0, -2.0, 0.8, 4.5, 1.9, 1.6, 2.5])
    psr = box_to_psr(box)
    assert psr["scale"] == {"x": 4.5, "y": 1.9, "z": 1.6} and psr["rotation"]["z"] == 2.5
    assert np.allclose(psr_to_box(psr), box)


def test_export_import_round_trip(tmp_path):
    labels = _labels()
    n = export_scene(labels, _frames_dir(tmp_path, labels), str(tmp_path / "data"), "rec")
    root = tmp_path / "data" / "rec"
    assert n == 10 and sorted(os.listdir(root)) == ["dog_info", "dog_meta.json", "label", "lidar"]
    name = frame_name(labels["frames"][0]["stamp"])
    ann = json.load(open(root / "label" / (name + ".json")))
    assert [(a["obj_id"], a["obj_type"], a["obj_attr"]) for a in ann] == [("1", "Car", "review"),
                                                                         ("2", "Pedestrian", "")]
    pcd = read_pcd(str(root / "lidar" / (name + ".pcd")))
    assert pcd.shape == (100, 4)                                                  # accumulated sweeps by default
    src = np.load(os.path.join(str(tmp_path / "frames"), "%.6f.npy" % labels["frames"][0]["stamp"]))
    # SUSTechPOINTS expects the ground 1.8 m below its origin: points and boxes are shifted down
    assert np.allclose(pcd[:, 2], src[:, 2] - 1.8, atol=1e-5)
    assert np.isclose(ann[0]["psr"]["position"]["z"], labels["frames"][0]["objects"][0]["box"][2] - 1.8)
    out, counts = import_scene(str(root))
    assert counts == {"vehicle": 10, "pedestrian": 10, "moved or new": 0, "still marked review": 10}
    for fa, fb in zip(labels["frames"], out["frames"]):
        assert np.isclose(fa["stamp"], fb["stamp"]) and np.allclose(fa["T_world_det"], fb["T_world_det"])
        for a, b in zip(fa["objects"], fb["objects"]):
            assert (a["track_id"], a["label"]) == (b["track_id"], b["label"])
            assert b["review"] == a["review"]           # the flagged car was never confirmed: still flagged
            assert b["subclass"] == ("car" if a["label"] == "vehicle" else a["label"])   # no subclass -> Car
            assert np.allclose(a["box"], b["box"]) and np.allclose(a["velocity"], b["velocity"])
            assert np.isclose(b["score"], a["score"])


def test_import_annotator_edits(tmp_path):
    labels = _labels()
    export_scene(labels, _frames_dir(tmp_path, labels), str(tmp_path / "data"), "rec", every=2, current_sweep_only=True)
    root = tmp_path / "data" / "rec"
    names = sorted(os.listdir(root / "label"))
    assert len(names) == 5 and read_pcd(str(root / "lidar" / names[0].replace(".json", ".pcd"))).shape == (50, 4)
    T = {fm["name"]: np.array(fm["T_world_det"]) for fm in json.load(open(root / "dog_meta.json"))["frames"]}
    for nm in names:
        ann = json.load(open(root / "label" / nm))
        ann[0]["obj_type"] = "Van"                                     # still a vehicle
        shift = np.linalg.inv(T[nm[:-5]])[:3, :3] @ [0.0, 0.3, 0.0]    # whole pedestrian track 0.3 m to the world's +y
        ann[1]["psr"]["position"]["x"] += shift[0]
        ann[1]["psr"]["position"]["y"] += shift[1]
        bike = (np.linalg.inv(T[nm[:-5]]) @ [8.0, 5.0, 0.6, 1.0])[:3]     # parked bicycle, fixed in the world
        ann.append({"obj_id": "bike", "obj_type": "Bicycle", "psr": box_to_psr([*bike, 1.7, 0.6, 1.1, 0.0])})
        ann.append({"obj_id": "77", "obj_type": "Cone", "psr": box_to_psr([1.0, 1.0, 0.3, 0.3, 0.3, 0.6, 0.0])})
        json.dump(ann, open(root / "label" / nm, "w"))
    out, counts = import_scene(str(root))
    assert counts["vehicle"] == 5 and counts["cyclist"] == 5 and counts["skipped type Cone"] == 5
    assert counts["moved or new"] == 10                                 # 5 shifted pedestrian boxes + 5 new bicycles
    peds = [(f, o) for f in out["frames"] for o in f["objects"] if o["label"] == "pedestrian"]
    v_world = [(f["T_world_det"][:3, :3] @ np.r_[o["velocity"], 0])[:2] for f, o in peds]
    assert np.allclose(v_world, [1.2, 0.0], atol=0.05)                  # recomputed from the shifted boxes
    bikes = [o for f in out["frames"] for o in f["objects"] if o["label"] == "cyclist"]
    assert len({o["track_id"] for o in bikes}) == 1 and bikes[0]["track_id"] == 3   # new id after 1, 2
    assert all(np.all(o["velocity"] == 0) and not o["moving"] for o in bikes)
    assert {o["subclass"] for f in out["frames"] for o in f["objects"] if o["label"] == "vehicle"} == {"car"}


def test_vehicle_subclass_types(tmp_path):
    """Export writes the vehicle subclass as the object type; import maps types back to car / truck / bus."""
    labels = _labels()
    for f in labels["frames"]:
        f["objects"][0]["subclass"] = "truck"
    export_scene(labels, _frames_dir(tmp_path, labels), str(tmp_path / "data"), "rec")
    root = tmp_path / "data" / "rec"
    names = sorted(os.listdir(root / "label"))
    retyped = ["Truck", "Bus", "LongVehicle", "ForkLift", "Van", "Tram", "Trimotorcycle", "Truck", "Truck", "Car"]
    for nm, typ in zip(names, retyped):
        ann = json.load(open(root / "label" / nm))
        assert ann[0]["obj_type"] == "Truck"
        ann[0]["obj_type"] = typ
        json.dump(ann, open(root / "label" / nm, "w"))
    out, counts = import_scene(str(root), type_map={"Trimotorcycle": "car"})
    subs = [f["objects"][0]["subclass"] for f in out["frames"]]
    assert subs == ["truck", "bus", "bus", "truck", "car", "bus", "car", "truck", "truck", "car"]
    assert {f["objects"][0]["label"] for f in out["frames"]} == {"vehicle"} and counts["vehicle"] == 10



def test_ignore_attribute_round_trip(tmp_path):
    """An ignored track is exported with obj_attr 'ignore'; setting it in the tool marks a box ignored."""
    labels = _labels()
    for f in labels["frames"]:
        f["objects"][0]["ignore"] = True
    export_scene(labels, _frames_dir(tmp_path, labels), str(tmp_path / "data"), "rec", current_sweep_only=True)
    root = tmp_path / "data" / "rec"
    names = sorted(os.listdir(root / "label"))
    ann = json.load(open(root / "label" / names[0]))
    assert [a["obj_attr"] for a in ann] == ["ignore", ""]
    ann[1]["obj_attr"] = "Ignore"
    json.dump(ann, open(root / "label" / names[0], "w"))
    out, _ = import_scene(str(root))
    flags = [(o["label"], o["ignore"]) for o in out["frames"][0]["objects"]]
    assert flags == [("vehicle", True), ("pedestrian", True)]
    assert not any(o["ignore"] for f in out["frames"][1:] for o in f["objects"] if o["label"] == "pedestrian")


def test_export_for_review_display(tmp_path):
    """What the vendored SUSTechPOINTS shows: the split (desc.json), per box the auto-label's score, speed,
    motion and points (dog_info/, merged into the annotations by its scene_reader.py), boxes to check in
    red (obj_attr review: a flagged track, or a teacher 'truck' the size of a car in the vehicle layout),
    and the sweep age per point (PCD field t, older sweeps drawn dimmer)."""
    import importlib.util
    labels = _labels()
    for k, f in enumerate(labels["frames"]):
        f["objects"][0].update(review=False, subclass="truck", det_score=0.0 if k == 3 else 0.5, interpolated=k == 3)
        f["objects"][1]["det_score"] = 0.8
    frames = _frames_dir(tmp_path, labels)
    export_scene(labels, frames, str(tmp_path / "data"), "rec", split="train", classes="vehicle")
    root = tmp_path / "data" / "rec"
    assert json.load(open(root / "desc.json")) == {"scene": "训练集", "split": "train"}
    name = frame_name(labels["frames"][3]["stamp"])
    ann = json.load(open(root / "label" / (name + ".json")))
    assert [a["obj_attr"] for a in ann] == ["review", ""]               # a "truck" 4.5 m long: to check
    info = json.load(open(root / "dog_info" / (name + ".json")))
    # no point of the random cloud falls in the car's box: left out of training whatever the review says
    assert info["1"]["train_as"] == "" and info["1"]["points_all"] == 0 and info["1"]["few_points"]
    assert info["1"]["interpolated"] and info["1"]["det_score"] == 0.0
    assert info["2"]["moving"] and abs(info["2"]["speed"] - 1.2) < 0.01 and info["2"]["det_score"] == 0.8
    pcd = read_pcd(str(root / "lidar" / (name + ".pcd")), with_age=True)
    assert pcd.shape == (100, 5) and np.allclose(np.unique(pcd[:, 4]), [0.0, 0.1])

    spec = importlib.util.spec_from_file_location(
        "scene_reader", os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "SUSTechPOINTS",
                                     "scene_reader.py"))
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reader.root_dir = str(tmp_path / "data")
    served = reader.read_annotations("rec", name)
    assert [a["dog"]["points_all"] for a in served] == [0, info["2"]["points_all"]]
    assert reader.get_scene_desc("rec")["split"] == "train"
    out, _ = import_scene(str(root))                                    # the extra files do not change the import
    assert len(out["frames"]) == 10 and all("dog" not in o for f in out["frames"] for o in f["objects"])


def test_propagate_keyframe_review_to_every_frame(tmp_path):
    """Training scenes are reviewed box by box on every 5th frame; the frames between get the reviewed
    boxes interpolated in the world frame, tracks the reviewer removed stay removed, and what nobody looked
    at becomes an ignore box."""
    from dog_perception.sustechpoints import propagate_keyframes

    frames = []
    for k in range(11):
        t = 0.1 * k
        T = make_T(rot_z(0.05 * k), [1.0 * t, 0.3 * t, 0])      # the robot walks and turns
        Ti = np.linalg.inv(T)

        def det(xw, yw, yaw_w, size=(4.5, 1.9, 1.6)):
            return np.r_[(Ti @ [xw, yw, 0.8, 1])[:3], size, yaw_w - 0.05 * k]
        objs = [dict(track_id=1, label="vehicle", box=det(12.0, 3.0, 0.3), velocity=np.zeros(2)),     # parked
                dict(track_id=2, label="vehicle", box=det(5.0 + 5.0 * t, -4.0, 0.0),                  # 5 m/s
                     velocity=(Ti[:3, :3] @ [5.0, 0, 0])[:2]),
                dict(track_id=3, label="vehicle", box=det(20.0, 10.0, 1.0), velocity=np.zeros(2))]    # false
        if k in (2, 3):
            objs.append(dict(track_id=4, label="vehicle", box=det(30.0, -8.0, 0.0), velocity=np.zeros(2)))
        objs.append(dict(track_id=5, label="pedestrian", box=det(8.0, 6.0, 0.0, (0.6, 0.6, 1.7)),  # not in the
                         velocity=np.zeros(2)))                                                   # vehicle review
        for o in objs:
            o.update(score=0.6, review=False, interpolated=False, moving=o["track_id"] == 2)
        frames.append(dict(stamp=1000.0 + t, T_world_det=T, objects=objs))
    full = dict(sequence="test", frames=frames)
    export_scene(full, _frames_dir(tmp_path, full), str(tmp_path / "data"), "rec", every=5, label_classes=["vehicle"])
    root = tmp_path / "data" / "rec"
    T = {fm["name"]: np.array(fm["T_world_det"]) for fm in json.load(open(root / "dog_meta.json"))["frames"]}
    for n, nm in enumerate(sorted(os.listdir(root / "label"))):           # keyframes 0, 5, 10
        Ti = np.linalg.inv(T[nm[:-5]])
        ann = [a for a in json.load(open(root / "label" / nm)) if a["obj_id"] != "3"]   # not a vehicle
        shift = Ti[:3, :3] @ [0.0, 0.5, 0.0]                                 # parked car moved 0.5 m (world +y)
        ann[0]["psr"]["position"]["x"] += shift[0]
        ann[0]["psr"]["position"]["y"] += shift[1]
        new = (Ti @ [0.0, 15.0 + 1.0 * 0.5 * n, -1.0, 1.0])[:3]             # a missed car, 2 m/s along +y,
        ann.append({"obj_id": "new%d" % n, "obj_type": "Car",              # drawn anew in each keyframe
                    "psr": box_to_psr([*new, 4.5, 1.9, 1.6, 1.57 - 0.25 * n])})
        json.dump(ann, open(root / "label" / nm, "w"))
    keyed, _ = import_scene(str(root))
    out, counts = propagate_keyframes(full, keyed)

    assert len(out["frames"]) == 11
    def world(f, tid):
        o = [o for o in f["objects"] if o["track_id"] == tid]
        return (f["T_world_det"] @ np.r_[o[0]["box"][:3], 1])[:3] if o else None, (o[0] if o else None)
    f2 = out["frames"][2]
    p1, o1 = world(f2, 1)
    assert np.allclose(p1[:2], [12.0, 3.5], atol=1e-6) and not o1["ignore"] and o1["interpolated"]
    p2, o2 = world(f2, 2)
    assert np.allclose(p2[:2], [6.0, -4.0], atol=1e-6) and np.isclose(np.hypot(*o2["velocity"]), 5.0, atol=0.2)
    assert all(world(f, 3)[0] is None for f in out["frames"])            # removed in review, removed everywhere
    assert world(f2, 4)[1]["ignore"] and world(out["frames"][3], 4)[1]["ignore"]   # nobody saw it
    added = [o for o in f2["objects"] if o["track_id"] > 5]
    assert len(added) == 1 and not added[0]["ignore"]
    pa = (f2["T_world_det"] @ np.r_[added[0]["box"][:3], 1])[:3]
    assert np.allclose(pa[:2], [0.0, 15.2], atol=1e-6)                     # linked by distance, interpolated
    assert counts["ignore: between keyframes only"] == 2 and counts["left out: removed by the reviewer"] == 8
    # the pedestrian nobody reviewed: as it was on every frame, not removed by the reviewer
    assert all([o for o in f["objects"] if o["label"] == "pedestrian"] == [frames[k]["objects"][-1]]
               for k, f in enumerate(out["frames"]))
    assert counts["other classes, copied as they are"] == 11
    del keyed["label_classes"]                                            # an export older than label_classes:
    assert propagate_keyframes(full, keyed)[1] == counts                  # the classes in the keyframes


def test_import_motion_attributes(tmp_path):
    """static / moving set by the reviewer override the auto-label's motion of an unmoved box."""
    labels = _labels()
    export_scene(labels, _frames_dir(tmp_path, labels), str(tmp_path / "data"), "rec")
    root = tmp_path / "data" / "rec"
    for nm in os.listdir(root / "label"):
        ann = json.load(open(root / "label" / nm))
        ann[0]["obj_attr"] = "moving"                      # speed from its reviewed boxes, which stand still
        ann[1]["obj_attr"] = "review,static"               # the walking pedestrian is said to stand
        json.dump(ann, open(root / "label" / nm, "w"))
    out, counts = import_scene(str(root))
    car = [o for f in out["frames"] for o in f["objects"] if o["label"] == "vehicle"]
    ped = [o for f in out["frames"] for o in f["objects"] if o["label"] == "pedestrian"]
    assert all(not o["moving"] for o in car)               # recomputed from its boxes: they do not move
    assert all(np.all(o["velocity"] == 0) and not o["moving"] for o in ped) and counts["still marked review"] == 10


def test_export_refuses_to_overwrite_a_reviewed_scene(tmp_path):
    """A second export over a scene someone changed in SUSTechPOINTS would throw the review away: refused.
    A save without changes (Ctrl+S re-writes every loaded frame, re-indented, 0.0 as 0) is not a change."""
    labels = _labels()
    frames = _frames_dir(tmp_path, labels)
    data = str(tmp_path / "data")
    export_scene(labels, frames, data, "rec", every=2)
    root = tmp_path / "data" / "rec"
    names = sorted(os.listdir(root / "label"))

    def js_save(nm, edit=None):
        ann = json.load(open(root / "label" / nm))
        for a in ann:
            a["psr"]["rotation"]["x"] = 0                     # JavaScript prints 0.0 as 0
        if edit:
            edit(ann)
        ann.sort(key=lambda a: -int(a["obj_id"]))
        json.dump(ann, open(root / "label" / nm, "w"), indent=2, sort_keys=True)
    for nm in names:
        js_save(nm)
    export_scene(labels, frames, data, "rec", every=2)          # nothing changed: exported again
    js_save(names[1], lambda ann: ann[0]["psr"]["position"].update(x=ann[0]["psr"]["position"]["x"] + 0.4))
    with pytest.raises(RuntimeError, match="1 frames"):
        export_scene(labels, frames, data, "rec", every=2)
    assert export_scene(labels, frames, data, "rec", every=5, overwrite=True) == 2
    assert len(os.listdir(root / "label")) == 2 and len(os.listdir(root / "lidar")) == 2   # stale frames removed


def _box_points(rng, box, n=200):
    """Points on the surface of a det-frame box (sides and top), as the lidar would hit a vehicle."""
    x, y, z, l, w, h, yaw = box
    u, v, s = rng.uniform(-0.5, 0.5, n) * l, rng.uniform(-0.5, 0.5, n) * w, rng.integers(0, 3, n)
    u = np.where(s == 0, np.sign(u) * l / 2, u)
    v = np.where(s == 1, np.sign(v) * w / 2, v)
    zz = np.where(s == 2, z + h / 2, rng.uniform(z - h / 2 + 0.3, z + h / 2, n))
    c, sn = np.cos(yaw), np.sin(yaw)
    return np.c_[x + c * u - sn * v, y + sn * u + c * v, zz]


def _candidate_scene(tmp_path):
    """The robot drives along x. World: car A (track 1, labelled), car C (track 3, the reviewer deleted it),
    car B (driving at 3 m/s along y, the teacher missed it), a ghost D where the cloud is empty.
    Returns (scene dir, labels, {model: results rows})."""
    from dog_perception.detection.boxes import Detections
    rng = np.random.default_rng(3)
    world = dict(A=[12.0, 3.0, 0.8, 4.5, 1.9, 1.6, 0.3], C=[8.0, 6.0, 0.8, 4.4, 1.8, 1.5, 1.2],
                 B=[20.0, -8.0, 0.8, 4.6, 1.9, 1.6, np.pi / 2], D=[30.0, 10.0, 0.8, 4.5, 1.9, 1.6, 0.0])
    frames, m1, m2 = [], [], []
    fdir = tmp_path / "frames"
    fdir.mkdir()
    for k in range(10):
        t = 1000.0 + 0.1 * k
        T = make_T(rot_z(0.05 * k), [0.5 * k, 0.0, 0.0])
        Ti = np.linalg.inv(T)
        det = {}
        for name, b in world.items():
            b = np.array(b, float)
            if name == "B":
                b[1] += 3.0 * 0.1 * k
            p = Ti @ np.r_[b[:3], 1.0]
            det[name] = np.r_[p[:3], b[3:6], b[6] - 0.05 * k]
        vel_b = (Ti[:3, :3] @ [0.0, 3.0, 0.0])[:2]
        objs = [dict(track_id=1, label="vehicle", subclass="car", box=det["A"], velocity=np.zeros(2), score=0.8,
                     review=False, interpolated=False, moving=False),
                dict(track_id=3, label="vehicle", subclass="car", box=det["C"], velocity=np.zeros(2), score=0.4,
                     review=True, interpolated=False, moving=False)]
        frames.append(dict(stamp=t, T_world_det=T, objects=objs))
        pts = np.vstack([_box_points(rng, det[n]) for n in "ACB"] + [np.c_[rng.uniform(-30, 30, (300, 2)), np.zeros(300)]])
        np.save(fdir / ("%.6f.npy" % t), np.c_[pts, np.full(len(pts), 100.0), np.zeros(len(pts))].astype(np.float32))
        boxes = np.stack([det["A"], det["B"], det["C"], det["D"]])
        vel = np.stack([np.zeros(2), vel_b, np.zeros(2), np.zeros(2)])
        m1.append(dict(stamp=t, detections=Detections(boxes, vel, np.array([0.9, 0.8, 0.7, 0.6]), np.zeros(4, np.int64),
                                                      subclasses=np.array([0, 0, 0, 0]))))
        jit = det["B"] + [0.3, -0.2, 0, 0, 0, 0, 0.05]           # the same car B, slightly off; and a pedestrian
        m2.append(dict(stamp=t, detections=Detections(np.stack([jit, [5.0, 1.0, 0.9, 0.6, 0.6, 1.7, 0.0]]),
                                                      np.stack([vel_b, np.zeros(2)]), np.array([0.6, 0.9]),
                                                      np.array([0, 1]), subclasses=np.array([1, 3]))))
    labels = dict(sequence="test", frames=frames)
    export_scene(labels, str(fdir), str(tmp_path / "data"), "rec", label_classes=["vehicle"])
    root = tmp_path / "data" / "rec"
    for nm in os.listdir(root / "label"):                         # the reviewer deletes C everywhere
        ann = [a for a in json.load(open(root / "label" / nm)) if a["obj_id"] != "3"]
        json.dump(ann, open(root / "label" / nm, "w"))
    return root, labels, dict(m1=m1, m2=m2)


def test_candidates_from_other_models(tmp_path):
    from dog_perception.sustechpoints import add_candidates
    root, labels, results = _candidate_scene(tmp_path)
    counts = add_candidates(str(root), results)
    assert counts["candidate boxes"] == 10 and counts["candidate objects"] == 1      # B, one id in every frame
    assert counts["already labelled"] == 10 and counts["deleted by the reviewer"] == 10
    assert counts["no points"] == 10 and counts["found by m2"] == 10 and "found by m1 only" not in counts
    assert len(os.listdir(root / "label_backup")) == 1
    name = frame_name(labels["frames"][4]["stamp"])
    ann = {a["obj_id"]: a for a in json.load(open(root / "label" / (name + ".json")))}
    assert set(ann) == {"1", "c1"} and ann["c1"]["obj_attr"] == "review" and ann["c1"]["obj_type"] == "Car"
    info = json.load(open(root / "dog_info" / (name + ".json")))["c1"]
    assert info["candidate"] and info["models"] == [["m1", 0.8], ["m2", 0.6]] and info["points"] > 100
    assert ann["c1"]["psr"]["position"]["z"] == pytest.approx(0.8 - 1.8)            # the best-scoring box, m1's

    # still red: an ignore box after import; confirmed: a vehicle with the speed of its annotated centers
    out, _ = import_scene(str(root))
    assert [o["review"] for f in out["frames"] for o in f["objects"] if o.get("added")] == [True] * 10
    for nm in os.listdir(root / "label"):
        ann = json.load(open(root / "label" / nm))
        for a in ann:
            if a["obj_id"] == "c1" and nm != name + ".json":
                a["obj_attr"] = ""                                 # confirmed; deleted in one frame
        json.dump([a for a in ann if not (a["obj_id"] == "c1" and nm == name + ".json")], open(root / "label" / nm, "w"))
    out, _ = import_scene(str(root))
    b = [(f, o) for f in out["frames"] for o in f["objects"] if o.get("added")]
    assert len(b) == 9 and not any(o["review"] for _, o in b)
    f, o = b[4]
    v_world = f["T_world_det"][:2, :2] @ o["velocity"]
    assert np.allclose(v_world, [0.0, 3.0], atol=0.3)

    # again: nothing new; the frame where the reviewer deleted it stays without
    counts = add_candidates(str(root), results)
    assert counts.get("candidate boxes", 0) == 0 and counts["already labelled"] == 19
    assert counts["deleted by the reviewer"] == 11


def _dog_fit():
    import importlib
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "SUSTechPOINTS"))
    try:
        return importlib.import_module("dog_fit")
    finally:
        sys.path.pop(0)


def test_fit_box_from_selected_points(tmp_path):
    """Ctrl + drag in SUSTechPOINTS: the L-shape fit follows the two visible sides where the principal axis does
    not; a model's box that holds the object wins; ground and a smaller group behind are not the object."""
    fit = _dog_fit()
    rng = np.random.default_rng(5)
    yaw = np.deg2rad(30.0)
    c, s = np.cos(yaw), np.sin(yaw)
    # a car 4.6 x 1.9 x 1.5 seen from (0, 0): its rear and right side, on the ground at z = -1.8 (SUSTechPOINTS frame)
    u = np.r_[np.full(80, -2.3), rng.uniform(-2.3, 2.3, 200)]
    v = np.r_[rng.uniform(-0.95, 0.95, 80), np.full(200, -0.95)]
    car = np.c_[10 + c * u - s * v, 6 + s * u + c * v, rng.uniform(-1.5, -0.3, 280)]
    ground = np.c_[rng.uniform(6, 14, (400, 2)) + [0, -1], rng.normal(-1.8, 0.02, 400)]
    behind = np.c_[rng.uniform(16, 16.5, (30, 2)), rng.uniform(-1.5, 0, 30)]
    pts = np.vstack([car, ground, behind])
    scene = tmp_path / "rec"
    (scene / "dog_dets").mkdir(parents=True)
    json.dump({"z_offset": -1.8}, open(scene / "dog_meta.json", "w"))

    got = fit.fit(str(scene), "f0", pts.tolist(), yaw_hint=yaw + 0.3)
    assert got["source"] == "fit" and got["points"] == 280
    p = got["psr"]
    assert abs(np.rad2deg(p["rotation"]["z"] - yaw)) < 2.0                       # heading toward the hint
    assert abs(p["scale"]["x"] - 4.6) < 0.1 and abs(p["scale"]["y"] - 1.9) < 0.1
    assert abs(p["position"]["z"] - p["scale"]["z"] / 2 + 1.8) < 0.05            # bottom on the ground
    assert np.hypot(p["position"]["x"] - 10, p["position"]["y"] - 6) < 0.1
    pca = np.linalg.eigh(np.cov((car[:, :2] - car[:, :2].mean(0)).T))[1][:, -1]
    assert abs(np.rad2deg(np.arctan2(pca[1], pca[0]) % np.pi - yaw)) > 5.0       # what SUSTechPOINTS' fallback gets
    alone = fit.fit(str(scene), "f0", (car + [0, 0, 1.0]).tolist())["psr"]        # far away: no ground in sight
    assert abs(alone["position"]["z"] - alone["scale"]["z"] / 2 - car[:, 2].min() - 1.0) < 1e-6
    back = fit.fit(str(scene), "f0", pts.tolist(), yaw_hint=yaw + np.pi)["psr"]["rotation"]["z"]
    assert abs(np.rad2deg(np.angle(np.exp(1j * (back - yaw - np.pi))))) < 2.0    # g's other heading

    model = {"position": {"x": 10.2, "y": 6.1, "z": -1.0}, "scale": {"x": 4.8, "y": 2.0, "z": 1.6},
             "rotation": {"x": 0, "y": 0, "z": yaw}}
    json.dump([{"model": "v1_pp", "score": 0.4, "obj_type": "Car", "psr": model},
               {"model": "nusc", "score": 0.9, "obj_type": "Truck",
                "psr": dict(model, position={"x": 30.0, "y": 6.0, "z": -1.0})}], open(scene / "dog_dets" / "f0.json", "w"))
    got = fit.fit(str(scene), "f0", pts.tolist())
    assert got["source"] == "v1_pp 0.40" and got["psr"] == model and got["obj_type"] == "Car"
    assert fit.fit(str(scene), "f0", ground.tolist()) == {}


def test_candidate_filters_and_same_place(tmp_path):
    """Fewer candidates to review: range, keyframes an object is seen in, its best score. A candidate where
    another frame has a label is marked with that label's id."""
    from dog_perception.sustechpoints import add_candidates
    root, labels, results = _candidate_scene(tmp_path)
    names = [frame_name(f["stamp"]) for f in labels["frames"]]
    ann = json.load(open(root / "label" / (names[0] + ".json")))
    det_b = results["m1"][0]["detections"].boxes[1]              # B, drawn by the reviewer in the first frame
    b = dict(ann[0], obj_id="7")
    b["psr"] = {"position": {"x": det_b[0], "y": det_b[1], "z": det_b[2] - 1.8},
                "scale": {"x": det_b[3], "y": det_b[4], "z": det_b[5]}, "rotation": {"x": 0, "y": 0, "z": det_b[6]}}
    json.dump(ann + [b], open(root / "label" / (names[0] + ".json"), "w"))
    for kw, left in ((dict(max_range=15), "beyond max range"), (dict(min_keyframes=10), "in fewer than 10 keyframes"),
                     (dict(object_min_score=0.85), "best box below 0.85")):
        counts = add_candidates(str(root), results, backup=False, **kw)
        assert counts.get("candidate boxes", 0) == 0 and any(left in k for k in counts if counts[k] >= 9), counts
    counts = add_candidates(str(root), results, backup=False, max_range=40, min_keyframes=9, object_min_score=0.75)
    assert counts["candidate boxes"] == 9 and counts["candidate objects"] == 1 and counts["already labelled"] == 11
    info = [json.load(open(root / "dog_info" / (n + ".json")))["c1"] for n in names[1:]]
    assert info[0]["labelled_as"] == "7" and info[-1]["labelled_as"] is None        # B drove 2.7 m away by then
    assert counts["at a place labelled in another frame"] == sum(i["labelled_as"] == "7" for i in info) >= 3


def test_candidates_of_an_export_without_label_classes(tmp_path):
    """Scenes exported before dog_meta.json recorded label_classes (the 0921 validation scenes): without
    classes every class came in, the second model's pedestrian too. classes=["vehicle"] keeps to vehicles."""
    from dog_perception.sustechpoints import add_candidates, write_detections
    root, labels, results = _candidate_scene(tmp_path)
    meta = json.load(open(root / "dog_meta.json"))
    del meta["label_classes"]
    json.dump(meta, open(root / "dog_meta.json", "w"))
    counts = add_candidates(str(root), results, backup=False)
    assert counts["no points"] == 20                       # ghost D and the pedestrian (no points there) considered
    counts = add_candidates(str(root), results, backup=False, classes=["vehicle"])
    assert counts["no points"] == 10 and counts["candidate objects"] == 1   # the pedestrian is not even looked at
    name = frame_name(labels["frames"][3]["stamp"])
    assert {a["obj_type"] for a in json.load(open(root / "label" / (name + ".json")))} == {"Car"}
    write_detections(str(root), results, classes=["vehicle"])
    assert {d["obj_type"] for d in json.load(open(root / "dog_dets" / (name + ".json")))} == {"Car", "Truck"}
