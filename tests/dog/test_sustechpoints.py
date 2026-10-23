import json
import os

import numpy as np

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
        for o in objs:
            o.update(score=0.6, review=False, interpolated=False, moving=o["track_id"] == 2)
        frames.append(dict(stamp=1000.0 + t, T_world_det=T, objects=objs))
    full = dict(sequence="test", frames=frames)
    export_scene(full, _frames_dir(tmp_path, full), str(tmp_path / "data"), "rec", every=5)
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
    added = [o for o in f2["objects"] if o["track_id"] > 4]
    assert len(added) == 1 and not added[0]["ignore"]
    pa = (f2["T_world_det"] @ np.r_[added[0]["box"][:3], 1])[:3]
    assert np.allclose(pa[:2], [0.0, 15.2], atol=1e-6)                     # linked by distance, interpolated
    assert counts["ignore: between keyframes only"] == 2 and counts["left out: removed by the reviewer"] == 8


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
