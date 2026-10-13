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
    assert n == 10 and sorted(os.listdir(root)) == ["dog_meta.json", "label", "lidar"]
    name = frame_name(labels["frames"][0]["stamp"])
    ann = json.load(open(root / "label" / (name + ".json")))
    assert [(a["obj_id"], a["obj_type"], a["obj_attr"]) for a in ann] == [("1", "Car", "review"),
                                                                         ("2", "Pedestrian", "")]
    assert read_pcd(str(root / "lidar" / (name + ".pcd"))).shape == (100, 4)      # accumulated sweeps by default
    out, counts = import_scene(str(root))
    assert counts == {"vehicle": 10, "pedestrian": 10, "moved or new": 0}
    for fa, fb in zip(labels["frames"], out["frames"]):
        assert np.isclose(fa["stamp"], fb["stamp"]) and np.allclose(fa["T_world_det"], fb["T_world_det"])
        for a, b in zip(fa["objects"], fb["objects"]):
            assert (a["track_id"], a["label"]) == (b["track_id"], b["label"]) and not b["review"]
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

