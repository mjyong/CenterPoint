import csv

import numpy as np
import pytest

from dog_perception.geometry import make_T, rot_z
from dog_perception.label_review import _world, apply_edits, read_edits, write_review_mcap, write_track_table


def _labels():
    """Two tracks over 10 frames; the robot drives along x and turns, so det != world."""
    frames = []
    for k in range(10):
        T = make_T(rot_z(0.1 * k), [0.5 * k, 0, 0])
        objs = [dict(track_id=1, label="vehicle", box=np.array([10.0 - 0.5 * k + 0.05 * k, 2, 0.8, 4.5, 1.9, 1.6, 0.2]),
                     velocity=np.array([0.3, 0.0]), score=0.4, review=True, interpolated=False, moving=False),
                dict(track_id=2, label="cyclist", box=np.array([5.0, -3, 0.8, 1.8, 0.7, 1.6, 1.0]),
                     velocity=np.array([0.0, 2.0]), score=0.7, review=False, interpolated=k == 4, moving=True)]
        frames.append(dict(stamp=100.0 + 0.1 * k, T_world_det=T, objects=objs))
    return dict(sequence="test", frames=frames)


def _edit(path, labels, rows):
    table = write_track_table(path, labels)
    by_id = {r["track_id"]: r for r in table}
    out = []
    for tid, action, t_from, t_to in rows:
        r = dict(by_id[tid], action=action, **{"from": t_from, "to": t_to})
        out.append(r)
    out += [r for r in table if r["track_id"] not in {e[0] for e in rows}]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(out)
    return read_edits(path)


def test_track_table_flags_first(tmp_path):
    rows = write_track_table(str(tmp_path / "t.csv"), _labels())
    assert [r["track_id"] for r in rows] == [1, 2] and rows[0]["review"] == 1 and rows[1]["interpolated"] == 1
    assert rows[0]["t_first"] == "0.00" and rows[0]["t_last"] == "0.90"


def test_apply_edits(tmp_path):
    labels = _labels()
    edits = _edit(str(tmp_path / "t.csv"), labels, [(1, "static", "", ""), (2, "drop", "0.5", ""),
                                                    (2, "pedestrian+flip", "", "")])
    out, counts = apply_edits(labels, edits)
    assert counts == {"static": 1, "drop": 1, "pedestrian": 1, "flip": 1}
    t1 = [(f, o) for f in out["frames"] for o in f["objects"] if o["track_id"] == 1]
    w = np.array([_world(f, [o]).boxes[0] for f, o in t1])
    assert np.ptp(w[:, 0]) < 1e-9 and np.ptp(w[:, 6]) < 1e-9          # one world position and heading
    assert all(np.all(o["velocity"] == 0) and not o["review"] for _, o in t1)
    t2 = [(f, o) for f in out["frames"] for o in f["objects"] if o["track_id"] == 2]
    assert len(t2) == 5 and all(o["label"] == "pedestrian" and not o["review"] for _, o in t2)
    assert np.isclose(t2[0][1]["box"][6], 1.0 - np.pi)
    assert labels["frames"][0]["objects"][1]["label"] == "cyclist"          # input untouched


def test_vehicle_subclass_actions(tmp_path):
    """car / truck / bus set a vehicle's subclass; changing the class drops a vehicle subclass,
    confirming 'vehicle' keeps it. The table shows the teacher's subclass."""
    labels = _labels()
    for f in labels["frames"]:
        f["objects"][0]["subclass"] = "car"
        f["objects"][1]["subclass"] = "cyclist"
    rows = write_track_table(str(tmp_path / "t.csv"), labels)
    assert {r["track_id"]: r["subclass"] for r in rows} == {1: "car", 2: "cyclist"}
    edits = _edit(str(tmp_path / "t.csv"), labels, [(1, "truck", "", "0.45"), (1, "vehicle", "0.5", ""),
                                                    (2, "vehicle", "", "")])
    out, _ = apply_edits(labels, edits)
    t1 = [o for f in out["frames"] for o in f["objects"] if o["track_id"] == 1]
    assert [o["subclass"] for o in t1] == ["truck"] * 5 + ["car"] * 5 and {o["label"] for o in t1} == {"vehicle"}
    t2 = [o for f in out["frames"] for o in f["objects"] if o["track_id"] == 2]
    assert {(o["label"], o["subclass"]) for o in t2} == {("vehicle", None)}


def test_vehicle_table_shows_training_use(tmp_path):
    """table --classes vehicle: vehicles only, with size and what training makes of them; a reviewed
    track counts, a flagged one is to be decided."""
    labels = _labels()
    rows = write_track_table(str(tmp_path / "t.csv"), labels, "vehicle")
    assert [(r["track_id"], r["train_as"], r["length_m"], r["height_m"]) for r in rows] == [(1, "ignore", "4.5", "1.6")]
    out, _ = apply_edits(labels, _edit(str(tmp_path / "t.csv"), labels, [(1, "ok", "", "")]))
    assert write_track_table(str(tmp_path / "t2.csv"), out, "vehicle")[0]["train_as"] == "car"
    assert [r["train_as"] for r in write_track_table(str(tmp_path / "t3.csv"), labels)] == ["ignore", "cyclist"]


def test_ignore_action(tmp_path):
    """'ignore' clears the review flag and marks the track as an ignore box for training."""
    labels = _labels()
    out, counts = apply_edits(labels, _edit(str(tmp_path / "t.csv"), labels, [(1, "ignore", "", "")]))
    t1 = [o for f in out["frames"] for o in f["objects"] if o["track_id"] == 1]
    assert counts == {"ignore": 1} and all(o["ignore"] and not o["review"] and o["reviewed"] for o in t1)
    assert not any(o.get("ignore") for f in out["frames"] for o in f["objects"] if o["track_id"] == 2)


def test_unknown_action(tmp_path):
    with pytest.raises(ValueError):
        _edit(str(tmp_path / "t.csv"), _labels(), [(1, "remove", "", "")])


def test_review_mcap(tmp_path):
    from mcap.reader import make_reader
    labels = _labels()
    frames = tmp_path / "frames"
    frames.mkdir()
    for f in labels["frames"]:
        np.save(frames / ("%.6f.npy" % f["stamp"]), np.random.default_rng(0).normal(size=(50, 5)).astype(np.float32))
    results = [dict(stamp=f["stamp"], detections=None) for f in labels["frames"]]
    n = write_review_mcap(str(tmp_path / "r.mcap"), labels, str(frames), results)
    with open(tmp_path / "r.mcap", "rb") as fh:
        topics = {ch.topic for _, ch, _ in make_reader(fh).iter_messages()}
    assert n > 30 and {"/tf", "/lidar", "/labels", "/log"} <= topics
