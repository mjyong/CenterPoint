"""Reviewed training labels made fuller (dog_perception/label_merge.py)."""
import numpy as np

from dog_perception.geometry import make_T, rot_z
from dog_perception.label_merge import add_teacher_ignores, fill_static_gaps


def _scene():
    """The robot drives along x and turns a little. A parked car at world (10, 5): track 1 in frames 0-4,
    nothing in 5-9, track 2 in 10-14. The cloud shows it in frames 5-7, not in 8-9 (hidden)."""
    frames, clouds = [], {}
    rng = np.random.default_rng(0)
    for k in range(15):
        T = make_T(rot_z(0.03 * k), [0.3 * k, 0.0, 0.0])
        Ti = np.linalg.inv(T)
        c = Ti @ [10.0, 5.0, 0.8, 1.0]
        box = np.r_[c[:3], 4.5, 1.9, 1.6, 0.2 - 0.03 * k]
        objs = []
        if k < 5 or k >= 10:
            objs.append(dict(track_id=1 if k < 5 else 2, label="vehicle", subclass="car", box=box,
                             velocity=np.zeros(2), score=0.7, review=False, reviewed=True, ignore=False,
                             interpolated=False, moving=False))
        stamp = 100.0 + 0.1 * k
        frames.append(dict(stamp=stamp, T_world_det=T, objects=objs, pose_bad=False))
        pts = np.zeros((0, 5))
        if k <= 7:
            p = rng.uniform(-0.5, 0.5, (40, 3)) * [4.0, 1.6, 1.2] + box[:3]
            pts = np.c_[p, np.full(40, 50.0), np.zeros(40)]
        clouds[round(stamp, 6)] = pts
    return dict(sequence="t", frames=frames), clouds


def test_fill_static_gaps_where_the_cloud_shows_the_car():
    labels, clouds = _scene()
    out, counts = fill_static_gaps(labels, lambda s: clouds[round(s, 6)])
    assert counts["gap frames filled"] == 3 and counts["gap frames hidden (left as they are)"] == 2
    assert counts["parked places split into several tracks"] == 1
    for k in (5, 6, 7):
        o = out["frames"][k]["objects"]
        assert len(o) == 1 and o[0]["filled"] and o[0]["track_id"] == 1 and not o[0]["ignore"]
        w = out["frames"][k]["T_world_det"] @ np.r_[o[0]["box"][:3], 1.0]
        assert np.allclose(w[:2], [10.0, 5.0], atol=1e-6)
    assert out["frames"][8]["objects"] == [] and len(labels["frames"][5]["objects"]) == 0   # input unchanged


def test_teacher_objects_the_review_never_covered_become_ignore_boxes():
    labels, _ = _scene()
    extra = {"frames": []}
    for f in labels["frames"]:
        objs = [dict(o, track_id=7) for o in f["objects"]]                       # the car again: covered
        objs.append(dict(track_id=9, label="vehicle", subclass="truck", box=np.array([-12.0, -6, 1.2, 8, 2.5, 3, 0]),
                         velocity=np.zeros(2), score=0.55, review=False, interpolated=False, moving=False))
        objs.append(dict(track_id=11, label="pedestrian", subclass="pedestrian", box=np.array([3.0, 3, 0.9, .6, .6, 1.7, 0]),
                         velocity=np.zeros(2), score=0.5, review=False, interpolated=False, moving=False))
        extra["frames"].append(dict(stamp=f["stamp"], objects=objs))
    out, counts = add_teacher_ignores(labels, extra)
    assert counts["added as ignore boxes"] == 15 and counts["tracks added"] == 1 and counts["already labelled"] == 10
    added = [o for o in out["frames"][3]["objects"] if o.get("added_by")]
    assert len(added) == 1 and added[0]["ignore"] and added[0]["track_id"] == 3 and added[0]["subclass"] == "truck"
