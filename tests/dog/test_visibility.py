"""Occluded / far labels: points in the box decide positive, ignore box or left out."""
import os
import sys

import numpy as np

from dog_perception.label_review import train_label, write_track_table
from dog_perception.visibility import box_point_counts

CAR = np.array([10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.5])


def _in_box(box, n, lag, rng, z=None):
    """n points inside ``box`` (standard, det frame) with sweep time lag ``lag``."""
    local = rng.uniform([-0.45, -0.45, -0.3], [0.45, 0.45, 0.45], (n, 3)) * box[3:6]   # body, not wheels
    c, s = np.cos(box[6]), np.sin(box[6])
    xyz = np.c_[box[0] + c * local[:, 0] - s * local[:, 1], box[1] + s * local[:, 0] + c * local[:, 1],
                box[2] + local[:, 2] if z is None else np.full(n, z)]
    return np.c_[xyz, np.zeros(n), np.full(n, lag)].astype(np.float32)


def test_box_point_counts_current_sweep_all_sweeps_and_ground():
    rng = np.random.default_rng(0)
    pts = np.concatenate([_in_box(CAR, 20, 0.0, rng), _in_box(CAR, 7, 0.2, rng),
                          _in_box(CAR, 30, 0.0, rng, z=0.05),                      # ground under the car
                          _in_box(CAR + [6, 0, 0, 0, 0, 0, 0], 10, 0.0, rng)])     # the car behind
    cur, total = box_point_counts(pts, np.stack([CAR, CAR + [0, 30, 0, 0, 0, 0, 0]]))
    assert cur.tolist() == [20, 0] and total.tolist() == [27, 0]


def _car(**kw):
    return dict(dict(track_id=1, label="vehicle", subclass="car", box=CAR, velocity=np.zeros(2), score=0.7,
                     review=False, interpolated=False, moving=False), **kw)


def test_train_label_occluded_and_far_boxes():
    counts = {}
    assert train_label(_car(points=40, points_all=150), "vehicle", counts=counts) == ("car", False)
    assert train_label(_car(points=2, points_all=30), "vehicle", counts=counts) == ("car", True)      # hidden now
    assert train_label(_car(points=0, points_all=0), "vehicle", counts=counts) == (None, False)       # nothing
    assert train_label(_car(), "vehicle") == ("car", False)                  # no counts: as before
    assert counts["no points in the input (left out)"] == 1
    assert counts["few points in the current sweep (ignored)"] == 1


def test_export_leaves_out_hidden_boxes_and_ignores_barely_seen_ones(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))
    from export_dataset import export_recording

    rng = np.random.default_rng(1)
    seen, hidden, far = CAR, CAR + [0, 8, 0, 0, 0, 0, 0], CAR + [30, -5, 0, 0, 0, 0, 0]
    frames = tmp_path / "frames"
    frames.mkdir()
    np.save(frames / "10.000000.npy", np.concatenate([_in_box(seen, 50, 0.0, rng), _in_box(far, 3, 0.0, rng)]))
    objs = [_car(track_id=k, box=b) for k, b in enumerate((seen, hidden, far))]
    labels = dict(frames=[dict(stamp=10.0, T_world_det=np.eye(4), objects=objs)])
    counts = {}
    info, = export_recording(str(frames), labels, str(tmp_path / "out"), "rec", classes="vehicle", counts=counts)
    assert info["gt_names"].tolist() == ["car", "car"] and info["gt_ignore"].tolist() == [False, True]


def test_track_table_counts_hidden_frames(tmp_path):
    """A track hidden in its first frame is still a car; few_points counts its hidden frames."""
    frames = [dict(stamp=100.0 + 0.1 * k, T_world_det=np.eye(4),
                   objects=[_car(points=0 if k < 2 else 30, points_all=0 if k < 2 else 90)]) for k in range(6)]
    row, = write_track_table(str(tmp_path / "t.csv"), dict(frames=frames), "vehicle")
    assert row["train_as"] == "car" and row["few_points"] == 2
