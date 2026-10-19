"""det3d/datasets/pipelines/dog_aug.py: older sweeps moved as INS pose errors would, labels untouched."""
import numpy as np
import pytest

pytest.importorskip("torch")


def test_sweep_pose_jitter_moves_only_older_sweeps():
    from det3d.datasets.pipelines.dog_aug import SweepPoseJitter
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.normal(0, 10, (500, 3)), rng.random(500), np.repeat([0.0, 0.1, 0.2, 0.3, 0.4], 100)])
    boxes = np.ones((2, 9))
    res = dict(mode="train", lidar=dict(points=pts.copy(), combined=pts.copy(), annotations=dict(boxes=boxes.copy())))
    out, _ = SweepPoseJitter(prob=1.0, seed=1)(res, {})
    p = out["lidar"]["points"]
    assert p is out["lidar"]["combined"] and np.array_equal(p[:100], pts[:100])          # current sweep kept
    assert np.array_equal(p[:, 2:], pts[:, 2:]) and np.array_equal(out["lidar"]["annotations"]["boxes"], boxes)
    move = np.linalg.norm(p[:, :2] - pts[:, :2], axis=1)
    per_sweep = [np.median(move[k * 100:(k + 1) * 100]) for k in range(5)]
    assert per_sweep[0] == 0 and all(a < b for a, b in zip(per_sweep[1:], per_sweep[2:]))   # older: further
    # each older sweep is a rigid motion of itself (distances within a sweep kept)
    k = slice(400, 500)
    d0 = np.linalg.norm(pts[k, None, :2] - pts[None, k, :2], axis=-1)
    d1 = np.linalg.norm(p[k, None, :2] - p[None, k, :2], axis=-1)
    assert np.allclose(d0, d1, atol=1e-6)
    res = dict(mode="val", lidar=dict(points=pts.copy()))
    assert np.array_equal(SweepPoseJitter(prob=1.0)(res, {})[0]["lidar"]["points"], pts)  # val: untouched
