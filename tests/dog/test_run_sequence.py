"""tools/dog/run_sequence.py --replay: the saved detector inputs and the poses of an earlier run."""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import run_sequence as rs  # noqa: E402
from dog_perception.detection.boxes import Detections  # noqa: E402
from dog_perception.geometry import make_T, rot_z  # noqa: E402


class FakeDetector:
    """One parked car at world (10, 2): its box in each frame's det frame."""

    def __init__(self):
        self.last_timing = {"network_ms": 1.0}
        self.T = {}

    def __call__(self, points, stamp):
        T = self.T[round(stamp, 6)]
        p = np.linalg.inv(T) @ [10.0, 2.0, 0.8, 1.0]
        return Detections(np.array([[p[0], p[1], p[2], 4.5, 1.9, 1.6, 0.0]]), np.zeros((1, 2)),
                          np.array([0.8]), np.array([0], dtype=np.int64))


def _run_dir(tmp_path, n=12):
    run = tmp_path / "runs" / "scene_a"
    (run / "frames").mkdir(parents=True)
    (run / "voxel").mkdir()
    rows = []
    for k in range(n):
        t = 1000.0 + 0.1 * k
        T = make_T(rot_z(0.02 * k), [0.5 * k * 0.1, 0.0, 0.0])
        rows.append(dict(stamp=t, T_world_det=T, T_world_body=T, pose_bad=k == 7))
        if k != 4:                                       # one frame was not saved
            np.save(run / "frames" / ("%.6f.npy" % t), np.random.default_rng(k).random((50, 5)).astype(np.float32))
    rows.reverse()                                       # any order: replay sorts by stamp
    with open(run / "voxel" / "results.pkl", "wb") as f:
        pickle.dump(rows, f)
    return run, rows


def _args(**kw):
    a = dict(replay=["x"], seq_root="nowhere", save_frames=False, no_mcap=True, mcap=None, no_predict=False,
             jobs=1, odom_latency=0.0)
    a.update(kw)
    return argparse.Namespace(**a)


def test_replay_frames_follow_the_saved_run(tmp_path):
    run, rows = _run_dir(tmp_path)
    got = list(rs.replay_frames(str(run)))
    assert [round(f.stamp, 1) for f, _ in got] == [round(1000.0 + 0.1 * k, 1) for k in range(12) if k != 4]
    by_t = {round(r["stamp"], 6): r for r in rows}
    for f, bad in got:
        assert np.allclose(f.T_world_det, by_t[round(f.stamp, 6)]["T_world_det"]) and f.points.shape == (50, 5)
        assert bad == (abs(f.stamp - 1000.7) < 1e-6)


def test_replay_scene_tracks_and_skips_prediction(tmp_path):
    run, rows = _run_dir(tmp_path)
    det = FakeDetector()
    det.T = {round(r["stamp"], 6): r["T_world_det"] for r in rows}
    out = tmp_path / "out"
    lines = rs.run_scene(str(run), str(out), {"pillar": det}, _args(no_predict=True))
    assert lines[0].startswith("[pillar] 11 frames")
    res = pickle.load(open(out / "pillar" / "results.pkl", "rb"))
    assert len(res) == 11 and all(r["predictions"] == [] for r in res)
    assert [r["pose_bad"] for r in res].count(True) == 1
    last = res[-1]["tracks"]
    assert len(last) == 1 and np.allclose(last[0].position[:2], [10.0, 2.0], atol=1e-6)   # one track, in place
    lines = rs.run_scene(str(run), str(tmp_path / "out2"), {"pillar": det}, _args())
    assert any(len(r["predictions"]) for r in pickle.load(open(tmp_path / "out2" / "pillar" / "results.pkl", "rb")))
