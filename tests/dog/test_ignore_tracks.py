"""tools/dog/ignore_tracks.py: forklift-sized tracks become ignore boxes in keyframe and 10 Hz labels."""
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import ignore_tracks as it  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _labels(n, every):
    """A car (track 1) and a forklift (track 2, added by the reviewer: 10 Hz id 50) for n frames."""
    frames = []
    for k in range(0, n, every):
        T = np.eye(4)
        objs = [dict(track_id=1, label="vehicle", box=np.array([10.0, 0, 0.8, 4.6, 1.9, 1.6, 0])),
                dict(track_id=2 if every > 1 else 50, label="vehicle", added=every > 1,
                     box=np.array([5.0, 3.0, 1.0, 3.3, 1.5, 2.0, 0])),
                dict(track_id=7, label="pedestrian", box=np.array([2.0, 2, 0.9, 0.5, 0.5, 1.7, 0]))]
        frames.append(dict(stamp=100.0 + 0.1 * k, T_world_det=T, objects=objs))
    return dict(frames=frames)


def test_forklifts_become_ignore_boxes(tmp_path):
    key, dense = _labels(20, 5), _labels(20, 1)
    sel = it.select_tracks(key, (2.3, 3.6), (1.3, 1.85), (1.95, 2.6))
    assert list(sel) == [2] and sel[2]["keyframes"] == 4
    d = tmp_path / "scene_a"
    d.mkdir()
    pickle.dump(key, open(d / "key.pkl", "wb"))
    pickle.dump(dense, open(d / "dense.pkl", "wb"))
    cmd = [sys.executable, os.path.join(ROOT, "tools", "dog", "ignore_tracks.py"), "--scenes", "scene_a",
           "--keyframes", str(tmp_path / "{scene}" / "key.pkl"), "--also", str(tmp_path / "{scene}" / "dense.pkl"),
           "--length", "2.3", "3.6", "--width", "1.3", "1.85", "--height", "1.95", "2.6", "--reason", "forklift"]
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True).stdout
    assert "key.pkl: 4 boxes" in out and "dense.pkl: 20 boxes" in out
    for name in ("key.pkl", "dense.pkl"):
        labels = pickle.load(open(d / name, "rb"))
        for f in labels["frames"]:
            by = {o["track_id"]: o for o in f["objects"]}
            fork = by.get(2, by.get(50))
            assert fork["ignore"] and fork["ignore_reason"] == "forklift"
            assert not by[1].get("ignore") and not by[7].get("ignore")
    again = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True).stdout
    assert "total: nothing" in again
