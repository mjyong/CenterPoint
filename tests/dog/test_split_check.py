"""tools/dog/split_check.py on two synthetic recordings with different ENU origins."""
import csv
import json
import os
import subprocess
import sys

import numpy as np

from dog_perception.inspvax import EnuOrigin

LAT0, LON0, H0 = 29.7332516, 106.7849784, 217.1


def _scene(root, name, split, origin, xy_common, t0, speed, lidar=None):
    """One sequence holding only meta.json and odom.npy (10 Hz); xy given in the common frame."""
    ref = EnuOrigin(LAT0, LON0, H0)
    off = ref.enu(*origin)[:2]
    t = t0 + 0.1 * np.arange(len(xy_common))
    odom = np.zeros((len(t), 11))
    odom[:, 0] = t
    odom[:, 1:3] = xy_common - off
    odom[:, 7] = 1.0
    odom[:, 8:10] = speed
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "scans"), exist_ok=True)
    np.save(os.path.join(d, "odom.npy"), odom)
    meta = dict(T_body_lidar=np.eye(4).tolist(), base_height=0.72, lat0=origin[0], lon0=origin[1], h0=origin[2],
                segment=dict(name=name, t_start=t0, t_end=t0 + 0.1 * len(t), split=split))
    if lidar:
        meta["lidar"] = lidar
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))


def _walk(p0, p1, n=300, static=0):
    """n samples from p0 to p1, the first ``static`` of them standing at p0."""
    a = np.r_[np.zeros(static), np.linspace(0.0, 1.0, n - static)]
    xy = np.asarray(p0, float) + a[:, None] * (np.asarray(p1, float) - np.asarray(p0, float))
    v = np.zeros((n, 2))
    if n > static:
        v[static:] = (np.asarray(p1, float) - np.asarray(p0, float)) / (0.1 * (n - static))
    return xy, v


def test_static_holdout_and_overlap(tmp_path):
    root = str(tmp_path / "seq")
    a, b = (LAT0, LON0, H0), (LAT0 + 100 / 110850.0, LON0, H0)      # B's origin ~100 m north of A's
    # A walks east along y = 0: s000 stands 10 s then 30 m, s001..s005 45 m each
    ends = [0, 30, 75, 120, 165, 210, 255]
    for k in range(6):
        xy, v = _walk((ends[k], 0), (ends[k + 1], 0), static=100 if k == 0 else 0)
        _scene(root, "rA_s%03d" % k, "train", a, xy, 1000.0 + 30 * k, v)
    # B: lies still at A's start, then walks north from there
    for k, (p0, p1) in enumerate([((0, 0), (0, 0)), ((0, 0), (0, 45)), ((0, 45), (0, 90))]):
        xy, v = _walk(p0, p1)
        _scene(root, "rB_s%03d" % k, "val", b, xy, 2000.0 + 30 * k, v)

    seqs = sorted(os.path.join(root, d) for d in os.listdir(root))
    cmd = [sys.executable, "tools/dog/split_check.py", "--seq", *seqs, "--out", str(tmp_path / "split"),
           "--holdout", "rA", "--holdout-s", "55", "--apply"]
    for _ in range(2):                                    # idempotent: starts from the survey split each time
        subprocess.run(cmd, check=True, capture_output=True)
        rows = {r["name"]: r for r in csv.DictReader(open(tmp_path / "split" / "split.csv"))}
        got = {n: r["split"] for n, r in rows.items()}
        assert got == {"rA_s000": "drop", "rA_s001": "train", "rA_s002": "train", "rA_s003": "drop",
                       "rA_s004": "val", "rA_s005": "val", "rB_s000": "drop", "rB_s001": "val", "rB_s002": "val"}
    assert rows["rB_s000"]["reason"].startswith("static val scene")
    assert rows["rA_s003"]["reason"] == "buffer next to the hold-out block"
    assert rows["rA_s000"]["reason"] == "overlaps val rB_s001"            # the two bags start at the same spot
    s = json.load(open(tmp_path / "split" / "split_check.json"))
    assert s["val_path_within_radius_of_train"] == 0.0 and s["after"]["val"]["scenes"] == 4
    seg = json.load(open(os.path.join(root, "rA_s004", "meta.json")))["segment"]
    assert seg["split"] == "val" and seg["split_survey"] == "train"
    assert os.path.getsize(tmp_path / "split" / "split_check.png") > 10000


def test_no_drop_keeps_every_scene_on_the_safe_side(tmp_path):
    """Same recordings as above with --no-drop: the static val scene stays val, the buffer goes to
    train, the train scene overlapping a val scene goes to val; nothing is dropped. The hold-out
    block now touches train (the buffer) at one end."""
    root = str(tmp_path / "seq")
    a, b = (LAT0, LON0, H0), (LAT0 + 100 / 110850.0, LON0, H0)
    ends = [0, 30, 75, 120, 165, 210, 255]
    for k in range(6):
        xy, v = _walk((ends[k], 0), (ends[k + 1], 0), static=100 if k == 0 else 0)
        _scene(root, "rA_s%03d" % k, "train", a, xy, 1000.0 + 30 * k, v)
    for k, (p0, p1) in enumerate([((0, 0), (0, 0)), ((0, 0), (0, 45)), ((0, 45), (0, 90))]):
        xy, v = _walk(p0, p1)
        _scene(root, "rB_s%03d" % k, "val", b, xy, 2000.0 + 30 * k, v)
    seqs = sorted(os.path.join(root, d) for d in os.listdir(root))
    subprocess.run([sys.executable, "tools/dog/split_check.py", "--seq", *seqs, "--out", str(tmp_path / "split"),
                    "--holdout", "rA", "--holdout-s", "55", "--no-drop"], check=True, capture_output=True)
    rows = {r["name"]: r for r in csv.DictReader(open(tmp_path / "split" / "split_check.csv"))}
    got = {n: r["split_after"] for n, r in rows.items()}
    assert got == {"rA_s000": "val", "rA_s001": "train", "rA_s002": "train", "rA_s003": "train",
                   "rA_s004": "val", "rA_s005": "val", "rB_s000": "val", "rB_s001": "val", "rB_s002": "val"}
    s = json.load(open(tmp_path / "split" / "split_check.json"))
    assert s["after"]["drop"]["scenes"] == 0 and 0 < s["val_path_within_radius_of_train"] < 0.2


def test_whole_recordings_held_out_as_test_and_val(tmp_path):
    """Short clips of another day (each its own recording, another lidar): the one farthest from
    train becomes test, the next val; --test by name also drops (--no-drop: moves to val) the train
    scene it overlaps."""
    root = str(tmp_path / "seq")
    a = (LAT0, LON0, H0)
    for k in range(3):                                          # XT32 recording along y = 0, x 0..135
        xy, v = _walk((45 * k, 0), (45 * (k + 1), 0))
        _scene(root, "rA_s%03d" % k, "train", a, xy, 1000.0 + 30 * k, v)
    clips = {"c1": ((300, 300), (300, 340)), "c2": ((50, 5), (90, 5)),        # c2 runs along rA_s001
             "c3": ((135, 8), (135, 48)), "c4": ((150, 200), (190, 200))}    # c3 starts near rA's end
    for k, (n, (p0, p1)) in enumerate(clips.items()):
        xy, v = _walk(p0, p1)
        _scene(root, "1016_%s_s000" % n, "train", a, xy, 5000.0 + 100 * k, v, lidar="MID360")
    seqs = sorted(os.path.join(root, d) for d in os.listdir(root))
    out = tmp_path / "split"

    def run(*extra):
        r = subprocess.run([sys.executable, "tools/dog/split_check.py", "--seq", *seqs, "--out", str(out), *extra],
                           check=True, capture_output=True, text=True)
        return {row["name"]: row for row in csv.DictReader(open(out / "split_check.csv"))}, r.stdout

    rows, stdout = run("--test-from", "1016_", "--test-share", "0.25", "--val-from", "1016_", "--val-share", "0.25")
    got = {n: r["split_after"] for n, r in rows.items()}
    assert got["1016_c1_s000"] == "test" and got["1016_c4_s000"] == "val"        # farthest from train first
    assert got["1016_c2_s000"] == got["1016_c3_s000"] == "train" and got["rA_s001"] == "train"
    assert "seconds by lidar: MID360 test 30, train 60, val 30; XT32 train 90" in stdout
    rows, _ = run("--test", "1016_c2", "--apply")
    assert rows["1016_c2_s000"]["split_after"] == "test" and rows["rA_s001"]["split_after"] == "drop"
    assert rows["rA_s001"]["reason"] == "overlaps test 1016_c2_s000"
    split = {r["name"]: (r["split"], r["lidar"]) for r in csv.DictReader(open(out / "split.csv"))}
    assert split["1016_c2_s000"] == ("test", "MID360") and split["rA_s000"] == ("train", "XT32")
    rows, _ = run("--test", "1016_c2", "--no-drop")
    assert rows["rA_s001"]["split_after"] == "val"
