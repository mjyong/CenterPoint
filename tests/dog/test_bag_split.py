"""Long-bag workflow on the moving sample bag: survey -> segments.csv -> one-pass split -> preview."""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

BAG = "samples/rosbag2_2026_09_21-16_41_27_moving"
TOPICS = ["--lidar-topic", "/robot/topic/rdap/top_point_cloud2", "--imu-topic", "/rte/rdap/rtk/imus/data_raw",
          "--inspvax-topic", "/rte/rdap/rtk/inspvax"]


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    pytest.importorskip("rosbags")
    if not os.path.isdir(BAG):
        pytest.skip("sample bag not checked out")
    root = tmp_path_factory.mktemp("split")
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True)   # noqa: E731
    run("tools/dog/bag_survey.py", "--bag", BAG, "--out", str(root / "survey"), "--scene-length", "1.0",
        "--min-length", "0.5", "--val-fraction", "0.3")
    run("tools/dog/rosbag_to_sequence.py", "--bag", BAG, "--out", str(root / "full"), *TOPICS)
    rows = list(csv.DictReader(open(root / "survey" / "segments.csv")))
    for r in rows:
        r["keep"] = 1                                     # also convert the buffer scene
    with open(root / "segments_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    run("tools/dog/rosbag_to_sequence.py", "--bag", BAG, "--out", str(root / "seq"), "--segments",
        str(root / "segments_all.csv"), *TOPICS)
    return root, rows


def test_survey_proposes_scenes_with_val_and_buffer(split):
    root, _ = split
    rows = list(csv.DictReader(open(root / "survey" / "segments.csv")))
    assert [r["split"] for r in rows] == ["train", "train", "val"] and [r["keep"] for r in rows] == ["1", "0", "1"]
    assert sum(int(r["scans"]) for r in rows) == 27
    s = json.load(open(root / "survey" / "survey.json"))
    assert s["lidar"]["frames"] == 27 and abs(s["lidar"]["rate_hz"] - 10) < 0.2 and s["ins"]["pos_type"] == {"56": 310}
    lag = s["log_minus_header"]
    assert 0.05 < lag["/robot/topic/rdap/top_point_cloud2"]["median_s"] < 0.2      # stamp = scan start
    assert abs(lag["/rte/rdap/rtk/inspvax"]["median_s"]) < 0.02
    assert os.path.getsize(root / "survey" / "survey.png") > 10000


def test_one_pass_split_matches_the_whole_bag(split):
    from dog_perception.io import SequenceReader
    root, rows = split
    full = SequenceReader(str(root / "full"))
    by_stamp = {float(np.load(f)["stamp"]): f for f in full.scan_files}
    total = 0
    for r in rows:
        seq = SequenceReader(str(root / "seq" / r["name"]))
        assert len(seq) == int(r["scans"])                 # survey and converter cut at the same stamps
        total += len(seq)
        for f in seq.scan_files:
            a = np.load(f)
            b = np.load(by_stamp[float(a["stamp"])])
            assert np.array_equal(a["xyz"], b["xyz"]) and np.array_equal(a["t"], b["t"])
        t0, t1 = float(r["t_start"]), float(r["t_end"])
        assert np.array_equal(seq.odom, full.odom[(full.odom[:, 0] >= t0 - 1) & (full.odom[:, 0] < t1 + 1)])
        assert seq.meta["lat0"] == full.meta["lat0"] and seq.meta["segment"]["split"] == r["split"]
    assert total == len(full)


def test_preview_passes_good_data_and_fails_a_wrong_extrinsic(split, tmp_path):
    root, _ = split
    bad = tmp_path / "bad_extrinsic"
    shutil.copytree(root / "full", bad)
    meta = json.load(open(bad / "meta.json"))
    meta["T_body_lidar"] = np.eye(4).tolist()             # the XT32 sits 180 deg in yaw on the kygi robot
    json.dump(meta, open(bad / "meta.json", "w"))
    subprocess.run([sys.executable, "tools/dog/sequence_preview.py", "--seq", str(root / "full"), str(bad),
                    "--out", str(tmp_path / "preview")], check=True, capture_output=True)
    checks = {c["name"]: c for c in json.load(open(tmp_path / "preview" / "checks.json"))}
    # the INS pulls its position back by up to 0.33 m while the robot brakes at the end of this bag
    assert checks["full"]["problems"] == "ins_step_m=WARN" and checks["full"]["align_m"] < 0.1
    assert 0.25 < checks["full"]["ins_step_m"] < 0.4 and checks["full"]["ins_steps_over_10cm"] == 5
    assert abs(checks["full"]["ground_z_m"]) < 0.1
    assert checks["bad_extrinsic"]["status"] == "FAIL" and checks["bad_extrinsic"]["align_m"] > 0.5
    assert (tmp_path / "preview" / "index.html").exists() and (tmp_path / "preview" / "full.png").exists()
