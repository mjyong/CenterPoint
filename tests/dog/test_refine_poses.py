"""tools/dog/refine_poses.py: lidar odometry (KISS-ICP) + RTK fusion.

* the fusion on synthetic data with ground truth: an INS that runs 0.5 m ahead for 1.5 s (as the kygi
  INS does while the robot brakes) is corrected back to the truth, and a gap where the odometry
  also fails is reported as bad;
* the moving sample bag: the preview's lidar shift check fails with the INS poses and passes after
  refinement; re-running is a no-op.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

pytest.importorskip("kiss_icp")

import refine_poses  # noqa: E402

from dog_perception.io import SequenceReader, odom_row  # noqa: E402

BAG = "samples/rosbag2_2026_09_21-16_41_27_moving"
TOPICS = ["--lidar-topic", "/robot/topic/rdap/top_point_cloud2", "--imu-topic", "/rte/rdap/rtk/imus/data_raw",
          "--inspvax-topic", "/rte/rdap/rtk/inspvax"]


def _synthetic(lo_stall=False, k=60):
    """10 Hz scans for 6 s, robot walking along +x at 1.2 m/s (stamps 1.0 .. 6.9 s). INS rows at
    100 Hz: truth plus a braking error (0.5 m ahead between 2.0 and 3.7 s) with true velocities,
    so position and velocity disagree there. Odometry: truth with 2 % scale drift and 5 mm noise,
    or stalled (no motion seen) during the braking."""
    rng = np.random.default_rng(0)
    stamps = 1.0 + 0.1 * np.arange(k)
    t = np.arange(0.0, 8.0, 0.01)
    truth = lambda tt: np.column_stack([1.2 * tt, 0.2 * np.sin(0.5 * tt), np.zeros_like(tt)])   # noqa: E731
    err = np.interp(t, [0, 2.0, 2.5, 3.3, 3.7, 99], [0, 0, 0.5, 0.5, 0, 0])
    p = truth(t) + np.column_stack([err, 0 * err, 0 * err])
    v = np.gradient(truth(t), t, axis=0)
    odom = np.array([odom_row(ti, np.eye(3), pi, vi) for ti, pi, vi in zip(t, p, v)])
    lo_xy = truth(stamps) * np.array([1.02, 1.02, 1.0]) + rng.normal(0, 0.005, (k, 3))
    if lo_stall:
        stall = (stamps > 2.0) & (stamps < 3.6)
        for i in np.nonzero(stall)[0]:
            lo_xy[i:] -= lo_xy[i] - lo_xy[i - 1]                  # this step saw no motion
    poses = np.repeat(np.eye(4)[None], k, axis=0)
    poses[:, :3, 3] = lo_xy - lo_xy[0]
    return stamps, poses, odom, truth(stamps)[:, :2], np.interp(stamps, t, err)


def test_fuse_bridges_braking_error():
    stamps, poses, odom, truth, ins_err = _synthetic()
    xy, corr, info, diag = refine_poses.fuse(stamps, poses, odom)
    e = np.linalg.norm(xy - truth, axis=1)
    assert ins_err.max() > 0.49 and corr.max() > 0.4
    assert e.max() < 0.06, e.round(3)
    assert info["bad_intervals"] == [] and 0.2 < info["ins_unhealthy_share"] < 0.5
    out = refine_poses.refined_odom(odom, stamps, xy)                # 100 Hz rows, heading untouched
    assert out.shape == odom.shape and np.array_equal(out[:, 3:8], odom[:, 3:8])
    assert np.allclose(np.interp(stamps, out[:, 0], out[:, 1]), xy[:, 0])


def test_fuse_keeps_good_odometry_while_the_ins_jumps():
    """While braking, the kygi INS position is pulled back in steps of up to 0.45 m. Those steps
    must not get good odometry steps rejected (and replaced by the INS jumps)."""
    stamps, poses, odom, truth, _ = _synthetic()
    t = odom[:, 0]
    saw = np.where((t > 2.0) & (t < 3.6), 0.45 * (((t - 2.0) / 0.4) % 1.0), 0.0)
    odom = odom.copy()
    odom[:, 1] = 1.2 * t + saw                                    # truth along x plus the sawtooth only
    xy, corr, info, diag = refine_poses.fuse(stamps, poses, odom)
    e = np.linalg.norm(xy - truth, axis=1)
    assert info["lo_steps_rejected"] == 0 and e.max() < 0.06, (info["lo_steps_rejected"], e.round(3))


def test_fuse_reports_a_stalled_odometry():
    """The odometry sees no motion during the braking (a wall-only corridor): neither source is
    right there. The gap is reported as bad; outside it the healthy INS holds the trajectory."""
    stamps, poses, odom, truth, _ = _synthetic(lo_stall=True)
    xy, corr, info, diag = refine_poses.fuse(stamps, poses, odom)
    e = np.linalg.norm(xy - truth, axis=1)
    assert len(info["bad_intervals"]) == 1
    t0, t1, closure = info["bad_intervals"][0]
    assert t0 <= 2.1 and t1 >= 3.6 and closure > 1.0
    outside = (stamps < 1.5) | (stamps > 4.3)
    assert e[outside].max() < 0.08, e.round(3)


@pytest.fixture(scope="module")
def moving_seq(tmp_path_factory):
    pytest.importorskip("rosbags")
    if not os.path.isdir(BAG):
        pytest.skip("sample bag not checked out")
    root = tmp_path_factory.mktemp("refine")
    subprocess.run([sys.executable, "tools/dog/rosbag_to_sequence.py", "--bag", BAG, "--out", str(root / "ins"), *TOPICS],
                   check=True, capture_output=True)
    shutil.copytree(root / "ins", root / "refined")
    return root


def _preview(seqs, out):
    subprocess.run([sys.executable, "tools/dog/sequence_preview.py", "--seq", *map(str, seqs), "--out", str(out)],
                   check=True, capture_output=True)
    return {c["pose_source"]: c for c in json.load(open(out / "checks.json"))}


def test_sample_bag_shift_check_before_and_after(moving_seq, tmp_path):
    root = moving_seq
    odom0 = np.load(root / "refined" / "odom.npy")
    subprocess.run([sys.executable, "tools/dog/refine_poses.py", "--seq", str(root / "refined")], check=True,
                   capture_output=True)
    rd = SequenceReader(str(root / "refined"))
    assert rd.meta["pose_source"] == "kiss-icp+ins" and np.array_equal(np.load(root / "refined" / "odom_ins.npy"), odom0)
    assert rd.meta["pose_refine"]["correction_max_m"] > 0.4                 # the braking at the end of the bag
    first = rd.odom.copy()
    refine_poses.refine(str(root / "refined"))                              # input stays odom_ins.npy
    assert np.allclose(SequenceReader(str(root / "refined")).odom, first, atol=1e-6)

    c = _preview([root / "ins", root / "refined"], tmp_path / "preview")
    assert c["ins"]["shift_p90_m"] > 0.35 and "shift_p90_m=FAIL" in c["ins"]["problems"]
    assert c["kiss-icp+ins"]["shift_p90_m"] < 0.2 and c["kiss-icp+ins"]["ins_step_m"] < 0.05
    assert c["kiss-icp+ins"]["status"] == "OK"
