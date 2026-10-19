"""tools/dog/refine_poses.py: lidar odometry (KISS-ICP) + RTK fusion.

* the fusion on synthetic data with ground truth: an INS that runs 0.5 m ahead for 1.5 s (as the kygi
  INS does while the robot brakes) is corrected back to the truth, and a gap where the odometry
  also fails is reported as bad; an odometry zigzag inside such a gap keeps its net motion;
* the INS lever arm: rows of a point 0.54 m behind the body origin circle around it while the
  robot turns; the fusion reports the lever arm left over, and none once it is applied;
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
    assert e.max() < 0.1, e.round(3)              # the odometry drifts 2 %; the INS only removes it slowly
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
    assert info["lo_steps_rejected"] == 0 and e.max() < 0.1, (info["lo_steps_rejected"], e.round(3))


def _lo_poses(xy):
    poses = np.repeat(np.eye(4)[None], len(xy), axis=0)
    poses[:, :2, 3] = xy[:, :2] - xy[0, :2]
    return poses


def test_fuse_lets_a_healthy_ins_carry_bad_odometry():
    """KISS-ICP jumps back and forth when a turn starts or stops (0921: 53 % of the steps above
    60 deg/s yaw rate) and can stall along a featureless wall. Where the INS is healthy it takes
    over those stretches."""
    stamps, poses, odom, truth, _ = _synthetic()
    t = odom[:, 0]
    odom = odom.copy()
    odom[:, 1] = 1.2 * t                                          # INS healthy everywhere
    lo = poses[:, :3, 3].copy()
    lo[:, :2] += truth[0]
    zig = (stamps > 4.0) & (stamps < 5.0)
    lo[zig, 1] += np.where(np.arange(zig.sum()) % 2, 0.4, -0.4)   # +-0.4 m back and forth
    stall = stamps > 5.5
    lo[stall, 0] = lo[np.argmax(stall), 0]                         # sees no motion along x
    xy, corr, info, diag = refine_poses.fuse(stamps, _lo_poses(lo), odom)
    e = np.linalg.norm(xy - truth, axis=1)
    assert info["ins_unhealthy_share"] == 0 and info["lo_unhealthy_share"] > 0.3
    assert e.max() < 0.15 and info["bad_intervals"] == [], (e.round(3), info["bad_intervals"])


def test_gyro_rotation_integrates_the_body_rates():
    from dog_perception.geometry import so3_log
    t = np.arange(0.0, 1.0, 0.01)
    imu = np.column_stack([t, np.full_like(t, 0.1), np.zeros_like(t), np.full_like(t, 0.5)])
    rv = so3_log(refine_poses.gyro_rotation(imu, 0.2, 0.3))
    assert np.allclose(rv, [0.01, 0.0, 0.05], atol=1e-6)
    assert np.allclose(refine_poses.gyro_rotation(np.zeros((0, 4)), 0.0, 0.1), np.eye(3))


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


def test_fuse_bridges_an_odometry_zigzag_inside_an_ins_gap():
    """While the robot brakes, the INS velocity overshoots (here: twice the true speed) and its
    position is not trusted; if KISS-ICP zigzags there too, its net motion over the zigzag is the
    best there is (v3 took the INS velocity there: 0.46 m off)."""
    stamps, poses, odom, truth, _ = _synthetic()
    t = odom[:, 0]
    odom = odom.copy()
    odom[(t > 2.0) & (t < 3.7), 8] *= 2.0
    lo = poses[:, :3, 3].copy()
    lo[:, :2] += truth[0]
    zig = (stamps > 2.6) & (stamps < 3.2)
    lo[zig, 1] += np.where(np.arange(zig.sum()) % 2, 0.3, -0.3)
    xy, corr, info, diag = refine_poses.fuse(stamps, _lo_poses(lo), odom)
    e = np.linalg.norm(xy - truth, axis=1)
    assert info["lo_steps_bridged"] >= zig.sum() and e.max() < 0.15, (info["lo_steps_bridged"], e.round(3))


def test_heading_follows_the_odometry_after_a_fast_turn():
    """After a fast turn the kygi INS heading is off by up to 10 deg and needs 3-5 s to come back
    (0921), while its position stays self-consistent. The fused heading follows the odometry
    (registered scans) from scan to scan; the INS only sets it in the long run."""
    stamps, poses, odom, truth, _ = _synthetic()
    odom = odom.copy()
    t = odom[:, 0]
    err = np.radians(np.interp(t, [0, 3.0, 3.3, 6.5, 99], [0, 0, 9.0, 0, 0]))   # INS heading excursion
    odom[:, 4:8] = [[0, 0, np.sin(e / 2), np.cos(e / 2)] for e in err]
    odom[:, 1] = 1.2 * t                                          # INS position healthy everywhere
    xy, corr, info, diag = refine_poses.fuse(stamps, poses, odom)
    # the excursion covers half of this 6 s sequence: it may tilt the whole of it by a degree or two,
    # but not bend it (what stacked sweeps and pinned labels need)
    assert info["heading_correction_max_deg"] > 6
    yaw = np.degrees(diag["yaw"])
    assert np.ptp(yaw) < 0.3 and np.abs(yaw).max() < 2.0, yaw.round(2)
    out = refine_poses.refined_odom(odom, stamps, xy, diag["yaw"])
    inside = (t >= stamps[0]) & (t <= stamps[-1])
    yaw_rows = np.degrees(2 * np.arctan2(out[:, 6], out[:, 7]))
    assert np.ptp(yaw_rows[inside]) < 0.3 and np.allclose(out[:, 4:6], 0)


def _turning(L, k=300):
    """30 s at 10 Hz: the robot walks 0.8 m/s along a figure of eight (heading spans 360 deg);
    INS rows (100 Hz) of the point ``L`` (body frame), odometry = truth."""
    from dog_perception.geometry import rot_z
    t = np.arange(0.0, 31.0, 0.01)
    yaw = np.pi / 2 + 2 * np.pi * np.sin(2 * np.pi * t / 30.0)
    vel = 0.8 * np.column_stack([np.cos(yaw), np.sin(yaw), 0 * t])
    p = np.concatenate([[[0.0, 0.0, 0.0]], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * 0.01, axis=0)])
    ins = np.array([odom_row(ti, rot_z(y), pi + rot_z(y) @ L, vi) for ti, y, pi, vi in zip(t, yaw, p, vel)])
    ins[:, 8:11] = np.gradient(ins[:, 1:4], t, axis=0)                  # the INS point's velocity
    stamps = 0.5 + 0.1 * np.arange(k)
    truth = np.column_stack([np.interp(stamps, t, p[:, i]) for i in (0, 1, 2)])
    R0 = rot_z(-np.interp(stamps[0], t, yaw))                           # odometry frame = first scan
    poses = np.repeat(np.eye(4)[None], k, axis=0)
    for i, s in enumerate(stamps):
        poses[i, :3, :3] = R0 @ rot_z(np.interp(s, t, yaw))
        poses[i, :3, 3] = R0 @ (truth[i] - truth[0])
    return stamps, poses, ins, truth[:, :2]


def test_lever_arm_is_reported_and_removed():
    from dog_perception.inspvax import KYGI_INS_LEVER_ARM, apply_lever_arm
    L = np.array(KYGI_INS_LEVER_ARM)
    stamps, poses, ins, truth = _turning(L)
    xy, _, info, _ = refine_poses.fuse(stamps, poses, ins)
    assert np.allclose(info["lever_residual_m"], L[:2], atol=0.02), info["lever_residual_m"]
    body = apply_lever_arm(ins, L)
    assert np.abs(np.hypot(body[:, 8], body[:, 9]) - 0.8).max() < 0.01        # the body origin's speed
    xy, _, info, _ = refine_poses.fuse(stamps, poses, body)
    assert np.allclose(info["lever_residual_m"], 0.0, atol=0.02) and np.abs(xy - truth).max() < 0.02
    assert np.array_equal(apply_lever_arm(ins, [0, 0, 0]), ins)


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
    assert rd.meta["ins_lever_arm"] == rd.meta["pose_refine"]["lever_arm"] == [-0.06, -0.54, 0.0]   # applied once
    assert rd.meta["pose_refine"]["correction_max_m"] > 0.4                 # the braking at the end of the bag
    first = rd.odom.copy()
    refine_poses.refine(str(root / "refined"))                              # input stays odom_ins.npy
    assert np.allclose(SequenceReader(str(root / "refined")).odom, first, atol=1e-6)

    c = _preview([root / "ins", root / "refined"], tmp_path / "preview")
    assert c["ins"]["shift_p90_m"] > 0.35 and "shift_p90_m=FAIL" in c["ins"]["problems"]
    assert c["kiss-icp+ins"]["shift_p90_m"] < 0.2 and c["kiss-icp+ins"]["ins_step_m"] < 0.05
    assert c["kiss-icp+ins"]["status"] == "OK"
