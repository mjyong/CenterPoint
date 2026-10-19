"""tools/dog/sequence_meta.py --fix-imu-axes on a sequence converted with the old kygi IMU axes."""
import json
import os
import subprocess
import sys

import numpy as np

from dog_perception.inspvax import forward_speed
from dog_perception.io import odom_row, write_sequence


def test_fix_imu_axes_flips_old_kygi_sequences_once(tmp_path):
    imu = np.column_stack([np.arange(5) * 0.01, [0.1] * 5, [-0.2] * 5, [0.3] * 5])
    old, generic = str(tmp_path / "old"), str(tmp_path / "generic")
    write_sequence(old, [], imu=imu, meta=dict(lat0=29.7, lon0=106.8, h0=217.0))
    write_sequence(generic, [], imu=imu, meta={})                  # not kygi: left alone
    cmd = [sys.executable, "tools/dog/sequence_meta.py", "--seq", old, generic, "--fix-imu-axes"]
    for _ in range(2):                                              # idempotent
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        assert np.allclose(np.load(os.path.join(old, "imu.npy"))[:, 1:], [[-0.1, 0.2, 0.3]] * 5)
        assert json.load(open(os.path.join(old, "meta.json")))["imu_axes"] == "body"
        assert np.array_equal(np.load(os.path.join(generic, "imu.npy")), imu)
    assert "body" in out


def test_forward_speed_tells_the_heading_convention():
    from dog_perception.geometry import rot_z
    yaw = 0.7                                                       # body +y points to 0.7 + 90 deg in ENU
    fwd = 1.5 * np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2), 0.0])
    rows = np.array([odom_row(0.01 * k, rot_z(yaw), [0, 0, 0], fwd) for k in range(50)])
    assert abs(forward_speed(rows) - 1.5) < 1e-9
    flipped = np.array([odom_row(0.01 * k, rot_z(yaw + np.pi), [0, 0, 0], fwd) for k in range(50)])
    assert forward_speed(flipped) < 0
    assert forward_speed(np.column_stack([rows[:, :8], np.zeros((50, 3))])) is None   # standing still


def test_azimuth_offset_heading_and_online_check():
    """Dual antennas swapped (0921): heading = 180 deg - azimuth; fixed: -azimuth. The robot then
    walks along body +y either way; HeadingCheck flags the wrong offset."""
    from types import SimpleNamespace

    from dog_perception.inspvax import (KYGI_IMU_IN_BODY, KYGI_INS_LEVER_ARM, EnuOrigin, HeadingCheck,
                                        default_lever_arm, pose_from_inspvax, turn_heading)
    origin = EnuOrigin(29.7, 106.8, 217.0)
    travel = 30.0                                                   # azimuth of the direction of travel
    v = 1.5 * np.array([np.sin(np.radians(travel)), np.cos(np.radians(travel)), 0.0])   # east, north

    def msg(az):
        return SimpleNamespace(ins_status=SimpleNamespace(data=3), latitude=29.7, longitude=106.8, height=217.0,
                               azimuth=az, east_velocity=v[0], north_velocity=v[1], up_velocity=0.0,
                               header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))
    _, R_swapped, _, _ = pose_from_inspvax(msg(travel + 180.0), origin)            # 0921: points backwards
    _, R_fixed, _, _ = pose_from_inspvax(msg(travel), origin, azimuth_offset_deg=0.0)
    assert np.allclose(R_swapped, R_fixed) and R_fixed[:2, 1] @ v[:2] > 1.49        # body +y forward
    _, R_wrong, _, _ = pose_from_inspvax(msg(travel), origin)                         # fixed antennas, old offset
    check = HeadingCheck(min_rows=10)
    for _ in range(10):
        check.add(R_wrong, v)
    assert check.wrong
    rows = np.array([odom_row(0.0, R_wrong, [0, 0, 0], v)])
    assert np.allclose(quat_R(turn_heading(rows, np.pi)[0, 4:8]), R_fixed)
    assert default_lever_arm(180) == KYGI_INS_LEVER_ARM and default_lever_arm(0.0) == KYGI_IMU_IN_BODY


def quat_R(q):
    from dog_perception.geometry import quat_to_R
    return quat_to_R(q)
