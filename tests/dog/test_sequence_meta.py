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
