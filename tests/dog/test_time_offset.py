"""Lidar-to-localization time offset: SequenceReader applies meta["lidar_time_offset"], and
tools/dog/time_offset_check.py finds an offset injected into a simulated sequence."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import time_offset_check as toc  # noqa: E402

from dog_perception.io import SequenceReader, odom_row, write_sequence  # noqa: E402
from dog_perception.sim import GaitModel, make_scenario  # noqa: E402

LAG = 0.05     # the simulated lidar stamps everything 50 ms late


def _sim_sequence(path, lag, n=20):
    # body yaw swings +-15 deg at 1 Hz (up to 94 deg/s): sweeps 0.5 s apart turn in opposite directions
    sim = make_scenario(seed=0, duration=n * 0.1 + 1, num_pedestrians=0, num_cyclists=0, num_vehicles=0,
                        gait=GaitModel(speed=1.0, yaw_rate=0.0, yaw_amp_deg=15.0, yaw_freq=1.0))
    scans = [sim.scan(k, t0=1.0)[0] for k in range(n)]
    it, iw = sim.imu(0.0, n * 0.1 + 1.6)
    odom = [odom_row(t, *sim.odometry(t)) for t in np.arange(0.0, n * 0.1 + 1.6, 0.01)]
    write_sequence(str(path), [s.shifted(lag) for s in scans], np.column_stack([it, iw]), odom,
                   dict(T_body_lidar=sim.T_bl, base_height=0.45))
    return scans


@pytest.fixture(scope="module")
def lagged(tmp_path_factory):
    path = tmp_path_factory.mktemp("lag") / "seq"
    return path, _sim_sequence(path, LAG)


def test_check_finds_the_injected_offset(lagged):
    path, _ = lagged
    res = toc.check(str(path), [-0.075, -0.05, -0.025, 0.0, 0.025])
    assert res["n_turn"] >= 10
    assert res["best_offset"] == pytest.approx(-LAG)
    assert res["turn_median_at_0"] > 0.3 and res["turn_median_at_best"] < 0.05
    v = toc.verdict(res["best_offset"], res["turn_median_at_0"], res["turn_median_at_best"], 1, 0.025)
    assert v.startswith("offset -0.050 s")


def test_meta_offset_is_applied_when_reading(lagged, tmp_path):
    path, truth = lagged
    meta_p = os.path.join(path, "meta.json")
    meta = json.load(open(meta_p))
    try:
        json.dump(dict(meta, lidar_time_offset=-LAG), open(meta_p, "w"))
        rd = SequenceReader(str(path))
        scans = [ev[1] for ev in rd.events() if ev[0] == "scan"]
        assert len(scans) == len(truth)
        for a, b in zip(scans, truth):
            assert a.stamp == pytest.approx(b.stamp) and np.allclose(a.point_times, b.point_times)
        res = toc.check(str(path), [-0.025, 0.0, 0.025])          # corrected: nothing left to find
        assert res["best_offset"] == 0.0 and res["turn_median_at_0"] < 0.05
    finally:
        json.dump(meta, open(meta_p, "w"))
