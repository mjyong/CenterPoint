"""tools/dog/sector_check.py: the blind wedge behind the robot, measured from the points."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import sector_check as sc  # noqa: E402


def test_blind_sector_across_the_back():
    rng = np.random.default_rng(0)
    frames = []
    for k in range(5):
        az = rng.uniform(-142, 142, 20000)                       # nothing beyond |az| 142: the wedge behind
        r = rng.uniform(2, 40, len(az))
        a = np.radians(az)
        pts = np.c_[r * np.sin(a), r * np.cos(a), np.zeros(len(az)), np.zeros(len(az)), np.zeros(len(az))]
        frames.append((pts, np.eye(4)))
    wedge = sc.blind_sector(frames, every=1)
    assert len(wedge) == 1 and abs(wedge[0][0] - 142) <= 1 and abs(wedge[0][1] - 218) <= 1
    assert sc.in_intervals([150, -150, 0, -100], wedge).tolist() == [True, True, False, False]
    assert sc.in_intervals([125], [(w[0] - 20, w[0]) for w in wedge]).tolist() == [True]
