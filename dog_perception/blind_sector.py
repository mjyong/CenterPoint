"""The XT32's blind wedge behind the kygi robot, and which zone a position lies in.

Something on the robot's back blocks the lidar in a wedge behind it: on 0921 the azimuths 142-218 deg
(0 deg straight ahead, measured by tools/dog/sector_check.py, T13). A structure cut off by the wedge
ends abruptly at its edge, where a detector may take it for the end of a vehicle, so evaluations
count errors per zone: ``blind`` (inside the wedge), ``edge`` (the ``EDGE_DEG`` band of the visible
side next to it) and ``elsewhere``.

Azimuth is in the robot body frame, 0 deg along body +y (forward on the kygi robot), positive towards
body +x. The det frame follows the body heading (``preprocess/frames.py``), so det-frame positions
give the same azimuth.
"""
import numpy as np

BLIND_SECTOR_DEG = (142.0, 218.0)
EDGE_DEG = 20.0
ZONES = ("elsewhere", "edge", "blind")


def body_azimuth(xy):
    """Degrees, 0 along body +y (forward), positive towards body +x."""
    xy = np.asarray(xy, float)
    return np.degrees(np.arctan2(xy[..., 0], xy[..., 1]))


def in_intervals(az, intervals, pad=0.0):
    a = np.asarray(az, float)
    hit = np.zeros(a.shape, bool)
    for lo, hi in intervals:
        for shift in (-360.0, 0.0, 360.0):
            hit |= (a + shift >= lo - pad) & (a + shift <= hi + pad)
    return hit


def zone_of(xy, sector=BLIND_SECTOR_DEG, edge=EDGE_DEG):
    """Zone name of each body / det-frame position (..., 2)."""
    az = body_azimuth(xy)
    lo, hi = sector
    blind = in_intervals(az, [(lo, hi)])
    near = in_intervals(az, [(lo - edge, lo), (hi, hi + edge)])
    out = np.where(blind, ZONES[2], np.where(near, ZONES[1], ZONES[0]))
    return out.item() if out.ndim == 0 else out
