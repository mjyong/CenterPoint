"""tools/dog/scene_stats.py: the per-scene counts used to choose scenes for review (DATA_COLLECTION_PLAN 3.4)."""
import os
import sys

import numpy as np

from .conftest import ROOT


def _obj(tid, x, moving, sub="car", label="vehicle", review=False):
    return dict(track_id=tid, label=label, subclass=sub, box=np.array([x, 0.0, 0.0, 4.5, 1.8, 1.6, 0.0]),
                velocity=np.array([3.0 if moving else 0.0, 0.0]), moving=moving, review=review)


def test_scene_stats_counts_moving_vehicles_near_the_robot():
    sys.path.insert(0, os.path.join(ROOT, "tools", "dog"))
    from scene_stats import scene_stats

    frames = []
    for k in range(40):
        objs = [_obj(1, 10.0, True),                              # moves at 10 m: 30 m and 15 m
                _obj(2, 25.0, True, "truck"),                     # moves at 25 m: 30 m only
                _obj(3, 50.0, True),                              # too far
                _obj(4, 8.0, k >= 15, "bus"),                     # parked, drives off after 1.5 s
                _obj(5, 5.0, False, review=True),                 # parked all along
                _obj(6, 5.0, True, "pedestrian", "pedestrian")]   # not a vehicle
        if k < 5:
            objs.append(_obj(7, 5.0, True))                       # moves for 0.5 s only
        frames.append(dict(stamp=100.0 + 0.1 * k, objects=objs, pose_bad=k == 39))
    row = scene_stats(dict(frames=frames), "s000")
    assert row["frames"] == 39 and row["seconds"] == 3.8 and row["vehicle_tracks"] == 6
    assert (row["moving_30m"], row["moving_15m"]) == (3, 2)
    assert (row["moving_30m_car"], row["moving_30m_truck"], row["moving_30m_bus"]) == (1, 1, 1)
    assert row["starts"] == 1 and row["review_tracks"] == 1
