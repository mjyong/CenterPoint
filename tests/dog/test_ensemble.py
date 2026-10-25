"""Two teachers' detections fused (dog_perception/detection/ensemble.py)."""
import numpy as np
import pytest

from dog_perception.detection.boxes import Detections
from dog_perception.detection.ensemble import fuse


def _dets(rows):
    rows = np.array(rows, float).reshape(-1, 10)
    return Detections(rows[:, :7], rows[:, 7:9], rows[:, 9], np.zeros(len(rows), np.int64), subclasses=np.zeros(len(rows), np.int64))


def test_fuse_two_teachers():
    a = _dets([[10, 0, 0.8, 4.5, 1.9, 1.6, 0.10, 3, 0, 0.8],      # the car both see
               [30, 5, 0.8, 4.5, 1.9, 1.6, 0.00, 0, 0, 0.6]])     # only teacher A
    b = _dets([[10.4, 0.2, 0.9, 4.3, 1.8, 1.5, 0.10 + np.pi, 2, 0, 0.4],   # same car, heading read the other way
               [-8, -8, 0.8, 4.5, 1.9, 1.6, 0.0, 0, 0, 0.9],       # only teacher B
               [40, -10, 1.5, 12, 2.5, 3.5, 0.0, 0, 0, 0.5]])
    a2 = _dets([[44.5, -10.2, 1.5, 3.0, 2.4, 3.4, 0.0, 0, 0, 0.5]])          # the front of that long truck
    out, agree = fuse([a, b])
    assert len(out) == 4 and sorted(agree.tolist()) == [1, 1, 1, 2]
    car = out.boxes[np.argmax(agree)]
    assert np.allclose(car[:2], [(10 * 0.8 + 10.4 * 0.4) / 1.2, (0.2 * 0.4) / 1.2]) and abs(car[6] - 0.10) < 1e-9
    assert out.scores[np.argmax(agree)] == pytest.approx((0.8 + 0.4) / 2)
    assert sorted(out.scores[agree == 1].tolist()) == pytest.approx([0.25, 0.3, 0.45])   # half their own score
    out, agree = fuse([a2, b])                                                   # truck: centers 4.5 m apart
    assert (agree == 2).sum() == 1
    single, agree = fuse([a])
    assert single is a and agree.tolist() == [1, 1]
