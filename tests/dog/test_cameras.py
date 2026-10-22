"""Fisheye intrinsics and mounting (dog_perception/cameras.py)."""
import numpy as np
import pytest

from dog_perception.cameras import H190XA_G12161021, KYGI_FISHEYES, T_BASE_BODY, T_body_camera
from dog_perception.geometry import transform_points


def test_factory_intrinsics_match_the_datasheet_field_of_view():
    cam = H190XA_G12161021
    uv, ok = cam.project([[0.0, 0.0, 1.0]])
    assert np.allclose(uv, [[cam.cx, cam.cy]]) and ok.all()
    th = np.radians(np.arange(0.0, 100.0, 0.05))
    x = cam.project(np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)]))[0][:, 0]
    y = cam.project(np.column_stack([np.zeros_like(th), np.sin(th), np.cos(th)]))[0][:, 1]
    hfov = 2 * np.degrees(th[np.searchsorted(x, cam.width)])
    vfov = 2 * np.degrees(th[np.searchsorted(y, cam.height)])
    assert abs(hfov - 196) < 6 and abs(vfov - 154) < 4          # datasheet: 196 x 154 deg
    assert np.all(np.diff(x) > 0)                                   # monotonic to 100 deg
    behind = cam.project([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])[1]
    assert not behind[0] and behind[1]                              # 90 deg off axis is still in the image


def test_matches_opencv_fisheye():
    cv2 = pytest.importorskip("cv2")
    cam = H190XA_G12161021
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3))
    p[:, 2] = np.abs(p[:, 2]) + 0.3
    ref, _ = cv2.fisheye.projectPoints(p[None], np.zeros(3), np.zeros(3), cam.K, np.array(cam.k))
    assert np.allclose(cam.project(p)[0], ref[0], atol=1e-6)


def test_mounting_looks_along_each_axis():
    front, left, right = KYGI_FISHEYES["front"], KYGI_FISHEYES["left"], KYGI_FISHEYES["right"]
    for m, target in ((front, (10.0, 0.0, 0.0424)), (left, (0.0, 10.0, -0.0279)), (right, (0.0, -10.0, -0.0279))):
        xyz = transform_points(np.linalg.inv(m.T_base_camera()), np.array([target]))
        uv, ok = m.intrinsics.project(xyz)
        assert ok.all() and np.allclose(uv, [[m.intrinsics.cx, m.intrinsics.cy]], atol=1e-6)
    # the front camera's image right is the robot's right, image down is down
    uv, _ = front.intrinsics.project(transform_points(np.linalg.inv(front.T_base_camera()),
                                                      np.array([[10.0, -1.0, -1.0]])))
    assert uv[0, 0] > front.intrinsics.cx and uv[0, 1] > front.intrinsics.cy
    # XT32 at (0.319, 0.043, 0.143) in the base frame; the front camera 0.13 m ahead of it, 0.10 m lower
    assert np.allclose(T_BASE_BODY[:3, 3], [0.319, 0.0426, 0.143], atol=1e-3)
    assert np.allclose(T_body_camera(front)[:3, 3], [0.0426, 0.1286, -0.1006], atol=1e-3)
    assert KYGI_FISHEYES["rear"].topic is None
