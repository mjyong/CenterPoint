"""Fisheye cameras of the kygi665 robot: intrinsics and mounting (docs/SENSORS.md section 4).

Intrinsics: OpenCV fisheye (Kannala-Brandt) model, r = f * theta_d with
theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), no tangential terms.
The factory calibration of one H190XA (SN H190XA-G12161021, RMS 0.0122 px) is used for every
camera until each has its own; it gives a 192 x 152 deg field of view on the 1920 x 1536 image
(datasheet: 196 x 154 deg), and undistorting a 1016 front frame with it makes the railings and
road edges straight.

Mounting: positions in the base frame (x forward, y left, z up, origin the body centre, the
origin of the robot's IMU -> body extrinsic) from the robot team. The rotations are not calibrated
yet: each camera is taken to look level along its axis (front +x, left +y, right -y, rear -x).
The front camera looks roughly 6-7 deg down (road vanishing point on a 1016 frame), so boxes
drawn with these rotations are only a guide. The rear camera is not recorded.

Camera frame: OpenCV optical frame (x right in the image, y down, z along the optical axis).
"""
from dataclasses import dataclass

import numpy as np

from .geometry import make_T, rot_z


@dataclass(frozen=True)
class FisheyeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple                    # k1..k4 of the OpenCV fisheye model
    width: int = 1920
    height: int = 1536
    max_theta: float = np.radians(100.0)    # the polynomial is calibrated up to the image edge (~96 deg)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, xyz):
        """(N, 3) points in the camera frame -> (N, 2) pixels and (N,) bool: in front of the lens
        within ``max_theta`` of the axis and inside the image."""
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        theta = np.arctan2(rho, xyz[:, 2])
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = np.where(rho > 1e-12, theta_d / np.maximum(rho, 1e-12), 0.0)
        uv = np.column_stack([self.fx * xyz[:, 0] * scale + self.cx, self.fy * xyz[:, 1] * scale + self.cy])
        ok = (theta <= self.max_theta) & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv, ok


# factory calibration of SN H190XA-G12161021
H190XA_G12161021 = FisheyeIntrinsics(fx=509.2360850025, fy=509.3809872125, cx=958.9823613579, cy=766.4637605342,
                                     k=(0.1361352807, -0.0372498948, 0.0000678490, 0.0006005096))


@dataclass(frozen=True)
class FisheyeMount:
    name: str
    topic: str                  # None: not recorded
    position: tuple             # in the base frame [m]
    yaw_deg: float              # direction of the optical axis in the base frame (0 = forward)
    intrinsics: FisheyeIntrinsics = H190XA_G12161021
    rotation_calibrated: bool = False

    def T_base_camera(self):
        """4x4 pose of the camera (OpenCV optical frame) in the base frame, level, looking along
        ``yaw_deg``. Columns: image right, image down, optical axis."""
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        R = np.array([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
        return make_T(R, self.position)


KYGI_FISHEYES = {
    "front": FisheyeMount("front", "/robot/topic/rdap/front_sensor_camera", (0.4476, 0.0, 0.0424), 0.0),
    "left": FisheyeMount("left", "/robot/topic/rdap/left_sensor_camera", (0.0, 0.1165, -0.0279), 90.0),
    "right": FisheyeMount("right", "/robot/topic/rdap/right_sensor_camera", (0.0, -0.1165, -0.0279), -90.0),
    "rear": FisheyeMount("rear", None, (-0.4475, 0.0, 0.0424), 180.0),
}

# the code's body frame B (x right, y forward, z up, origin the XT32) in the base frame: robot
# calibration lidar -> IMU (T (-0.0213, 0.18, 0.009), 180 deg yaw) and IMU -> base
# (T (0.13898, 0.02129, 0.134), IMU y forward = base x)
T_BASE_IMU = make_T(np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), (0.13898, 0.02129, 0.134))
T_IMU_LIDAR = make_T(rot_z(np.pi), (-0.0213, 0.18, 0.009))
T_BASE_BODY = T_BASE_IMU @ make_T(np.eye(3), T_IMU_LIDAR[:3, 3])     # B axes = IMU axes, origin at the lidar


def T_body_camera(mount):
    """Pose of a camera (optical frame) in the code's body frame B."""
    return np.linalg.inv(T_BASE_BODY) @ mount.T_base_camera()
