"""kygi665 INSPVAX -> localization pose in a local ENU frame.

Matches the onboard parser in ``kygi665_msgs/parse_inspvx``: a solution is
usable when ``ins_status`` is INS_SOLUTION_GOOD (3), position is
latitude/longitude/height converted to ENU about the first fix, and heading
is ``yaw = (-azimuth + 180) deg`` about +z. Velocity is east/north/up, which
is already the ENU world frame. Roll / pitch are not used: on the moving
sample bag the INS pitch swings by 10 deg while braking although the gyro
and the ground plane show the body tilting by less than 2 deg; the residual
tilt is levelled from the ground instead (``DetFrameConfig.auto_level``).

Sensor axes and units, measured on the moving sample bag (robot driving
straight at 1.5-2.5 m/s, see docs/DOG_PIPELINE.md section 9):

* the RTK IMU (``/rte/rdap/rtk/imus/data_raw``) publishes angular velocity
  in deg/s and acceleration in g, not the sensor_msgs units (gyro z
  integrates to the azimuth change at scale 1.06 when read as deg/s; |acc| is
  1.0 at rest). Its axes are the INS axes: x right, y along the azimuth, z up;
* the body frame above (yaw = 180 deg - azimuth) is the INS frame turned by
  180 deg about z, so body-frame rates are ``(-x, -y, z)``;
* the XT32 axes match the INS axes as well: scan matching puts the motion on
  lidar -y while the INS puts it on body +y, so ``T_body_lidar`` has a
  180 deg yaw (``KYGI_LIDAR_RPY``). A static recording cannot show this; a
  wrong yaw there only rotates the map, while the robot moves it shifts the
  older sweeps the wrong way and static objects get the robot's speed;
* the INSPVAX position is not that of the lidar (the body origin) but of a
  point 0.54 m behind it (``KYGI_INS_LEVER_ARM``): against the lidar
  odometry, the INS position moves by R(yaw) @ lever arm whenever the robot
  turns (up to 1.1 m after a U-turn). Fitted on the 14 0921 scenes that turn
  by more than 45 deg: (-0.06, -0.54) m, every leave-one-out fit within 3 cm
  (docs/BAG_SPLIT.md 5.6). ``apply_lever_arm`` moves the rows to the body
  origin; the height of the point does not matter with a heading-only
  attitude.
"""
from pathlib import Path

import numpy as np

from .geometry import quat_to_R, rot_z
from .ros_utils import stamp_of

INS_SOLUTION_GOOD = 3
INS_RTKFIXED = 56                        # pos_type of an RTK-fixed INS solution
KYGI_GYRO_SCALE = np.pi / 180.0          # deg/s -> rad/s
R_BODY_KYGI_IMU = rot_z(np.pi)           # IMU / INS axes in the body frame
KYGI_LIDAR_RPY = (0.0, 0.0, np.pi)       # XT32 orientation in the body frame (roll, pitch, yaw) [rad]
KYGI_INS_LEVER_ARM = (-0.06, -0.54, 0.0)  # INSPVAX position point in the body frame [m]
_A = 6378137.0
_E2 = 6.69437999014e-3

_MSG_ORDER = (
    "Kygi665Header",
    "InertialSolutionStatus",
    "PositionOrVelocityType",
    "INSExtendedSolutionStatus",
    "INSPVAX",
)


def register_kygi(typestore, msg_dir=None):
    """Register kygi665_msgs on a rosbags typestore. Dependencies first."""
    from rosbags.typesys import get_types_from_msg

    root = Path(msg_dir) if msg_dir else Path(__file__).resolve().parents[1] / "kygi665_msgs" / "msg"
    for name in _MSG_ORDER:
        typestore.register(get_types_from_msg((root / f"{name}.msg").read_text(), f"kygi665_msgs/msg/{name}"))
    return typestore


def _lla_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)
    n = _A / np.sqrt(1.0 - _E2 * sl * sl)
    return np.array([(n + h) * cl * cn, (n + h) * cl * sn, (n * (1.0 - _E2) + h) * sl])


class EnuOrigin:
    def __init__(self, lat, lon, h):
        self.lat = float(lat)
        self.lon = float(lon)
        self.h = float(h)
        self.ecef = _lla_to_ecef(self.lat, self.lon, self.h)
        lat0, lon0 = np.radians(self.lat), np.radians(self.lon)
        sl, cl = np.sin(lat0), np.cos(lat0)
        sn, cn = np.sin(lon0), np.cos(lon0)
        self.R = np.array([[-sn, cn, 0.0], [-sl * cn, -sl * sn, cl], [cl * cn, cl * sn, sl]])

    def enu(self, lat, lon, h):
        return self.R @ (_lla_to_ecef(lat, lon, h) - self.ecef)


def gyro_from_kygi_imu(msg):
    """sensor_msgs/Imu from the kygi665 RTK unit -> body-frame angular velocity [rad/s]."""
    w = msg.angular_velocity
    return R_BODY_KYGI_IMU @ (np.array([w.x, w.y, w.z], dtype=np.float64) * KYGI_GYRO_SCALE)


def pose_from_inspvax(msg, origin, lever_arm=None):
    """Return (t, R, p, v) or None when the INS solution is not good. With ``lever_arm`` (the INS
    position point in the body frame) p is the body origin; v stays that of the INS point (off by
    yaw rate x lever arm, 0.3 m/s at 30 deg/s: ``apply_lever_arm`` corrects both on a whole series)."""
    if int(msg.ins_status.data) != INS_SOLUTION_GOOD:
        return None
    if origin is None:
        raise ValueError("origin is required")
    p = origin.enu(msg.latitude, msg.longitude, msg.height)
    R = rot_z((-float(msg.azimuth) + 180.0) * np.pi / 180.0)
    if lever_arm is not None:
        p = p - R @ np.asarray(lever_arm, np.float64)
    v = np.array([msg.east_velocity, msg.north_velocity, msg.up_velocity], dtype=np.float64)
    return stamp_of(msg.header), R, p, v


def apply_lever_arm(odom, lever_arm):
    """Odometry rows (t, xyz, quat xyzw, vel; the sequence ``odom.npy``) of the INS point -> rows of
    the body origin: position minus R @ lever_arm, velocity minus its time derivative (central
    differences). Returns a float64 copy; a zero lever arm returns it unchanged."""
    out = np.array(odom, np.float64, copy=True)
    L = np.asarray(lever_arm, np.float64)
    if len(out) == 0 or not np.any(L):
        return out
    off = np.array([quat_to_R(q) @ L for q in out[:, 4:8]])
    out[:, 1:4] -= off
    if len(out) > 1 and np.isfinite(out[:, 8:11]).all():
        out[:, 8:11] -= np.gradient(off, out[:, 0], axis=0)
    return out
