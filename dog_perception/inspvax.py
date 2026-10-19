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
  1.0 at rest). Its axes are the body axes: x right, y forward, z up
  (``R_BODY_KYGI_IMU`` is the identity). While the robot brakes, the ground
  plane shows the body pitching nose-down by 1.5 deg and the integrated gyro
  x by 1.6 deg (0.1-0.3 s changes correlate at 0.85-0.94); the accelerometer
  y follows the forward acceleration of the lidar odometry (correlation
  0.63, -2 to -4 m/s^2 while braking), x does not. The robot calibration
  says the same (lidar -> IMU: 180 deg yaw, like ``KYGI_LIDAR_RPY``). Until
  2026-10 x and y were negated here (taken from the azimuth below, which
  only the yaw rate could confirm);
* the INSPVAX azimuth points backwards, along the IMU's -y: the body frame
  (yaw = 180 deg - azimuth) has +y forward, and the INS velocity in it is
  along +y. The INS also integrates the braking the wrong way (its speed
  goes 1.9 -> 2.5 m/s while the lidar odometry shows 2.1 -> 0.1 m/s), as
  if it used the IMU turned by 180 deg about z: most likely its mounting is
  configured that way (docs/BAG_SPLIT.md 3.5). If that gets fixed on the
  robot, the azimuth points forward and the heading becomes -azimuth:
  ``forward_speed`` (checked by the converter) turns negative then;
* the XT32 axes are the IMU axes turned by 180 deg about z: scan matching
  puts the motion on lidar -y while the INS puts it on body +y, so
  ``T_body_lidar`` has a 180 deg yaw (``KYGI_LIDAR_RPY``). A static
  recording cannot show this; a wrong yaw there only rotates the map, while
  the robot moves it shifts the older sweeps the wrong way and static
  objects get the robot's speed;
* the INSPVAX position is not that of the lidar (the body origin) but of a
  point 0.54 m behind it (``KYGI_INS_LEVER_ARM``): against the lidar
  odometry, the INS position moves by R(yaw) @ lever arm whenever the robot
  turns (up to 1.1 m after a U-turn). Fitted on the 14 0921 scenes that turn
  by more than 45 deg: (-0.06, -0.54) m, every leave-one-out fit within 3 cm
  (docs/BAG_SPLIT.md 5.6). ``apply_lever_arm`` moves the rows to the body
  origin; the height of the point does not matter with a heading-only
  attitude. The robot's calibration puts the IMU only 0.18 m behind the lidar
  (lidar -> IMU: T = (-0.0213, 0.18, 0.009) m, R = 180 deg yaw; in the body
  frame (0.02, -0.18) m). That value explains less of the 0921 offsets (rms
  0.086 m against 0.076 m, 0.097 m without a lever arm), so the INSPVAX point
  is not the IMU centre; most likely the GNSS antenna, if the INS has no or a
  wrong IMU -> antenna offset configured. The fitted value is used.
"""
from pathlib import Path

import numpy as np

from .geometry import quat_to_R, rot_z
from .ros_utils import stamp_of

INS_SOLUTION_GOOD = 3
INS_RTKFIXED = 56                        # pos_type of an RTK-fixed INS solution
KYGI_GYRO_SCALE = np.pi / 180.0          # deg/s -> rad/s
R_BODY_KYGI_IMU = np.eye(3)              # RTK IMU axes in the body frame (x right, y forward, z up)
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
    """sensor_msgs/Imu from the kygi665 RTK unit -> body-frame angular velocity [rad/s]
    (``meta["imu_axes"] = "body"`` marks sequences converted with these axes)."""
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


def forward_speed(odom, min_speed=0.5):
    """Median speed [m/s] along the body +y axis over the rows moving faster than ``min_speed``
    (None if none): positive when the heading convention is right (the robot walks forward)."""
    v = np.asarray(odom, np.float64)[:, 8:10]
    m = np.isfinite(v).all(axis=1) & (np.hypot(v[:, 0], v[:, 1]) > min_speed)
    if not m.any():
        return None
    fwd = [(quat_to_R(q)[:2, 1] @ vi) for q, vi in zip(np.asarray(odom)[m, 4:8], v[m])]
    return float(np.median(fwd))
