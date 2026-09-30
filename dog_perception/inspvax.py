"""kygi665 INSPVAX -> localization pose in a local ENU frame.

Matches the onboard parser in ``kygi665_msgs/parse_inspvx``: a solution is
usable when ``ins_status`` is INS_SOLUTION_GOOD (3), position is
latitude/longitude/height converted to ENU about the first fix, and heading
is ``yaw = (-azimuth + 180) deg`` about +z. Velocity is east/north/up, which
is already the ENU world frame.
"""
from pathlib import Path

import numpy as np

from .geometry import rot_z
from .ros_utils import stamp_of

INS_SOLUTION_GOOD = 3
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


def pose_from_inspvax(msg, origin):
    """Return (t, R, p, v) or None when the INS solution is not good."""
    if int(msg.ins_status.data) != INS_SOLUTION_GOOD:
        return None
    if origin is None:
        raise ValueError("origin is required")
    p = origin.enu(msg.latitude, msg.longitude, msg.height)
    yaw = (-float(msg.azimuth) + 180.0) * np.pi / 180.0
    v = np.array([msg.east_velocity, msg.north_velocity, msg.up_velocity], dtype=np.float64)
    return stamp_of(msg.header), rot_z(yaw), p, v
