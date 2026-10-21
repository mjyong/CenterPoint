"""Lidars of the kygi665 robot: topic, per-point time format, mounting and self filter.

Body frame B (as in ``inspvax.py``): x right, y forward, z up, origin at the lidar, so the
extrinsic is a rotation only and ``base_height`` is the lidar's height above the ground.

``xt32``: Hesai XT32 on top (0921 recordings), absolute float64 ``timestamp`` in seconds,
mounted turned 180 deg about z.

``mid360``: Livox Mid-360 at the front (1016 recordings, ``/robot/topic/rdap/front_point_cloud2``,
livox_ros_driver2 PointCloud2: x, y, z, intensity, tag, line 0-3, ``timestamp`` in absolute
nanoseconds; ~20,000 points per 0.1 s of which about half are empty returns). Mounted upside down
and tilted: on 1016 sensor_recording_03 the ground normal in the lidar frame is
(0.304, -0.025, -0.952), 162 deg from the lidar's +z, and the lidar is 0.52 m above the ground.
Its upward-looking side (+x, up to ~24 deg above the horizon) faces forward: towards -x the field
of view ends 10 deg below the horizon and returns stop at ~1.3 m (the robot's own body). Beyond
~3 m behind the robot the Mid-360 sees nothing. The rotation below is that estimate (roll / pitch
from the ground plane, forward = lidar +x on the ground, about +-10 deg), **provisional until the
calibrated lidar-to-IMU extrinsic is known**; sequences converted with it carry
``meta["extrinsic_provisional"] = True``. Livox ``tag`` bits 0-1 flag spatial noise; such points
are dropped.
"""
from dataclasses import dataclass, field

from .inspvax import KYGI_LIDAR_RPY


@dataclass(frozen=True)
class LidarPreset:
    name: str
    topic: str
    time_field: str = "timestamp"
    time_mode: str = "absolute"           # ros_utils.scan_from_msg
    noise_tag_mask: int = 0
    rpy: tuple = (0.0, 0.0, 0.0)          # lidar in body, rot_zyx(yaw, pitch, roll) [rad]
    t: tuple = (0.0, 0.0, 0.0)
    self_filter: dict = field(default_factory=dict)   # SelfFilterConfig fields; empty: the defaults
    provisional: bool = False             # mounting estimated from data, not calibrated


LIDARS = {
    "xt32": LidarPreset("XT32", "/robot/topic/rdap/top_point_cloud2", rpy=tuple(KYGI_LIDAR_RPY)),
    "mid360": LidarPreset(
        "MID360", "/robot/topic/rdap/front_point_cloud2", time_mode="absolute_ns", noise_tag_mask=0x03,
        rpy=(-3.1156, -0.3091, 1.5708),
        # the robot is behind the lidar: legs and body within ~1.1 m behind and 0.45 m to each side
        self_filter=dict(body_boxes=((-0.45, 0.45, -1.10, 0.15, -0.60, 0.30),), min_range=0.4),
        provisional=True),
}

# cameras recorded with the Mid-360 (1016): 1920 x 1536 fisheye, H.264 at 24 Hz (foxglove_msgs/CompressedVideo)
KYGI_CAMERA_TOPICS = ("/robot/topic/rdap/front_sensor_camera", "/robot/topic/rdap/left_sensor_camera",
                      "/robot/topic/rdap/right_sensor_camera")
COMPRESSED_VIDEO_MSG = "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n"


def register_compressed_video(typestore):
    """foxglove_msgs/msg/CompressedVideo on a rosbags typestore (the bags carry no definitions)."""
    from rosbags.typesys import get_types_from_msg

    typestore.register(get_types_from_msg(COMPRESSED_VIDEO_MSG, "foxglove_msgs/msg/CompressedVideo"))
    return typestore


def self_filter_config(meta):
    """SelfFilterConfig for a sequence: ``meta["self_filter"]`` (written by the converter) or the defaults."""
    from .preprocess import SelfFilterConfig

    sf = dict(meta.get("self_filter") or {})
    if "body_boxes" in sf:
        sf["body_boxes"] = tuple(tuple(float(v) for v in b) for b in sf["body_boxes"])
    return SelfFilterConfig(**sf)
