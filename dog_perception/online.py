"""The online node's message handling, without ROS (``tools/dog/ros2_node.py``).

Messages are read by attribute only, so the same code takes rclpy messages online and ``rosbags``
messages offline: ``tools/deploy/cpp_parity.py python --bag`` replays a bag through it in arrival
order, the reference for the C++ node (``deploy/ros2``, the same steps in ``dogperc/online.hpp``).

    inputs = OnlineInputs(node_parameters("kygi"), pipeline, log)
    inputs.on_imu(msg); inputs.on_inspvax(msg)        # or on_odom(msg) with localization odom
    out = inputs.on_cloud(msg)                        # PipelineOutput, or None while no pose covers the scan
"""
import numpy as np

from .geometry import quat_to_R
from .inspvax import (INS_RTKFIXED, INS_SOLUTION_GOOD, EnuOrigin, HeadingCheck, gyro_from_kygi_imu,
                      pose_from_inspvax)
from .node_config import ins_lever_arm
from .ros_utils import scan_from_msg, stamp_of

TIME_MODES = ("absolute", "absolute_ns", "relative", "relative_ns")


def _no_log(level, text, throttle=0.0):
    pass


class OnlineInputs:
    """``params``: the node parameters (``node_config.node_parameters``); ``log(level, text, throttle)``
    with level "info" / "warn" / "error" and ``throttle`` the seconds between repeats (0: always)."""

    def __init__(self, params, pipe, log=None):
        g = params.get
        self.params = params
        self.pipe = pipe
        self.log = log or _no_log
        self.localization = g("localization")
        if self.localization not in ("odom", "inspvax"):
            raise ValueError("localization must be odom or inspvax, got %r" % self.localization)
        if g("time_mode") not in TIME_MODES:
            raise ValueError("time_mode must be one of %s, got %r" % (", ".join(TIME_MODES), g("time_mode")))
        self.odom_twist = g("odom_twist")
        self.time_field, self.time_mode = g("time_field"), g("time_mode")
        self.noise_tag_mask = int(g("noise_tag_mask"))
        self.kygi_imu = g("imu_convention") == "kygi"
        self.lidar_time_offset = float(g("lidar_time_offset"))
        self.ins_lever_arm = ins_lever_arm(params)
        self.ins_azimuth_offset = float(g("ins_azimuth_offset_deg"))
        self.origin = None
        self.heading_check, self.heading_warned = HeadingCheck(), False

    def on_imu(self, msg):
        if self.kygi_imu:
            gyro = gyro_from_kygi_imu(msg)
        else:
            w = msg.angular_velocity
            gyro = (w.x, w.y, w.z)
        self.pipe.on_imu(stamp_of(msg.header), gyro)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        R = quat_to_R([q.x, q.y, q.z, q.w])
        v = None
        if self.odom_twist != "none":
            lin = msg.twist.twist.linear
            v = np.array([lin.x, lin.y, lin.z])
            if self.odom_twist == "body":
                v = R @ v
        self.pipe.on_odometry(stamp_of(msg.header), R, [p.x, p.y, p.z], v)

    def on_inspvax(self, msg):
        """Same reading as the bag converter: heading only, ENU origin at the first good fix."""
        if self.origin is None:
            if int(msg.ins_status.data) != INS_SOLUTION_GOOD:
                self.log("warn", "waiting for a good INS solution", 2.0)
                return
            self.origin = EnuOrigin(msg.latitude, msg.longitude, msg.height)
            self.log("info", "ENU origin %.8f %.8f %.2f" % (msg.latitude, msg.longitude, msg.height))
        got = pose_from_inspvax(msg, self.origin, self.ins_lever_arm, self.ins_azimuth_offset)
        if got is None:
            self.log("warn", "INS solution not good (ins_status %d)" % int(msg.ins_status.data), 2.0)
            return
        if int(msg.pos_type.data) != INS_RTKFIXED:
            self.log("warn", "RTK not fixed (pos_type %d)" % int(msg.pos_type.data), 5.0)
        t, R, p, v = got
        self.heading_check.add(R, v)
        if self.heading_check.wrong and not self.heading_warned:
            self.heading_warned = True
            self.log("error", "the robot walks backwards in the body frame: ins_azimuth_offset_deg %.0f does not "
                     "match the INS antennas (180 swapped as on 0921, 0 fixed)" % self.ins_azimuth_offset)
        self.pipe.on_odometry(t, R, p, v)

    def scan(self, msg):
        return scan_from_msg(msg, self.time_field, self.time_mode,
                             noise_tag_mask=self.noise_tag_mask).shifted(self.lidar_time_offset)

    def on_cloud(self, msg, detections=None):
        """``detections`` (det frame) bypass the detector, as ``PerceptionPipeline.on_scan``."""
        out = self.pipe.on_scan(self.scan(msg), detections=detections)
        if out is None:
            self.log("warn", "no pose for scan yet (waiting for odometry / IMU)", 2.0)
        return out
