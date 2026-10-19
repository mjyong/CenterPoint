"""ROS 2 node running the four-stage pipeline online.

    ros2 run --prefix 'python3' ... or simply:
    python3 tools/dog/ros2_node.py --ros-args -p detector:=pillar -p checkpoint:=pp.pth \
        -p lidar_topic:=/lidar_points -p imu_topic:=/imu/data -p odom_topic:=/Odometry \
        -p extrinsic_t:="[0.2, 0.0, 0.15]" -p base_height:=0.45

kygi665 robot (RTK INSPVAX localization; the det frame is built exactly like the training
sequences: base height 0.72 m, auto_ground, auto_level - see dog_perception/node_config.py):

    python3 tools/dog/ros2_node.py --ros-args -p robot:=kygi -p config:=dog_pillar -p checkpoint:=<weights>

Publishes (world = LIO odom frame, ``world_frame`` parameter):
    ~/tracks        visualization_msgs/MarkerArray  boxes + ids + velocity arrows
    ~/predictions   visualization_msgs/MarkerArray  one line strip per predicted mode
    ~/cloud         sensor_msgs/PointCloud2         (debug) accumulated det-frame cloud
Planner-facing data in the gravity-aligned local frame is available from
``PerceptionPipeline.to_local``; wire it into your own message type.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import rclpy  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Imu, PointCloud2, PointField  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from dog_perception.geometry import quat_to_R  # noqa: E402
from dog_perception.inspvax import INS_RTKFIXED, INS_SOLUTION_GOOD, EnuOrigin, HeadingCheck, pose_from_inspvax  # noqa: E402
from dog_perception.node_config import node_parameters, preprocess_config  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.ros_utils import scan_from_msg  # noqa: E402

COLORS = {"vehicle": (0.2, 0.6, 1.0), "pedestrian": (1.0, 0.3, 0.3), "cyclist": (1.0, 0.8, 0.1)}


class DogPerceptionNode(Node):
    def __init__(self):
        super().__init__("dog_perception")
        # robot:=kygi presets topics, localization, extrinsic and det frame (dog_perception/node_config.py)
        robot = self.declare_parameter("robot", "").value
        params = {n: self.declare_parameter(n, d).value for n, d in node_parameters(robot).items()}
        g = params.get
        self.get_logger().info("robot %r: base_height %.2f, auto_ground %s, auto_level %s, localization %s" % (
            robot, g("base_height"), g("auto_ground"), g("auto_level"), g("localization")))

        from dog_perception.detection import build_detector
        kind = g("config") or g("detector")
        det = build_detector(kind, checkpoint=g("checkpoint") or None, device=g("device") or None)
        pcfg = preprocess_config(params)
        predictor = "imm"
        if g("predictor_model"):
            from dog_perception.prediction.learned import LearnedPredictor
            predictor = LearnedPredictor(g("predictor_model"))
        self.pipe = PerceptionPipeline(pcfg, det, predictor=predictor)
        self.world_frame = g("world_frame")
        self.odom_twist = g("odom_twist")
        self.time_field, self.time_mode = g("time_field"), g("time_mode")
        self.publish_cloud = bool(g("publish_cloud"))
        self.kygi_imu = g("imu_convention") == "kygi"
        self.lidar_time_offset = float(g("lidar_time_offset"))
        self.ins_lever_arm = np.asarray(g("ins_lever_arm"), np.float64)
        self.ins_azimuth_offset = float(g("ins_azimuth_offset_deg"))
        self.heading_check, self.heading_warned = HeadingCheck(), False

        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, g("imu_topic"), self.on_imu, 400)
        if g("localization") == "inspvax":
            from kygi665_msgs.msg import INSPVAX     # built in the robot's ROS workspace
            self.origin = None
            self.create_subscription(INSPVAX, g("inspvax_topic"), self.on_inspvax, 200)
        elif g("localization") == "odom":
            self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 50)
        else:
            raise ValueError("localization must be odom or inspvax, got %r" % g("localization"))
        self.create_subscription(PointCloud2, g("lidar_topic"), self.on_cloud, best_effort)
        self.pub_tracks = self.create_publisher(MarkerArray, "~/tracks", 5)
        self.pub_preds = self.create_publisher(MarkerArray, "~/predictions", 5)
        self.pub_cloud = self.create_publisher(PointCloud2, "~/cloud", 2)
        self.get_logger().info("dog perception up: detector=%s" % kind)

    @staticmethod
    def _t(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_imu(self, msg):
        if self.kygi_imu:
            from dog_perception.inspvax import gyro_from_kygi_imu
            gyro = gyro_from_kygi_imu(msg)
        else:
            w = msg.angular_velocity
            gyro = (w.x, w.y, w.z)
        self.pipe.on_imu(self._t(msg.header.stamp), gyro)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        R = quat_to_R([q.x, q.y, q.z, q.w])
        v = None
        if self.odom_twist != "none":
            l = msg.twist.twist.linear
            v = np.array([l.x, l.y, l.z])
            if self.odom_twist == "body":
                v = R @ v
        self.pipe.on_odometry(self._t(msg.header.stamp), R, [p.x, p.y, p.z], v)

    def on_inspvax(self, msg):
        """Same reading as the bag converter: heading only, ENU origin at the first good fix."""
        if self.origin is None:
            if int(msg.ins_status.data) != INS_SOLUTION_GOOD:
                self.get_logger().warn("waiting for a good INS solution", throttle_duration_sec=2.0)
                return
            self.origin = EnuOrigin(msg.latitude, msg.longitude, msg.height)
            self.get_logger().info("ENU origin %.8f %.8f %.2f" % (msg.latitude, msg.longitude, msg.height))
        got = pose_from_inspvax(msg, self.origin, self.ins_lever_arm, self.ins_azimuth_offset)
        if got is None:
            self.get_logger().warn("INS solution not good (ins_status %d)" % int(msg.ins_status.data),
                                   throttle_duration_sec=2.0)
            return
        if int(msg.pos_type.data) != INS_RTKFIXED:
            self.get_logger().warn("RTK not fixed (pos_type %d)" % int(msg.pos_type.data), throttle_duration_sec=5.0)
        t, R, p, v = got
        self.heading_check.add(R, v)
        if self.heading_check.wrong and not self.heading_warned:
            self.heading_warned = True
            self.get_logger().error("the robot walks backwards in the body frame: ins_azimuth_offset_deg %.0f does "
                                    "not match the INS antennas (180 swapped as on 0921, 0 fixed)" % self.ins_azimuth_offset)
        self.pipe.on_odometry(t, R, p, v)

    def on_cloud(self, msg):
        scan = scan_from_msg(msg, self.time_field, self.time_mode).shifted(self.lidar_time_offset)
        out = self.pipe.on_scan(scan)
        if out is None:
            self.get_logger().warn("no pose for scan yet (waiting for odometry / IMU)", throttle_duration_sec=2.0)
            return
        self.publish(out, msg.header.stamp)
        self.get_logger().info(
            "%d dets, %d tracks | " % (len(out.detections), len(out.tracks)) +
            " ".join("%s %.1f" % (k, v) for k, v in out.timings.items() if k.endswith("_ms")),
            throttle_duration_sec=1.0)

    def publish(self, out, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for s in out.tracks:
            color = COLORS.get(s.name, (1.0, 1.0, 1.0))
            m = Marker()
            m.header.frame_id, m.header.stamp = self.world_frame, stamp
            m.ns, m.id, m.type = "box", s.track_id, Marker.CUBE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, s.position)
            m.pose.orientation.z, m.pose.orientation.w = float(np.sin(s.yaw / 2)), float(np.cos(s.yaw / 2))
            m.scale.x, m.scale.y, m.scale.z = map(float, s.size)
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.25 if s.coasting else 0.6
            arr.markers.append(m)
            txt = Marker()
            txt.header = m.header
            txt.ns, txt.id, txt.type = "id", s.track_id, Marker.TEXT_VIEW_FACING
            txt.pose.position.x, txt.pose.position.y = float(s.position[0]), float(s.position[1])
            txt.pose.position.z = float(s.position[2] + s.size[2])
            txt.scale.z = 0.5
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = "%s %d %.1fm/s" % (s.name[:3], s.track_id, float(np.hypot(*s.velocity)))
            arr.markers.append(txt)
        self.pub_tracks.publish(arr)

        parr = MarkerArray()
        parr.markers.append(clear)
        k = 0
        for pr in out.predictions:
            for mode, prob in zip(pr.modes, pr.probs):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.world_frame, stamp
                m.ns, m.id, m.type = "pred", k, Marker.LINE_STRIP
                k += 1
                m.scale.x = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, float(0.2 + 0.8 * prob)
                m.points = [Point(x=float(x), y=float(y), z=0.2) for x, y in mode]
                parr.markers.append(m)
        self.pub_preds.publish(parr)

        if self.publish_cloud:
            pts = out.frame.points
            msg = PointCloud2()
            msg.header.frame_id, msg.header.stamp = "dog_det", stamp
            msg.height, msg.width = 1, len(pts)
            msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                          for i, n in enumerate(("x", "y", "z", "intensity", "dt"))]
            msg.point_step, msg.row_step = 20, 20 * len(pts)
            msg.is_bigendian, msg.is_dense = False, True
            msg.data = pts.astype(np.float32).tobytes()
            self.pub_cloud.publish(msg)


def main():
    rclpy.init()
    node = DogPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
