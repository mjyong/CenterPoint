"""ROS 2 node running the four-stage pipeline online.

    ros2 run --prefix 'python3' ... or simply:
    python3 tools/dog/ros2_node.py --ros-args -p detector:=pillar -p checkpoint:=pp.pth \
        -p lidar_topic:=/lidar_points -p imu_topic:=/imu/data -p odom_topic:=/Odometry \
        -p extrinsic_t:="[0.2, 0.0, 0.15]" -p base_height:=0.45

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

from dog_perception.geometry import make_T, quat_to_R, rot_zyx  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig  # noqa: E402
from dog_perception.ros_utils import scan_from_msg  # noqa: E402

COLORS = {"vehicle": (0.2, 0.6, 1.0), "pedestrian": (1.0, 0.3, 0.3), "cyclist": (1.0, 0.8, 0.1)}


class DogPerceptionNode(Node):
    def __init__(self):
        super().__init__("dog_perception")
        p = self.declare_parameter
        p("detector", "pillar")
        p("config", "")
        p("checkpoint", "")
        p("device", "")
        p("predictor_model", "")
        p("lidar_topic", "/lidar_points")
        p("imu_topic", "/imu/data")
        p("odom_topic", "/Odometry")
        p("odom_twist", "none")           # none | body | world
        p("time_field", "timestamp")
        p("time_mode", "absolute")
        p("extrinsic_t", [0.0, 0.0, 0.0])
        p("extrinsic_rpy", [0.0, 0.0, 0.0])
        p("base_height", 0.45)
        p("num_sweeps", 5)
        p("world_frame", "camera_init")
        p("publish_cloud", False)
        g = lambda n: self.get_parameter(n).value

        from dog_perception.detection import build_detector
        kind = g("config") or g("detector")
        det = build_detector(kind, checkpoint=g("checkpoint") or None, device=g("device") or None)
        rpy, t = g("extrinsic_rpy"), g("extrinsic_t")
        pcfg = PreprocessConfig(T_body_lidar=make_T(rot_zyx(rpy[2], rpy[1], rpy[0]), t),
                                num_sweeps=int(g("num_sweeps")),
                                det_frame=DetFrameConfig(base_height=float(g("base_height"))))
        predictor = "imm"
        if g("predictor_model"):
            from dog_perception.prediction.learned import LearnedPredictor
            predictor = LearnedPredictor(g("predictor_model"))
        self.pipe = PerceptionPipeline(pcfg, det, predictor=predictor)
        self.world_frame = g("world_frame")
        self.odom_twist = g("odom_twist")
        self.time_field, self.time_mode = g("time_field"), g("time_mode")
        self.publish_cloud = bool(g("publish_cloud"))

        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, g("imu_topic"), self.on_imu, 400)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 50)
        self.create_subscription(PointCloud2, g("lidar_topic"), self.on_cloud, best_effort)
        self.pub_tracks = self.create_publisher(MarkerArray, "~/tracks", 5)
        self.pub_preds = self.create_publisher(MarkerArray, "~/predictions", 5)
        self.pub_cloud = self.create_publisher(PointCloud2, "~/cloud", 2)
        self.get_logger().info("dog perception up: detector=%s" % kind)

    @staticmethod
    def _t(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_imu(self, msg):
        w = msg.angular_velocity
        self.pipe.on_imu(self._t(msg.header.stamp), (w.x, w.y, w.z))

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

    def on_cloud(self, msg):
        scan = scan_from_msg(msg, self.time_field, self.time_mode)
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
