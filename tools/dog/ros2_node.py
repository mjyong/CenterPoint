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
``PerceptionPipeline.to_local``; the C++ node (deploy/ros2, docs/DEPLOY_ORIN.md section 8) publishes it
as dog_perception_msgs/ObjectArray. The message handling is ``dog_perception/online.py``, shared with
the offline bag replay (tools/deploy/cpp_parity.py python --bag).
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

from dog_perception.node_config import node_parameters, preprocess_config  # noqa: E402
from dog_perception.online import OnlineInputs  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402

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
        self.inputs = OnlineInputs(params, self.pipe, self.log)
        self.world_frame = g("world_frame")
        self.publish_cloud = bool(g("publish_cloud"))
        if g("localization") == "inspvax":
            self.get_logger().info("INS: heading = %.0f deg - azimuth, lever arm %s m" % (
                self.inputs.ins_azimuth_offset, np.round(self.inputs.ins_lever_arm, 3).tolist()))

        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, g("imu_topic"), self.inputs.on_imu, 400)
        if g("localization") == "inspvax":
            from kygi665_msgs.msg import INSPVAX     # built in the robot's ROS workspace
            self.create_subscription(INSPVAX, g("inspvax_topic"), self.inputs.on_inspvax, 200)
        else:
            self.create_subscription(Odometry, g("odom_topic"), self.inputs.on_odom, 50)
        self.create_subscription(PointCloud2, g("lidar_topic"), self.on_cloud, best_effort)
        self.pub_tracks = self.create_publisher(MarkerArray, "~/tracks", 5)
        self.pub_preds = self.create_publisher(MarkerArray, "~/predictions", 5)
        self.pub_cloud = self.create_publisher(PointCloud2, "~/cloud", 2)
        self.get_logger().info("dog perception up: detector=%s" % kind)

    def log(self, level, text, throttle=0.0):
        logger = self.get_logger()
        fn = {"info": logger.info, "warn": logger.warn, "error": logger.error}[level]
        if throttle:
            fn(text, throttle_duration_sec=throttle)
        else:
            fn(text)

    def on_cloud(self, msg):
        out = self.inputs.on_cloud(msg)
        if out is None:
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
            txt.text = "%s %d %.1fm/s %s" % (s.name[:3], s.track_id, float(np.hypot(*s.velocity)), s.motion)
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
