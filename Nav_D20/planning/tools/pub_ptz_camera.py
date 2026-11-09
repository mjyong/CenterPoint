#!/usr/bin/env python3

import signal
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_interface.action import PTZCamera
from geometry_msgs.msg import Vector3


class PtzCameraActionClient(Node):
    def __init__(self):
        super().__init__("ptz_camera_action_client")
        self._action_client = ActionClient(self, PTZCamera, "/nav/ptz_camera")
        self._goal_handle = None
        self.get_logger().info("PtzCameraActionClient init.")

    def send_goal(
        self,
        sensor_id=0,
        target_x=1.0,
        target_y=0.0,
        target_z=-1.0,
        scale_x=2.0,
        scale_y=2.0,
        scale_z=2.0,
        img_scale=0.5,
    ):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Can not find PTZ Camera action server.")
            return

        goal_msg = PTZCamera.Goal()
        goal_msg.sensor_id = sensor_id
        goal_msg.target_point = Vector3(x=target_x, y=target_y, z=target_z)
        goal_msg.target_scale = Vector3(x=scale_x, y=scale_y, z=scale_z)
        goal_msg.img_scale = img_scale

        self.get_logger().info(
            f"Sending goal: sensor_id={sensor_id}, "
            f"target=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), "
            f"scale=({scale_x:.2f}, {scale_y:.2f}, {scale_z:.2f}), "
            f"img_scale={img_scale:.2f}"
        )

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info("Canceling goal...")
            future = self._goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().info("No active goal to cancel.")
            rclpy.shutdown()

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if cancel_response is not None and len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("Goal canceled successfully.")
        else:
            self.get_logger().warn("Failed to cancel goal.")
        rclpy.shutdown()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Server rejected goal.")
            rclpy.shutdown()
            return
        self.get_logger().info("Server accepted goal.")
        self._goal_handle = goal_handle

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Feedback: state={feedback.state}, "
            f"ptz=({feedback.ptz.x:.3f}, {feedback.ptz.y:.3f}, "
            f"{feedback.ptz.z:.3f})"
        )

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.message}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    action_client = PtzCameraActionClient()

    # Parse command-line arguments
    sensor_id = 0
    target_x, target_y, target_z = 3.0, -1.0, 0.5
    scale_x, scale_y, scale_z = 1.0, 1.0, 1.0
    img_scale = 1.0

    if len(sys.argv) > 1:
        try:
            sensor_id = int(sys.argv[1])
            target_x = float(sys.argv[2]) if len(sys.argv) > 2 else target_x
            target_y = float(sys.argv[3]) if len(sys.argv) > 3 else target_y
            target_z = float(sys.argv[4]) if len(sys.argv) > 4 else target_z
            scale_x = float(sys.argv[5]) if len(sys.argv) > 5 else scale_x
            scale_y = float(sys.argv[6]) if len(sys.argv) > 6 else scale_y
            scale_z = float(sys.argv[7]) if len(sys.argv) > 7 else scale_z
            img_scale = float(sys.argv[8]) if len(sys.argv) > 8 else img_scale
        except ValueError:
            action_client.get_logger().error(
                "Usage: python3 pub_ptz_camera.py [sensor_id target_x target_y "
                "target_z scale_x scale_y scale_z img_scale]"
            )
            rclpy.shutdown()
            return

    action_client.send_goal(
        sensor_id=sensor_id,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        scale_x=scale_x,
        scale_y=scale_y,
        scale_z=scale_z,
        img_scale=img_scale,
    )

    def sigint_handler(sig, frame):
        action_client.get_logger().info("Ctrl+C received, canceling goal...")
        action_client.cancel_goal()

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
