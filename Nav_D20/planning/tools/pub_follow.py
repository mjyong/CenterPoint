#!/usr/bin/env python3

import signal
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav2_interface.action import FollowObject


class FollowActionClient(Node):
    def __init__(self, action_name="/nav/follow"):
        super().__init__("follow_action_client")
        self._action_client = ActionClient(self, FollowObject, action_name)
        self._goal_handle = None
        self.get_logger().info(
            f"FollowActionClient init, action: {action_name}")

    def send_goal(self, object_id=0):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Can not find FollowObject action server.")
            return

        goal_msg = FollowObject.Goal()
        goal_msg.box.id = object_id

        self.get_logger().info(f"Sending goal: object_id={object_id}")

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info("Canceling goal...")
            future = self._goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().info("No active goal to cancel.")
            if rclpy.ok():
                rclpy.shutdown()

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if (cancel_response is not None
                and len(cancel_response.goals_canceling) > 0):
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
            f"Feedback: state={feedback.state}, id={feedback.id}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.message}")
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    action_name = sys.argv[1] if len(sys.argv) > 1 else "/nav/follow"
    object_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    action_client = FollowActionClient(action_name)
    action_client.send_goal(object_id=object_id)

    def sigint_handler(sig, frame):
        action_client.get_logger().info("Ctrl+C received, canceling goal...")
        action_client.cancel_goal()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while rclpy.ok():
            rclpy.spin_once(action_client, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
