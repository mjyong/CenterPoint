#!/usr/bin/env python3

import signal
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav2_interface.action import BackCharge
from geometry_msgs.msg import Pose


class BackChargeActionClient(Node):
    def __init__(self):
        super().__init__("back_charge_action_client")
        self._action_client = ActionClient(
            self, BackCharge, "/nav/back_charge")
        self._goal_handle = None
        self.get_logger().info("BackChargeActionClient init.")

    def send_goal(self, task_id=0, pos_x=0.0, pos_y=0.0, pos_z=0.0,
                  ori_x=0.0, ori_y=0.0, ori_z=0.0, ori_w=1.0):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Can not find BackCharge action server.")
            return

        goal_msg = BackCharge.Goal()
        goal_msg.task_id = task_id
        goal_msg.pose = Pose()
        goal_msg.pose.position.x = pos_x
        goal_msg.pose.position.y = pos_y
        goal_msg.pose.position.z = pos_z
        goal_msg.pose.orientation.x = ori_x
        goal_msg.pose.orientation.y = ori_y
        goal_msg.pose.orientation.z = ori_z
        goal_msg.pose.orientation.w = ori_w

        self.get_logger().info(
            f"Sending goal: task_id={task_id}, "
            f"pos=({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}), "
            f"ori=({ori_x:.2f}, {ori_y:.2f}, {ori_z:.2f}, {ori_w:.2f})")

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
        self.get_logger().info(f"Feedback: state={feedback.state}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.message}")
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    action_client = BackChargeActionClient()

    # Default: charging station behind the robot (negative x, yaw=pi)
    task_id = 0
    pos_x, pos_y, pos_z = -1.5, 0.0, 0.0
    ori_x, ori_y, ori_z, ori_w = 0.0, 0.0, 0.0, 1.0

    if len(sys.argv) > 1:
        try:
            task_id = int(sys.argv[1])
            pos_x = float(sys.argv[2]) if len(sys.argv) > 2 else pos_x
            pos_y = float(sys.argv[3]) if len(sys.argv) > 3 else pos_y
            pos_z = float(sys.argv[4]) if len(sys.argv) > 4 else pos_z
            ori_x = float(sys.argv[5]) if len(sys.argv) > 5 else ori_x
            ori_y = float(sys.argv[6]) if len(sys.argv) > 6 else ori_y
            ori_z = float(sys.argv[7]) if len(sys.argv) > 7 else ori_z
            ori_w = float(sys.argv[8]) if len(sys.argv) > 8 else ori_w
        except ValueError:
            action_client.get_logger().error(
                "Usage: python3 pub_back_charge.py "
                "[task_id pos_x pos_y pos_z ori_x ori_y ori_z ori_w]")
            rclpy.shutdown()
            return

    action_client.send_goal(
        task_id=task_id, pos_x=pos_x, pos_y=pos_y, pos_z=pos_z,
        ori_x=ori_x, ori_y=ori_y, ori_z=ori_z, ori_w=ori_w)

    def sigint_handler(sig, frame):
        action_client.get_logger().info("Ctrl+C received, canceling goal...")
        action_client.cancel_goal()

    signal.signal(signal.SIGINT, sigint_handler)
    # 不能用 rclpy.spin()：它会无限期阻塞在 C 层 rcl_wait，Python 信号处理器
    # 无法在字节码间隙执行，Ctrl+C 无反应。改用带超时的 spin_once 循环。
    try:
        while rclpy.ok():
            rclpy.spin_once(action_client, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
