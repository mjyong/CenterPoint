# /bin/python3

import signal
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_interface.action import Navigation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


class NavigationActionClient(Node):
    def __init__(self):
        super().__init__("navigation_action_client")
        self._action_client = ActionClient(self, Navigation, "/nav/navigation")
        self._goal_handle = None
        self.get_logger().info("NavigationActionClient init.")

    def send_goal(self):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Can Not Find Action Server.")
            return

        goal_msg = Navigation.Goal()
        goal_msg.obstacle_mode = Navigation.Goal.OBSTACLE_MODE_AVOID
        goal_msg.map_uuid = ""
        goal_msg.max_speed = 10

        pose = PoseStamped()
        pose.header = Header(frame_id="map")
        pose.pose.position.x = -615.07
        pose.pose.position.y = 729.70
        pose.pose.position.z = 0.0
        # pose.pose.orientation.x = 0.02299641299891112
        # pose.pose.orientation.y = 0.035259116957613225
        # pose.pose.orientation.z = 0.9878281948614683
        # pose.pose.orientation.w = 0.14886060992685396
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0        
        goal_msg.poses.append(pose)

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.get_logger().info("NavigationActionClient send goal.")

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
        if (cancel_response is not None
                and len(cancel_response.goals_canceling) > 0):
            self.get_logger().info("Goal canceled successfully.")
        else:
            self.get_logger().warn("Failed to cancel goal.")
        rclpy.shutdown()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Server Rejected Goal.")
            rclpy.shutdown()
            return
        self.get_logger().info("Server Accept Goal.")
        self._goal_handle = goal_handle

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"Feedback: state={feedback.state}, "
            f"path size={len(feedback.path)}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.message}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    action_client = NavigationActionClient()
    action_client.send_goal()

    def sigint_handler(sig, frame):
        action_client.get_logger().info("Ctrl+C received, canceling goal...")
        action_client.cancel_goal()

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
