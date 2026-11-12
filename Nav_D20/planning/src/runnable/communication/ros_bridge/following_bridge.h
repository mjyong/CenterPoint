#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "communication_conf.pb.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav2_interface/action/follow_object.hpp"
#include "nav_interface/msg/follow_target.hpp"
#include "perception_interface/msg/objects.hpp"
#include "src/runnable/communication/ros_bridge/timeout_state.h"

namespace planning {
class PlanningEntry;

namespace communication {

/**
 * @brief FollowObject action server bridge for the Following mission.
 *
 * Accepts nav2_interface FollowObject goals, translates them into Following
 * mission starts and feeds object / follow target observations into the
 * mission. All servers and subscriptions are created in the constructor;
 * Publish() is called every communication cycle by RosBridge. Timeout
 * entries are registered into the TimeoutStates map owned by RosBridge.
 */
class FollowingBridge {
public:
  FollowingBridge(
      const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
      const planning::config::CommunicationConf::RosBridgeConf &config,
      TimeoutStates &timeout_states);

  // Per-cycle FollowObject action feedback.
  void Publish();

private:
  using FollowAction = nav2_interface::action::FollowObject;
  using GoalHandleFollow = rclcpp_action::ServerGoalHandle<FollowAction>;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const FollowAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleFollow> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleFollow> goal_handle);

  void objects_callback(const perception_interface::msg::Objects::SharedPtr msg);
  void
  follow_target_callback(const nav_interface::msg::FollowTarget::SharedPtr msg);

  void
  clicked_point_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg);

  rclcpp::Node::SharedPtr node_;
  planning::config::CommunicationConf::RosBridgeConf config_;
  PlanningEntry *planning_entry_{nullptr};
  TimeoutStates &timeout_states_;

  rclcpp_action::Server<FollowAction>::SharedPtr follow_server_;
  std::shared_ptr<GoalHandleFollow> current_handle_;
  rclcpp::Subscription<perception_interface::msg::Objects>::SharedPtr
      sub_objects_;
  rclcpp::Subscription<nav_interface::msg::FollowTarget>::SharedPtr
      sub_follow_target_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr
      sub_clicked_point_;

  uint32_t following_object_id_{999};
  double following_object_ts_{0.0};
};

} // namespace communication
} // namespace planning
