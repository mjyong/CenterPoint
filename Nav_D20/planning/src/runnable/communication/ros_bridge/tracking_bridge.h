#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "communication_conf.pb.h"
#include "nav2_interface/action/navigation.hpp"

namespace planning {
class PlanningEntry;

namespace communication {

/**
 * @brief Navigation action server bridge for the Tracking mission.
 *
 * Accepts nav2_interface Navigation goals, translates them into Tracking
 * mission starts and reports the mission state back as action feedback.
 * All servers are created in the constructor; Publish() is called every
 * communication cycle by RosBridge.
 */
class TrackingBridge {
public:
  TrackingBridge(
      const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
      const planning::config::CommunicationConf::RosBridgeConf &config);

  // Per-cycle navigation action feedback (mission state -> feedback state).
  void Publish();

private:
  using NavigationAction = nav2_interface::action::Navigation;
  using GoalHandleNavigation =
      rclcpp_action::ServerGoalHandle<NavigationAction>;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const NavigationAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleNavigation> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleNavigation> goal_handle);

  rclcpp::Node::SharedPtr node_;
  planning::config::CommunicationConf::RosBridgeConf config_;
  PlanningEntry *planning_entry_{nullptr};
  rclcpp_action::Server<NavigationAction>::SharedPtr navigation_server_;
  std::shared_ptr<GoalHandleNavigation> current_handle_;
};

} // namespace communication
} // namespace planning
