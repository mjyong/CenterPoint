#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "communication_conf.pb.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mc_interface/msg/state.hpp"
#include "nav2_interface/action/back_charge.hpp"
#include "rte_interface/msg/battery_states.hpp"
#include "src/runnable/communication/ros_bridge/timeout_state.h"

namespace planning {
class PlanningEntry;

namespace communication {

/**
 * @brief BackCharge action server bridge for the Recharging mission.
 *
 * Accepts nav2_interface BackCharge goals, translates them into Recharging
 * mission starts, feeds charging goal / battery / motion state subscriptions
 * into the mission and reports the mission state back as action feedback.
 * All servers and subscriptions are created in the constructor; Publish() is
 * called every communication cycle by RosBridge. Timeout entries are
 * registered into the TimeoutStates map owned by RosBridge.
 */
class RechargingBridge {
public:
  RechargingBridge(
      const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
      const planning::config::CommunicationConf::RosBridgeConf &config,
      TimeoutStates &timeout_states);

  // Per-cycle BackCharge action feedback (mission state -> 6xxx state code).
  void Publish();

private:
  using BackChargeAction = nav2_interface::action::BackCharge;
  using GoalHandleBackCharge =
      rclcpp_action::ServerGoalHandle<BackChargeAction>;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const BackChargeAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleBackCharge> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleBackCharge> goal_handle);

  void charging_goal_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void
  battery_states_callback(const rte_interface::msg::BatteryStates::SharedPtr msg);
  void motion_state_callback(const mc_interface::msg::State::SharedPtr msg);

  rclcpp::Node::SharedPtr node_;
  planning::config::CommunicationConf::RosBridgeConf config_;
  PlanningEntry *planning_entry_{nullptr};
  TimeoutStates &timeout_states_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      sub_charging_pose_;
  rclcpp::Subscription<rte_interface::msg::BatteryStates>::SharedPtr
      sub_battery_states_;
  rclcpp::Subscription<mc_interface::msg::State>::SharedPtr sub_motion_state_;

  rclcpp_action::Server<BackChargeAction>::SharedPtr back_charge_server_;
  std::shared_ptr<GoalHandleBackCharge> current_handle_;
  uint32_t last_feedback_state_{0xFFFFFFFFu};
  uint32_t final_state_{0};
};

} // namespace communication
} // namespace planning
