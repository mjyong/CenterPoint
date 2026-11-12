#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "communication_conf.pb.h"
#include "nav2_interface/action/skill.hpp"
#include "src/runnable/communication/ros_bridge/timeout_state.h"

namespace planning {
class PlanningEntry;

namespace communication {

class SkillBridge {
public:
  SkillBridge(
      const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
      const planning::config::CommunicationConf::RosBridgeConf &config,
      TimeoutStates &timeout_states);

  // Per-cycle Skill action feedback.
  void Publish();

private:
  using SkillAction = nav2_interface::action::Skill;
  using GoalHandleSkill = rclcpp_action::ServerGoalHandle<SkillAction>;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const SkillAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleSkill> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleSkill> goal_handle);

  rclcpp::Node::SharedPtr node_;
  planning::config::CommunicationConf::RosBridgeConf config_;
  PlanningEntry *planning_entry_{nullptr};
  TimeoutStates &timeout_states_;

  rclcpp_action::Server<SkillAction>::SharedPtr skill_server_;
  std::shared_ptr<GoalHandleSkill> current_handle_;
};

} // namespace communication
} // namespace planning
