#include "src/runnable/communication/ros_bridge/skill_bridge.h"

#include "src/common/util/debug_datas.h"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/skill/skill.h"

namespace planning {
namespace communication {
using planning::common::util::DebugDatas;

using std::placeholders::_1;
using std::placeholders::_2;

SkillBridge::SkillBridge(
    const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
    const planning::config::CommunicationConf::RosBridgeConf &config,
    TimeoutStates &timeout_states)
    : node_(node), config_(config), planning_entry_(planning_entry),
      timeout_states_(timeout_states) {
  if (config_.has_action_skill()) {
    skill_server_ = rclcpp_action::create_server<SkillAction>(
        node_, config_.action_skill().name(),
        std::bind(&SkillBridge::handle_goal, this, _1, _2),
        std::bind(&SkillBridge::handle_cancel, this, _1),
        std::bind(&SkillBridge::handle_accepted, this, _1));
  }
}

void SkillBridge::Publish() {
  if (!current_handle_) {
    return;
  }
  auto mission_entry = planning_entry_->mission_entry_ptr();
  const bool mission_idle =
      mission_entry->GetCurrentMissionType() == MissionConf::Idle;
  if (current_handle_->is_active()) {
    auto feedback = std::make_shared<SkillAction::Feedback>();
    auto mission_status =
        mission_idle ? MissionFSMBase::StateType::off
                     : mission_entry->GetCurrentMissionPtr()->mission_status();
    if (mission_status == MissionFSMBase::StateType::active) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_DOING;
      current_handle_->publish_feedback(feedback);
    } else if (mission_status == MissionFSMBase::StateType::off) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_SUCCESS;
      current_handle_->publish_feedback(feedback);
      auto result = std::make_shared<SkillAction::Result>();
      result->message = "Skill completed";
      current_handle_->succeed(result);
      current_handle_ = nullptr;
    } else if (mission_status == MissionFSMBase::StateType::fault) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_FAILED;
      current_handle_->publish_feedback(feedback);
      auto result = std::make_shared<SkillAction::Result>();
      result->message = "Skill failed";
      current_handle_->abort(result);
      current_handle_ = nullptr;
    }
  } else if (mission_idle) {
    current_handle_ = nullptr;
  }
}

rclcpp_action::GoalResponse
SkillBridge::handle_goal(const rclcpp_action::GoalUUID &uuid,
                         std::shared_ptr<const SkillAction::Goal> goal) {
  (void)uuid;
  AINFO << "Received skill goal request.";
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    AWARN << "Mission is running, stop current mission first.";
    mission_entry->GetCurrentMissionPtr()->stop();
    return rclcpp_action::GoalResponse::REJECT;
  }
  const auto skill_id = goal->skill_id;
  const size_t required_parameters =
      skill_id == nav2_interface::action::Skill::Goal::SKILL_ID_ROTATE     ? 1
      : skill_id == nav2_interface::action::Skill::Goal::SKILL_ID_MOVE_DIS ? 2
                                                                           : 0;
  if (required_parameters == 0 || goal->para.size() < required_parameters ||
      !std::all_of(goal->para.begin(), goal->para.end(),
                   [](float value) { return std::isfinite(value); })) {
    AERROR << "Invalid skill goal, id: " << static_cast<int>(skill_id);
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
SkillBridge::handle_cancel(const std::shared_ptr<GoalHandleSkill> goal_handle) {
  (void)goal_handle;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  AINFO << "Received skill cancel request.";
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

void SkillBridge::handle_accepted(
    const std::shared_ptr<GoalHandleSkill> goal_handle) {
  current_handle_ = goal_handle;
  auto goal = goal_handle->get_goal();
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle &&
      mission_entry->GetCurrentMissionPtr()->mission_status() !=
          MissionFSMBase::StateType::off) {
    AINFO << "Mission is not off, stop current mission.";
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  mission_entry->SetCurrentMissionType(MissionConf::Skill);
  auto skill =
      dynamic_cast<Skill *>(mission_entry->GetMissionPtr(MissionConf::Skill));
  switch (goal->skill_id) {
  case SkillAction::Goal::SKILL_ID_ROTATE:
    AINFO << "Skill set Rotating: " << goal->para[0];
    skill->set_rotating_skill(goal->para[0]);
    break;
  case SkillAction::Goal::SKILL_ID_MOVE_DIS:
    AINFO << "Skill set Moving: " << goal->para[0] << " " << goal->para[1];
    skill->set_moving_skill(goal->para[0], goal->para[1]);
    break;
  default:
    AERROR << "Unknown skill type: " << goal->skill_id;
    break;
  }

  mission_entry->GetCurrentMissionPtr()->start();
  AINFO << "Skill started";
}

} // namespace communication
} // namespace planning
