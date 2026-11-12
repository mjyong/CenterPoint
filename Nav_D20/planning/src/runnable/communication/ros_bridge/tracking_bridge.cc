#include "src/runnable/communication/ros_bridge/tracking_bridge.h"

#include "nav2_interface/msg/nav_state.hpp"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/tracking/tracking.h"

namespace planning {
namespace communication {

using std::placeholders::_1;
using std::placeholders::_2;

TrackingBridge::TrackingBridge(
    const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
    const planning::config::CommunicationConf::RosBridgeConf &config)
    : node_(node), config_(config), planning_entry_(planning_entry) {
  if (config_.has_action_navigation()) {
    navigation_server_ = rclcpp_action::create_server<NavigationAction>(
        node_, config_.action_navigation().name(),
        std::bind(&TrackingBridge::handle_goal, this, _1, _2),
        std::bind(&TrackingBridge::handle_cancel, this, _1),
        std::bind(&TrackingBridge::handle_accepted, this, _1));
  }
}

void TrackingBridge::Publish() {
  if (!current_handle_)
    return;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  const bool mission_idle =
      mission_entry->GetCurrentMissionType() == MissionConf::Idle;
  if (current_handle_->is_active()) {
    auto feedback = std::make_shared<NavigationAction::Feedback>();
    auto mission_status =
        mission_idle ? MissionFSMBase::StateType::off
                     : mission_entry->GetCurrentMissionPtr()->mission_status();
    if (mission_status == MissionFSMBase::StateType::standby) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_PLAN_DOING;
      auto tracking = dynamic_cast<Tracking *>(
          mission_entry->GetMissionPtr(MissionConf::Tracking));
      auto *global_path = tracking->global_path_2d();
      for (size_t i = 0; i < global_path->size(); i++) {
        geometry_msgs::msg::Point pt;
        pt.x = global_path->at(i).position.x();
        pt.y = global_path->at(i).position.y();
        pt.z = 0.0;
        feedback->path.emplace_back(pt);
      }
      current_handle_->publish_feedback(feedback);
    } else if (mission_status == MissionFSMBase::StateType::active) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_NAVGATING;
      current_handle_->publish_feedback(feedback);
    } else if (mission_status == MissionFSMBase::StateType::standstill) {
      auto tracking = dynamic_cast<Tracking *>(
          mission_entry->GetMissionPtr(MissionConf::Tracking));
      if (tracking->nav_blocked()) {
        feedback->state = 3003;
      } else {
        feedback->state = 2014;
      }
      current_handle_->publish_feedback(feedback);
    } else if (mission_status == MissionFSMBase::StateType::off) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_PLAN_DONE;
      current_handle_->publish_feedback(feedback);
      auto result = std::make_shared<NavigationAction::Result>();
      result->message = "Navigation completed";
      current_handle_->succeed(result);
      current_handle_ = nullptr;
    } else if (mission_status == MissionFSMBase::StateType::fault) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_PLAN_PATH_FAILED;
      current_handle_->publish_feedback(feedback);
      auto result = std::make_shared<NavigationAction::Result>();
      result->message = "Navigation failed";
      current_handle_->abort(result);
      current_handle_ = nullptr;
    }
  } else if (mission_idle) {
    current_handle_ = nullptr;
  }
}

rclcpp_action::GoalResponse TrackingBridge::handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const NavigationAction::Goal> goal) {
  AINFO << "Navigation goal received, obstacle_mode: "
        << (int)goal->obstacle_mode << ", poses size: " << goal->poses.size()
        << ", max_speed: " << (int)goal->max_speed;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    AWARN << "Mission is running, stop current mission first.";
    mission_entry->GetCurrentMissionPtr()->stop();
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
TrackingBridge::handle_cancel(const std::shared_ptr<GoalHandleNavigation>) {
  AINFO << "Navigation cancel requested";
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

void TrackingBridge::handle_accepted(
    const std::shared_ptr<GoalHandleNavigation> goal_handle) {
  current_handle_ = goal_handle;
  auto goal = goal_handle->get_goal();
  auto mission_entry = planning_entry_->mission_entry_ptr();

  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle &&
      mission_entry->GetCurrentMissionPtr()->mission_status() !=
          MissionFSMBase::StateType::off) {
    AINFO << "Mission is not off, stop current mission.";
    mission_entry->GetCurrentMissionPtr()->stop();
  }

  mission_entry->SetCurrentMissionType(MissionConf::Tracking);
  auto tracking = dynamic_cast<Tracking *>(
      mission_entry->GetMissionPtr(MissionConf::Tracking));
  if (!goal->poses.empty()) {
    Pose3d target_pose;
    const auto &pose_st = goal->poses.front();
    target_pose.position.x() = pose_st.pose.position.x;
    target_pose.position.y() = pose_st.pose.position.y;
    target_pose.position.z() = pose_st.pose.position.z;
    target_pose.orientation.x() = pose_st.pose.orientation.x;
    target_pose.orientation.y() = pose_st.pose.orientation.y;
    target_pose.orientation.z() = pose_st.pose.orientation.z;
    target_pose.orientation.w() = pose_st.pose.orientation.w;
    bool constraint_theta = fabs(pose_st.pose.orientation.z) > 0.01 ||
                            fabs(pose_st.pose.orientation.y) > 0.01 ||
                            fabs(pose_st.pose.orientation.w) > 0.01 ||
                            fabs(pose_st.pose.orientation.x) > 0.01;
    tracking->setTargetPose(target_pose, constraint_theta);
    AINFO << "Navigation set target pose: " << target_pose.position.transpose()
          << ", constraint_theta: " << constraint_theta << ", max_speed: " << goal->max_speed;
  }
  if (goal->max_speed > 0.01) {
    tracking->setTargetSpeed(goal->max_speed);
  }

  mission_entry->GetCurrentMissionPtr()->start();
  AINFO << "Navigation started";
}
} // namespace communication
} // namespace planning
