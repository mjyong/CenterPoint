#include "src/runnable/communication/ros_bridge/following_bridge.h"

#include "src/common/util/debug_datas.h"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/following/following.h"

namespace planning {
namespace communication {
using planning::common::util::DebugDatas;

using std::placeholders::_1;
using std::placeholders::_2;

FollowingBridge::FollowingBridge(
    const rclcpp::Node::SharedPtr &node, PlanningEntry *planning_entry,
    const planning::config::CommunicationConf::RosBridgeConf &config,
    TimeoutStates &timeout_states)
    : node_(node), config_(config), planning_entry_(planning_entry),
      timeout_states_(timeout_states) {
  if (config_.has_sub_objects()) {
    sub_objects_ =
        node_->create_subscription<perception_interface::msg::Objects>(
            config_.sub_objects().name(), config_.sub_objects().queue_size(),
            std::bind(&FollowingBridge::objects_callback, this, _1));
    if (config_.sub_objects().has_timeout()) {
      timeout_states_[config_.sub_objects().name()] = {
          0.0, true, config_.sub_objects().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Following;
          }};
    }
  }

  if (config_.has_sub_follow_target()) {
    sub_follow_target_ =
        node_->create_subscription<nav_interface::msg::FollowTarget>(
            config_.sub_follow_target().name(),
            config_.sub_follow_target().queue_size(),
            std::bind(&FollowingBridge::follow_target_callback, this, _1));
    if (config_.sub_follow_target().has_timeout()) {
      timeout_states_[config_.sub_follow_target().name()] = {
          0.0, true, config_.sub_follow_target().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Following;
          }};
    }
  }

  if (config_.has_action_following()) {
    follow_server_ = rclcpp_action::create_server<FollowAction>(
        node_, config_.action_following().name(),
        std::bind(&FollowingBridge::handle_goal, this, _1, _2),
        std::bind(&FollowingBridge::handle_cancel, this, _1),
        std::bind(&FollowingBridge::handle_accepted, this, _1));
  }

  sub_clicked_point_ =
      node_->create_subscription<geometry_msgs::msg::PointStamped>(
          "/clicked_point", 1,
          std::bind(&FollowingBridge::clicked_point_callback, this, _1));
}

void FollowingBridge::Publish() {
  if (!current_handle_) {
    return;
  }
}

rclcpp_action::GoalResponse FollowingBridge::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const FollowAction::Goal> goal) {
  (void)uuid;
  AINFO << "Received follow object goal request.";
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    AWARN << "Mission is running, stop current mission first.";
    mission_entry->GetCurrentMissionPtr()->stop();
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse FollowingBridge::handle_cancel(
    const std::shared_ptr<GoalHandleFollow> goal_handle) {
  (void)goal_handle;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  AINFO << "Received follow object cancel request.";
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

void FollowingBridge::handle_accepted(
    const std::shared_ptr<GoalHandleFollow> goal_handle) {
  current_handle_ = goal_handle;
  auto goal = goal_handle->get_goal();
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle &&
      mission_entry->GetCurrentMissionPtr()->mission_status() !=
          MissionFSMBase::StateType::off) {
    AINFO << "Mission is not off, stop current mission.";
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  mission_entry->SetCurrentMissionType(MissionConf::Following);
  auto following = dynamic_cast<Following *>(
      mission_entry->GetMissionPtr(MissionConf::Following));
  following_object_id_ = goal->box.id;
  mission_entry->GetCurrentMissionPtr()->start();
  AINFO << "Following started";
}

void FollowingBridge::objects_callback(
    const perception_interface::msg::Objects::SharedPtr msg) {
  (void)msg;
  auto it = timeout_states_.find(config_.sub_objects().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = msg->head.stamp.sec + msg->head.stamp.nanosec * 1e-9;
  }
  if (!current_handle_)
    return;
  for (const auto &object : msg->objs) {
    if (object.box_2d.id == following_object_id_) {
      following_object_ts_ =
          msg->head.stamp.sec + msg->head.stamp.nanosec * 1e-9;
      auto r11_head_bridge =
          planning_entry_->communication_entry_ptr()->r11_head_bridge();
      if (r11_head_bridge) {
        r11_head_bridge->set_target_img_position(
            object.box_2d.center.x, object.box_2d.center.y,
            object.box_2d.size_x, object.box_2d.size_y, following_object_ts_);
      }
      auto following = dynamic_cast<Following *>(
          planning_entry_->mission_entry_ptr()->GetMissionPtr(
              MissionConf::Following));
      if (following) {
        following->set_target_position(object.box_3d.x, object.box_3d.y);
      }
      break;
    }
  }
}

void FollowingBridge::follow_target_callback(
    const nav_interface::msg::FollowTarget::SharedPtr msg) {
  (void)msg;
  auto it = timeout_states_.find(config_.sub_follow_target().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = msg->time;
  }
  if (!current_handle_)
    return;
  auto following = dynamic_cast<Following *>(
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Following));
  if (!following) {
    return;
  }
  // xabs/yabs 为全局系目标位置
  following->set_target_position(msg->xabs, msg->yabs);
  following->set_target_velocity(msg->vx, msg->vy);
  DebugDatas::Instance()->emplace("/follow/input_target/x", msg->xabs);
  DebugDatas::Instance()->emplace("/follow/input_target/y", msg->yabs);
  DebugDatas::Instance()->emplace("/follow/input_target/source", msg->searching_mode);
  DebugDatas::Instance()->emplace("/follow/input_target/vx", msg->vx);
  DebugDatas::Instance()->emplace("/follow/input_target/vy", msg->vy);
}

void FollowingBridge::clicked_point_callback(
    const geometry_msgs::msg::PointStamped::SharedPtr msg) {
}

} // namespace communication
} // namespace planning
