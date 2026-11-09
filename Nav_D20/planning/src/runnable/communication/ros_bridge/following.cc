#include "src/runnable/mission/following/following.h"
#include "src/runnable/communication/ros_bridge/ros_bridge.h"
#include "src/runnable/entry/planning_entry.h"

namespace planning {
namespace communication {

static uint32_t following_object_id_ = 999;
static double following_object_ts_ = 0.0;

void RosBridge::FollowingInitPipe() {
  if (config_.has_sub_objects()) {
    sub_objects_ =
        node_->create_subscription<perception_interface::msg::Objects>(
            config_.sub_objects().name(), config_.sub_objects().queue_size(),
            std::bind(&RosBridge::objects_callback, this, _1));
    if (config_.sub_objects().has_timeout()) {
      timeout_states_[config_.sub_objects().name()] = {
          0.0, true, config_.sub_objects().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Following;
          }};
    }
  }

  if (config_.has_action_following()) {
    follow_server_ = rclcpp_action::create_server<FollowAction>(
        node_, config_.action_following().name(),
        std::bind(&RosBridge::follow_handle_goal, this, _1, _2),
        std::bind(&RosBridge::follow_handle_cancel, this, _1),
        std::bind(&RosBridge::follow_handle_accepted, this, _1));
  }
}

void RosBridge::FollowingFeedback() {
  if (!current_follow_handle_) {
    return;
  }
  following_object_ts_;
}

rclcpp_action::GoalResponse
RosBridge::follow_handle_goal(const rclcpp_action::GoalUUID &uuid,
                              std::shared_ptr<const FollowAction::Goal> goal) {
  (void)uuid;
  ADEBUG << "Received follow object goal request.";
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    AWARN << "Mission is running, stop current mission first.";
    mission_entry->GetCurrentMissionPtr()->stop();
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse RosBridge::follow_handle_cancel(
    const std::shared_ptr<GoalHandleFollow> goal_handle) {
  (void)goal_handle;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  ADEBUG << "Received follow object cancel request.";
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

void RosBridge::follow_handle_accepted(
    const std::shared_ptr<GoalHandleFollow> goal_handle) {
  current_follow_handle_ = goal_handle;
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

void RosBridge::objects_callback(
    const perception_interface::msg::Objects::SharedPtr msg) {
  (void)msg;
  auto it = timeout_states_.find(config_.sub_objects().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = msg->head.stamp.sec + msg->head.stamp.nanosec * 1e-9;
  }
  if (!current_follow_handle_)
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
      following->set_target_position(object.box_3d.x, object.box_3d.y);
      break;
    }
  }
}

} // namespace communication
} // namespace planning
