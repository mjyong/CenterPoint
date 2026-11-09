#include "src/runnable/mission/recharging/recharging.h"
#include "src/runnable/communication/ros_bridge/ros_bridge.h"
#include "src/runnable/entry/planning_entry.h"
namespace planning {
namespace communication {
using planning::common::util::normalize_angle;

void RosBridge::RechargingInitPipe() {
  if (config_.has_sub_charging_goal()) {
    sub_charging_pose_ =
        node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            config_.sub_charging_goal().name(),
            config_.sub_charging_goal().queue_size(),
            std::bind(&RosBridge::charging_goal_pose, this, _1));
    if (config_.sub_charging_goal().has_timeout()) {
      timeout_states_[config_.sub_charging_goal().name()] = {
          0.0, true, config_.sub_charging_goal().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            if (entry->GetCurrentMissionType() != MissionConf::Recharging)
              return false;
            auto recharging = dynamic_cast<Recharging *>(
                entry->GetMissionPtr(MissionConf::Recharging));
            return !recharging->IsBackStake();
          }};
    }
  }

  if (config_.has_action_backcharge()) {
    back_charge_server_ = rclcpp_action::create_server<BackChargeAction>(
        node_, config_.action_backcharge().name(),
        std::bind(&RosBridge::back_charge_handle_goal, this, _1, _2),
        std::bind(&RosBridge::back_charge_handle_cancel, this, _1),
        std::bind(&RosBridge::back_charge_handle_accepted, this, _1));
  }

  if (config_.has_sub_battery_states()) {
    sub_battery_states_ =
        node_->create_subscription<rte_interface::msg::BatteryStates>(
            config_.sub_battery_states().name(),
            config_.sub_battery_states().queue_size(),
            std::bind(&RosBridge::battery_states_callback, this, _1));
    if (config_.sub_battery_states().has_timeout()) {
      timeout_states_[config_.sub_battery_states().name()] = {
          0.0, true, config_.sub_battery_states().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            if (entry->GetCurrentMissionType() != MissionConf::Recharging)
              return false;
            auto recharging = dynamic_cast<Recharging *>(
                entry->GetMissionPtr(MissionConf::Recharging));
            return !recharging->IsBackStake();
          }};
    }
  }

  if (config_.has_sub_motion_state()) {
    sub_motion_state_ = node_->create_subscription<mc_interface::msg::State>(
        config_.sub_motion_state().name(),
        config_.sub_motion_state().queue_size(),
        std::bind(&RosBridge::motion_state_callback, this, _1));
    if (config_.sub_motion_state().has_timeout()) {
      timeout_states_[config_.sub_motion_state().name()] = {
          0.0, true, config_.sub_motion_state().timeout(), [this]() {
            auto entry = planning_entry_->mission_entry_ptr();
            return entry->GetCurrentMissionType() == MissionConf::Recharging;
          }};
    }
  }
}

void RosBridge::RechargingFeedback() {
  if (!current_back_charge_handle_ ||
      !current_back_charge_handle_->is_active()) {
    return;
  }
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Recharging) {
    return;
  }
  auto recharging = dynamic_cast<Recharging *>(
      mission_entry->GetMissionPtr(MissionConf::Recharging));
  if (recharging == nullptr) {
    return;
  }

  const auto outer = recharging->mission_status();
  const auto inner = recharging->recharging_status();
  const bool back_stake = recharging->IsBackStake();

  // 双层状态 -> feedback 状态码映射（BackCharge 专用 6xxx 桩业务段：
  // 6001桩未找到 6002回桩失败 | 6010搜索 6011对准 6012进桩 6013坐下
  // 6014充电成功 | 6020起立 6021退桩 6022退桩完成 | 6030暂停 6031重试）
  uint32_t state = 0;
  switch (outer) {
  case MissionFSMBase::StateType::passive:
  case MissionFSMBase::StateType::standby:
  case MissionFSMBase::StateType::active:
    switch (inner) {
    case RechargingFSMBase::StateType::searching:
      state = 6010; // 搜索充电桩中
      break;
    case RechargingFSMBase::StateType::aligning:
      state = 6011; // 对准充电桩中
      break;
    case RechargingFSMBase::StateType::inserting:
      state = 6012; // 进桩中
      break;
    case RechargingFSMBase::StateType::sitting:
      state = 6013; // 坐下/充电确认中
      break;
    case RechargingFSMBase::StateType::standing_up:
      // 退桩流程：起立执行；充电失败重试：起立后退回预备点重新对准
      state = back_stake ? 6020 : 6031; // 起立中 / 对准重试中
      break;
    case RechargingFSMBase::StateType::backing_off:
      state = back_stake ? 6021 : 6031; // 退桩中 / 对准重试中
      break;
    case RechargingFSMBase::StateType::done:
      state = back_stake ? 6022 : 6014; // 退桩完成 / 充电成功
      // 记录终态，任务自动到 off 后据此 succeed
      back_charge_final_state_ = state;
      break;
    case RechargingFSMBase::StateType::idle:
    default:
      // 启动空拍瞬态（on_entry_s_passive 同步 Start，正常仅存在一个周期）
      state = back_stake ? 6020 : 6010; // 起立中 / 搜索充电桩中
      break;
    }
    break;
  case MissionFSMBase::StateType::standstill:
    state = 6030; // 任务暂停(standstill)
    break;
  case MissionFSMBase::StateType::fault:
    // fault 后内层已复位，暂不细分故障原因（如需区分 6001 搜索超时，
    // 需在 Recharging 补充故障原因记录）
    state =
        nav2_interface::msg::NavState::NAV_STA_STAKE_TO_FAILE; // 6002 回桩失败
    break;
  case MissionFSMBase::StateType::off:
  default: {
    // 任务结束，终止 goal：正常完成 -> succeed；被取消 -> canceled
    auto result = std::make_shared<BackChargeAction::Result>();
    if (back_charge_final_state_ != 0) {
      result->message = (back_charge_final_state_ == 6022) ? "back stake done"
                                                           : "charge done";
      current_back_charge_handle_->succeed(result);
    } else {
      result->message = "cancelled";
      current_back_charge_handle_->canceled(result);
    }
    AINFO << "[BackCharge] goal terminated: " << result->message;
    current_back_charge_handle_.reset();
    back_charge_final_state_ = 0;
    return;
  }
  }

  // 仅状态变化时发布，避免按周期刷屏
  if (state == last_back_charge_feedback_state_) {
    return;
  }
  last_back_charge_feedback_state_ = state;

  auto feedback = std::make_shared<BackChargeAction::Feedback>();
  feedback->state = state;
  current_back_charge_handle_->publish_feedback(feedback);
  AINFO << "[BackCharge] feedback state: " << state;

  // fault：发布失败 feedback 后 abort goal（mission 停在 fault，
  // 等待新任务下发或外部取消）
  if (outer == MissionFSMBase::StateType::fault) {
    auto result = std::make_shared<BackChargeAction::Result>();
    result->message = "recharging fault (6002)";
    current_back_charge_handle_->abort(result);
    AINFO << "[BackCharge] goal aborted: recharging fault";
    current_back_charge_handle_.reset();
    back_charge_final_state_ = 0;
  }
}

rclcpp_action::GoalResponse RosBridge::back_charge_handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const BackChargeAction::Goal> goal) {
  (void)uuid;
  int task_id = goal->task_id;
  AINFO << "Received task id: " << task_id;
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    AWARN << "Mission is running, stop current mission first.";
    mission_entry->GetCurrentMissionPtr()->stop();
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse RosBridge::back_charge_handle_cancel(
    const std::shared_ptr<GoalHandleBackCharge> goal_handle) {
  (void)goal_handle;
  AINFO << "Received cancel request.";
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle) {
    mission_entry->GetCurrentMissionPtr()->stop();
  }
  return rclcpp_action::CancelResponse::ACCEPT;
}

void RosBridge::back_charge_handle_accepted(
    const std::shared_ptr<GoalHandleBackCharge> goal_handle) {
  // 新 goal 打断旧任务：终止旧 handle，避免其永远挂起
  if (current_back_charge_handle_ &&
      current_back_charge_handle_ != goal_handle &&
      current_back_charge_handle_->is_active()) {
    auto old_result = std::make_shared<BackChargeAction::Result>();
    old_result->message = "aborted by new goal";
    current_back_charge_handle_->abort(old_result);
  }
  current_back_charge_handle_ = goal_handle;
  // 新任务：重置 feedback 状态跟踪，确保首帧状态必发
  last_back_charge_feedback_state_ = 0xFFFFFFFFu;
  back_charge_final_state_ = 0;
  auto goal = goal_handle->get_goal();
  auto mission_entry = planning_entry_->mission_entry_ptr();

  if (mission_entry->GetCurrentMissionType() != MissionConf::Idle &&
      mission_entry->GetCurrentMissionPtr()->mission_status() !=
          MissionFSMBase::StateType::off) {
    AINFO << "Mission is not off, stop current mission.";
    mission_entry->GetCurrentMissionPtr()->stop();
  }

  auto recharging = dynamic_cast<Recharging *>(
      mission_entry->GetMissionPtr(MissionConf::Recharging));
  recharging->SetBackStake(goal->task_id == 1);

  mission_entry->SetCurrentMissionType(MissionConf::Recharging);
  mission_entry->GetCurrentMissionPtr()->start();
  AINFO << "Recharging started, task_id: " << goal->task_id;
}

void RosBridge::charging_goal_pose(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  auto mission_entry = planning_entry_->mission_entry_ptr();
  if (mission_entry->GetCurrentMissionType() != MissionConf::Recharging) {
    return;
  }

  Pose3dStamped charging_pose;
  charging_pose.pose.position.x() = msg->pose.position.x;
  charging_pose.pose.position.y() = msg->pose.position.y;
  charging_pose.pose.position.z() = 0.0;
  charging_pose.pose.orientation.w() = msg->pose.orientation.w;
  charging_pose.pose.orientation.x() = msg->pose.orientation.x;
  charging_pose.pose.orientation.y() = msg->pose.orientation.y;
  charging_pose.pose.orientation.z() = msg->pose.orientation.z;

  const double yaw = normalize_angle(charging_pose.pose.euler().z() + M_PI);
  charging_pose.pose.orientation =
      Quaternion(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  charging_pose.timestamp =
      msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  auto recharging = dynamic_cast<Recharging *>(
      mission_entry->GetMissionPtr(MissionConf::Recharging));

  // 充电桩位置纵向偏移（base_link 坐标系 x 方向，正数向前）
  charging_pose.pose.position.x() += recharging->charging_station_offset_x();

  recharging->SetChargingStation(charging_pose);
  auto it = timeout_states_.find(config_.sub_charging_goal().name());
  if (it != timeout_states_.end()) {
    it->second.last_time = charging_pose.timestamp;
  }
}

void RosBridge::battery_states_callback(
    const rte_interface::msg::BatteryStates::SharedPtr msg) {
  BatteryState state;
  if (!msg->states.empty()) {
    bool all_charging = true;
    for (const auto &battery : msg->states) {
      if (battery.charge_state != 1) {
        all_charging = false;
        break;
      }
    }
    state.charge_state = all_charging ? 1 : 0;
    const auto &battery = msg->states[0];
    state.rsoc = battery.rsoc * 0.1;
    state.voltage = battery.total_voltage * 0.01;
    state.current = battery.current * 0.01;
    state.timestamp = TimeKeeper::TimeNowSec();
    state.valid = true;
  }
  auto recharging = dynamic_cast<Recharging *>(
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Recharging));
  if (recharging) {
    recharging->UpdateBatteryState(state);
  }
  if (state.valid) {
    auto it = timeout_states_.find(config_.sub_battery_states().name());
    if (it != timeout_states_.end()) {
      it->second.last_time = TimeKeeper::TimeNowSec();
    }
  }
}

void RosBridge::motion_state_callback(
    const mc_interface::msg::State::SharedPtr msg) {
  auto recharging = dynamic_cast<Recharging *>(
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Recharging));
  if (!recharging) {
    return;
  }
  // 协议转换：仅保留有效 motion 条目，业务判定(DONE等)由 Recharging 解读
  std::vector<MotionState> states;
  for (const auto &ms : msg->state) {
    if (ms.motion != 0) {
      MotionState state;
      state.motion = ms.motion;
      state.state = ms.state;
      state.timestamp = TimeKeeper::TimeNowSec();
      state.valid = true;
      states.push_back(state);
    }
  }
  recharging->UpdateMotionStates(states);
}

} // namespace communication
} // namespace planning
