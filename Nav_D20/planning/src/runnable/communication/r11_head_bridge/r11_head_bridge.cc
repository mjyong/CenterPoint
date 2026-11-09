#include "src/runnable/communication/r11_head_bridge/r11_head_bridge.h"
#include "src/runnable/entry/planning_entry.h"

namespace planning {
namespace communication {

R11HeadBridge::R11HeadBridge(const CommunicationConf::Conf &conf) {
  conf_ = conf;
  config_ = conf_.r11_head_bridge_conf();
}

R11HeadBridge::~R11HeadBridge() {}

bool R11HeadBridge::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  node_ = std::make_shared<rclcpp::Node>("r11_head_bridge");

  if (config_.has_sub_head_state()) {
    sub_head_state_ =
        node_->create_subscription<mc_interface::msg::JointsState>(
            config_.sub_head_state().name(),
            config_.sub_head_state().queue_size(),
            std::bind(&R11HeadBridge::head_state_callback, this,
                      std::placeholders::_1));
  }

  if (config_.has_pub_head_cmd()) {
    pub_head_cmd_ = node_->create_publisher<mc_interface::msg::JointsControl>(
        config_.pub_head_cmd().name(), config_.pub_head_cmd().queue_size());
  }

  sub_test_object_ = node_->create_subscription<geometry_msgs::msg::Quaternion>(
      "/planning/debug/r11/img_object_info", 10,
      std::bind(&R11HeadBridge::test_object_callback, this,
                std::placeholders::_1));

  joint_infos_[HeadMotorID::PITCH] = JointInfo();
  joint_infos_[HeadMotorID::PITCH].param.max_position =
      config_.motor_pitch().max_position();
  joint_infos_[HeadMotorID::PITCH].param.min_position =
      config_.motor_pitch().min_position();
  joint_infos_[HeadMotorID::PITCH].param.kp = config_.motor_pitch().kp();
  joint_infos_[HeadMotorID::PITCH].param.kd = config_.motor_pitch().kd();
  joint_infos_[HeadMotorID::ROLL] = JointInfo();
  joint_infos_[HeadMotorID::ROLL].param.max_position =
      config_.motor_roll().max_position();
  joint_infos_[HeadMotorID::ROLL].param.min_position =
      config_.motor_roll().min_position();
  joint_infos_[HeadMotorID::ROLL].param.kp = config_.motor_roll().kp();
  joint_infos_[HeadMotorID::ROLL].param.kd = config_.motor_roll().kd();
  joint_infos_[HeadMotorID::YAW] = JointInfo();
  joint_infos_[HeadMotorID::YAW].param.max_position =
      config_.motor_yaw().max_position();
  joint_infos_[HeadMotorID::YAW].param.min_position =
      config_.motor_yaw().min_position();
  joint_infos_[HeadMotorID::YAW].param.kp = config_.motor_yaw().kp();
  joint_infos_[HeadMotorID::YAW].param.kd = config_.motor_yaw().kd();
  return true;
}

bool R11HeadBridge::Proc() {
  if (config_.has_sub_head_state() && config_.sub_head_state().has_timeout()) {
    double time_now = TimeKeeper::TimeNowSec();
    if (time_now - joint_infos_.begin()->second.feedback.timestamp >
        config_.sub_head_state().timeout()) {
      AWARN << "JointState feedback timeout.";
      return false;
    }
  }

  CalcTrackingCmd();
  rclcpp::spin_some(node_);
  return true;
}

bool R11HeadBridge::Publish() {
  if (pub_head_cmd_) {
    auto mission_entry = planning_entry_->mission_entry_ptr();
    // TODO: following
    if (mission_entry->GetCurrentMissionType() != MissionConf::Tracking) {
      return false;
    }

    // if (mission_entry->GetCurrentMissionType() != MissionConf::Following ||
    //     mission_entry->GetCurrentMissionPtr()->mission_status() !=
    //         MissionFSMBase::StateType::active) {
    //   return false;
    // }
    mc_interface::msg::JointsControl cmd_msg;
    for (auto &joint_info : joint_infos_) {
      joint_info.second.command_clamp();
      mc_interface::msg::Joint joint;
      joint.id = static_cast<uint8_t>(joint_info.first);
      joint.q = joint_info.second.command.q;
      joint.dq = joint_info.second.command.qd;
      joint.tau = joint_info.second.command.tau;
      joint.kp = joint_info.second.command.kp;
      joint.kd = joint_info.second.command.kd;
      cmd_msg.cmds.push_back(joint);
    }
    pub_head_cmd_->publish(cmd_msg);
  }
  return true;
}

void R11HeadBridge::CalcTrackingCmd() {
  double now = TimeKeeper::TimeNowSec();
  if (now - object_info_.timestamp > config_.tracking_timeout()) {
    return;
  }

  double alpha = config_.tracking_lowpass_alpha();
  lpf_img_x_ =
      alpha * object_info_.img_position.x() + (1.0 - alpha) * lpf_img_x_;
  lpf_img_y_ =
      alpha * object_info_.img_position.y() + (1.0 - alpha) * lpf_img_y_;

  double w_norm = object_info_.img_size.x();
  double h_norm = object_info_.img_size.y();

  double dead_zone = std::max(config_.tracking_dead_zone(), w_norm * 0.3);
  double ex = (lpf_img_x_ - 0.5) / w_norm;
  double ey = (lpf_img_y_ - 0.5) / h_norm;

  if (std::fabs(ex) < dead_zone && std::fabs(ey) < dead_zone) {
    return;
  }

  double fill_ratio = w_norm * h_norm;
  double K = config_.tracking_gain() / (1.0 + fill_ratio * 10.0);

  joint_infos_[HeadMotorID::YAW].command.q =
      joint_infos_[HeadMotorID::YAW].feedback.q + K * ex;
  joint_infos_[HeadMotorID::PITCH].command.q =
      joint_infos_[HeadMotorID::PITCH].feedback.q + K * ey;
  joint_infos_[HeadMotorID::ROLL].command.q = 0.0;
}

void R11HeadBridge::head_state_callback(
    const mc_interface::msg::JointsState::SharedPtr msg) {
  MotorInfo motor_feedback;
  motor_feedback.timestamp =
      msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  for (const auto &joint_state : msg->states) {
    motor_feedback.q = joint_state.q;
    motor_feedback.qd = joint_state.dq;
    motor_feedback.tau = joint_state.tau;
    motor_feedback.kp = joint_state.kp;
    motor_feedback.kd = joint_state.kd;
    AINFO << "id: " << (int)(joint_state.id) << ", q: " << motor_feedback.q ;
    if (joint_state.id == static_cast<uint8_t>(HeadMotorID::PITCH)) {
      joint_infos_[HeadMotorID::PITCH].feedback = motor_feedback;
      AINFO << "PITCH: " << joint_infos_[HeadMotorID::PITCH].feedback.q;
    } else if (joint_state.id == static_cast<uint8_t>(HeadMotorID::ROLL)) {
      joint_infos_[HeadMotorID::ROLL].feedback = motor_feedback;
      AINFO << "ROLL: " << joint_infos_[HeadMotorID::ROLL].feedback.q;
    } else if (joint_state.id == static_cast<uint8_t>(HeadMotorID::YAW)) {
      joint_infos_[HeadMotorID::YAW].feedback = motor_feedback;
      AINFO << "YAW: " << joint_infos_[HeadMotorID::YAW].feedback.q;
    }
  }
}

void R11HeadBridge::test_object_callback(
    const geometry_msgs::msg::Quaternion::SharedPtr msg) {
  object_info_.img_position.x() = msg->x;
  object_info_.img_position.y() = msg->y;
  object_info_.img_size.x() = msg->z;
  object_info_.img_size.y() = msg->w;
  object_info_.timestamp = TimeKeeper::TimeNowSec();
}

} // namespace communication
} // namespace planning
