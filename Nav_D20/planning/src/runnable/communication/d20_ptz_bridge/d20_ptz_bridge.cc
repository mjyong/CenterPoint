#include "src/runnable/communication/d20_ptz_bridge/d20_ptz_bridge.h"
#include "src/common/util/debug_datas.h"
#include "src/common/util/math.h"
#include "src/common/util/tf_buffer.h"
#include "src/runnable/entry/planning_entry.h"

namespace planning {
namespace communication {
using planning::common::util::DebugDatas;
using planning::common::util::normalize_angle;
using planning::common::util::TfBuffer;
using planning::common::util::wrap_angle;

D20PtzBridge::D20PtzBridge(const CommunicationConf::Conf &conf) {
  conf_ = conf;
  config_ = conf_.d20_ptz_bridge_conf();
}

D20PtzBridge::~D20PtzBridge() {}

bool D20PtzBridge::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  node_ = std::make_shared<rclcpp::Node>("d20_ptz_bridge");

  if (config_.has_sub_ptz_state()) {
    sub_ptz_state_ = node_->create_subscription<rte_interface::msg::CurrentPTZ>(
        config_.sub_ptz_state().name(), config_.sub_ptz_state().queue_size(),
        std::bind(&D20PtzBridge::ptz_state_callback, this,
                  std::placeholders::_1));
  }

  if (config_.has_pub_ptz_cmd()) {
    pub_ptz_cmd_ = node_->create_publisher<rte_interface::msg::CurrentPTZ>(
        config_.pub_ptz_cmd().name(), config_.pub_ptz_cmd().queue_size());
  }

  if (config_.has_pub_focus_point()) {
    pub_focus_point_ =
        node_->create_publisher<visualization_msgs::msg::MarkerArray>(
            config_.pub_focus_point().name(),
            config_.pub_focus_point().queue_size());
  }

  if (config_.has_action_ptz_camera()) {
    ptz_camera_server_ = rclcpp_action::create_server<PtzCameraAction>(
        node_, config_.action_ptz_camera().name(),
        std::bind(&D20PtzBridge::handle_ptz_camera_goal, this, _1, _2),
        std::bind(&D20PtzBridge::handle_ptz_camera_cancel, this, _1),
        std::bind(&D20PtzBridge::handle_ptz_camera_accepted, this, _1));
  }
  UpdateCameraTF();
  return true;
}

bool D20PtzBridge::Proc() {
  if (config_.has_pub_ptz_cmd()) {
    static int last_ptz_cmd_sub_count_ = 0;
    const size_t cur_count = pub_ptz_cmd_->get_subscription_count();
    if (cur_count > last_ptz_cmd_sub_count_) {
      rte_interface::msg::CurrentPTZ cmd_msg;
      cmd_msg.header.stamp = node_->now();
      cmd_msg.header.frame_id = "d20_ptz_bridge";
      cmd_msg.p = 0;
      cmd_msg.t = 0;
      cmd_msg.z = 1;
      pub_ptz_cmd_->publish(cmd_msg);
    }
    last_ptz_cmd_sub_count_ = cur_count;
  }

  double time_now = TimeKeeper::TimeNowSec();
  if (config_.sub_ptz_state().has_timeout() && ptz_state_timestamp_ > 0.1) {
    ptz_state_timeout_ =
        time_now - ptz_state_timestamp_ > config_.sub_ptz_state().timeout();
    AWARN_IF(ptz_state_timeout_)
        << config_.sub_ptz_state().name() << ", PTZ state timeout";
  }

  if (exec_info_.exec_state == ExecState::Failed ||
      exec_info_.exec_state == ExecState::Cancelled) {
    if (time_now - exec_info_.exec_end_time > config_.time_to_idle()) {
      exec_info_.exec_state = ExecState::Idle;
    }
  } else if (exec_info_.exec_state == ExecState::Executing) {
    CalcPtzCmd();
    exec_info_.exec_end_time = time_now;
  }
  return true;
}

bool D20PtzBridge::Publish() {
  if (pub_focus_point_) {
    VisualCameraFocusPoint();
  }
  if (pub_ptz_cmd_ && exec_info_.exec_state != ExecState::Idle) {
    rte_interface::msg::CurrentPTZ cmd_msg;
    cmd_msg.header.stamp = node_->now();
    cmd_msg.header.frame_id = "d20_ptz_bridge";
    cmd_msg.p = wrap_angle(exec_info_.ptz_cmd.x()) / M_PI * -180.0;
    cmd_msg.t = exec_info_.ptz_cmd.y() / M_PI * 180.0;
    cmd_msg.z = exec_info_.ptz_cmd.z();
    pub_ptz_cmd_->publish(cmd_msg);
  }
  if (current_ptz_camera_handle_ && current_ptz_camera_handle_->is_active()) {
    auto feedback = std::make_shared<PtzCameraAction::Feedback>();
    feedback->ptz.x = exec_info_.ptz_state.x() / M_PI * 180.0;
    feedback->ptz.y = exec_info_.ptz_state.y() / M_PI * 180.0;
    feedback->ptz.z = exec_info_.ptz_state.z();
    if (exec_info_.exec_state == ExecState::Executing) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_DOING;
      current_ptz_camera_handle_->publish_feedback(feedback);
    } else if (exec_info_.exec_state == ExecState::Succeeded) {
      feedback->state =
          nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_SUCCESS;
      current_ptz_camera_handle_->publish_feedback(feedback);
      // auto result = std::make_shared<PtzCameraAction::Result>();
      // result->message = "Ptz Camera completed";
      // current_ptz_camera_handle_->succeed(result);
      // current_ptz_camera_handle_ = nullptr;
    } else if (exec_info_.exec_state == ExecState::Failed) {
      feedback->state = nav2_interface::msg::NavState::NAV_STA_TASK_EXEC_FAILED;
      current_ptz_camera_handle_->publish_feedback(feedback);
      auto result = std::make_shared<PtzCameraAction::Result>();
      result->message = "Ptz Camera failed";
      current_ptz_camera_handle_->abort(result);
      current_ptz_camera_handle_ = nullptr;
    }
  }
  DebugDatas::Instance()->emplace("ptz_exec_state",
                                  static_cast<int>(exec_info_.exec_state));
  DebugDatas::Instance()->emplace("ptz_cmd_x", exec_info_.ptz_cmd.x());
  DebugDatas::Instance()->emplace("ptz_cmd_y", exec_info_.ptz_cmd.y());
  DebugDatas::Instance()->emplace("ptz_cmd_z", exec_info_.ptz_cmd.z());
  DebugDatas::Instance()->emplace("ptz_state_x", exec_info_.ptz_state.x());
  DebugDatas::Instance()->emplace("ptz_state_y", exec_info_.ptz_state.y());
  DebugDatas::Instance()->emplace("ptz_state_z", exec_info_.ptz_state.z());
  DebugDatas::Instance()->emplace("ptz_state_timestamp", ptz_state_timestamp_);
  rclcpp::spin_some(node_);
  return true;
}

bool D20PtzBridge::CalcPtzCmd() {
  double time_now = TimeKeeper::TimeNowSec();
  Transform3d tf;
  if (exec_info_.sensor_id == 2) {
    if (!TfBuffer::Instance()->get("map", "ptz_infrared", tf)) {
      AERROR << "Failed to get tf from map to ptz_infrared";
      return false;
    }
  } else {
    if (!TfBuffer::Instance()->get("map", "ptz_visible", tf)) {
      AERROR << "Failed to get tf from map to ptz_visible";
      return false;
    }
  }

  auto base_pt = tf.inverse() * exec_info_.target_point;
  double delta_yaw = std::atan2(base_pt.y(), base_pt.x());
  double delta_pitch =
      std::atan2(base_pt.z(), std::hypot(base_pt.x(), base_pt.y()));
  exec_info_.ptz_cmd.x() = exec_info_.ptz_state.x() + delta_yaw;
  exec_info_.ptz_cmd.y() = exec_info_.ptz_state.y() + delta_pitch;
  AINFO << "delta_yaw: " << delta_yaw << ", delta_pitch: " << delta_pitch;
  double dist_to_target = base_pt.norm();
  double target_h = exec_info_.target_point.z();
  double focal_length = config_.cmos_size() * exec_info_.target_scale.z() *
                        config_.cmos_scale_h() * dist_to_target /
                        (target_h + 0.1);
  AINFO << "dist_to_target: " << dist_to_target << ", target_h: " << target_h
        << ", focal_length: " << focal_length;
  focal_length = std::clamp(focal_length, config_.min_focal_length(),
                            config_.max_focal_length());
  exec_info_.ptz_cmd.z() = focal_length / config_.min_focal_length();

  if (fabs(delta_yaw) < config_.pan_delta_th() &&
      fabs(delta_pitch) < config_.tilt_delta_th() &&
      fabs(exec_info_.ptz_cmd.z() - exec_info_.ptz_state.z()) <
          config_.zoom_delta_th()) {
    if (time_now - exec_info_.exec_hold_time > config_.target_hold_time()) {
      exec_info_.exec_state = ExecState::Succeeded;
    }
  } else {
    exec_info_.exec_hold_time = time_now;
  }

  return true;
}

void D20PtzBridge::ptz_state_callback(
    const rte_interface::msg::CurrentPTZ::SharedPtr msg) {
  ptz_state_timestamp_ =
      msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  exec_info_.ptz_state.x() = normalize_angle(msg->p / -180.0 * M_PI);
  exec_info_.ptz_state.y() = msg->t / 180.0 * M_PI;
  exec_info_.ptz_state.z() = msg->z;
  UpdateCameraTF();
}

void D20PtzBridge::UpdateCameraTF() {
  Eigen::Vector3d ptz_translate(0.204199, 0.0, 0.353303);
  Eigen::Vector3d visible_to_ptz(0.065, 0.107, -0.007);
  Eigen::Vector3d infrared_to_ptz(0.054, -0.1088, -0.007);
  Quaternion ptz_rotate;
  Eigen::Vector3d euler(exec_info_.ptz_state.x(), -exec_info_.ptz_state.y(), 0);
  Eigen::AngleAxisd Rx(euler(2), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd Ry(euler(1), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd Rz(euler(0), Eigen::Vector3d::UnitZ());
  ptz_rotate = Rz * Ry * Rx;
  // ptz_visible_to_base
  Transform3d ptz_visible_to_base = Transform3d::Identity();
  ptz_visible_to_base.translate(ptz_translate + visible_to_ptz);
  ptz_visible_to_base.rotate(ptz_rotate);
  TransformStamped ptz_visible_tf(ptz_state_timestamp_, "base_link",
                                  "ptz_visible", ptz_visible_to_base);
  TfBuffer::Instance()->add(ptz_visible_tf);
  // ptz_infrared_to_base
  Transform3d ptz_infrared_to_base = Transform3d::Identity();
  ptz_infrared_to_base.translate(ptz_translate + infrared_to_ptz);
  ptz_infrared_to_base.rotate(ptz_rotate);
  TransformStamped ptz_infrared_tf(ptz_state_timestamp_, "base_link",
                                   "ptz_infrared", ptz_infrared_to_base);
  TfBuffer::Instance()->add(ptz_infrared_tf);

  Transform3d tf;
  if (!TfBuffer::Instance()->get("map", "base_link", tf)) {
    AWARN << "get tf error, map to base_link";
    return;
  }
  // ptz visible to map
  Transform3d ptz_visible_to_map = tf * ptz_visible_to_base;
  TransformStamped ptz_visible_to_map_tf(ptz_state_timestamp_, "map",
                                         "ptz_visible", ptz_visible_to_map);
  TfBuffer::Instance()->add(ptz_visible_to_map_tf);
  // ptz infrared to map
  Transform3d ptz_infrared_to_map = tf * ptz_infrared_to_base;
  TransformStamped ptz_infrared_to_map_tf(ptz_state_timestamp_, "map",
                                          "ptz_infrared", ptz_infrared_to_map);
  TfBuffer::Instance()->add(ptz_infrared_to_map_tf);
}

rclcpp_action::GoalResponse D20PtzBridge::handle_ptz_camera_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const PtzCameraAction::Goal> goal) {
  (void)uuid;
  AINFO << "Received Ptz Camera goal request.";
  if (exec_info_.exec_state != ExecState::Idle) {
    AWARN << "Ptz camera is busy, reject goal.";
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse D20PtzBridge::handle_ptz_camera_cancel(
    const std::shared_ptr<GoalHandlePtzCamera> goal_handle) {
  (void)goal_handle;
  AINFO << "Received Ptz Camera cancel request.";
  if (current_ptz_camera_handle_ == goal_handle) {
    current_ptz_camera_handle_.reset();
  }
  exec_info_.ptz_cmd = Eigen::Vector3d::Zero();
  exec_info_.exec_state = ExecState::Cancelled;
  exec_info_.exec_end_time = TimeKeeper::TimeNowSec();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void D20PtzBridge::handle_ptz_camera_accepted(
    const std::shared_ptr<GoalHandlePtzCamera> goal_handle) {
  current_ptz_camera_handle_ = goal_handle;
  exec_info_.exec_state = ExecState::Executing;
  exec_info_.exec_start_time = TimeKeeper::TimeNowSec();
  exec_info_.sensor_id = goal_handle->get_goal()->sensor_id;
  exec_info_.target_point.x() = goal_handle->get_goal()->target_point.x;
  exec_info_.target_point.y() = goal_handle->get_goal()->target_point.y;
  exec_info_.target_point.z() = goal_handle->get_goal()->target_point.z;
  exec_info_.target_scale.x() = goal_handle->get_goal()->target_scale.x;
  exec_info_.target_scale.y() = goal_handle->get_goal()->target_scale.y;
  exec_info_.target_scale.z() = goal_handle->get_goal()->target_scale.z;
  exec_info_.target_image_scale = goal_handle->get_goal()->img_scale;
}

void D20PtzBridge::VisualCameraFocusPoint() {
  double min_focal = config_.min_focal_length();
  double focal_length = min_focal * exec_info_.ptz_state.z();
  focal_length = std::clamp(focal_length, config_.min_focal_length(),
                            config_.max_focal_length());

  double dist_to_target = focal_length * exec_info_.target_scale.z() /
                          (config_.cmos_size() * exec_info_.target_image_scale *
                           config_.cmos_scale_h());

  Transform3d camera_tf;
  if (!TfBuffer::Instance()->get("map", "ptz_visible", camera_tf)) {
    AERROR << "Failed to get tf from map to ptz_visible";
  }

  Eigen::Vector3d x(1.0, 0.0, 0.0);
  Eigen::Vector3d target_camera = x * dist_to_target;
  Eigen::Vector3d target_world = camera_tf * target_camera;

  visualization_msgs::msg::Marker focus_point_marker;
  focus_point_marker.header.frame_id = "map";
  focus_point_marker.header.stamp = node_->now();
  focus_point_marker.ns = "focus_point";
  focus_point_marker.id = 0;
  focus_point_marker.type = visualization_msgs::msg::Marker::SPHERE;
  focus_point_marker.action = visualization_msgs::msg::Marker::ADD;
  focus_point_marker.pose.position.x = target_world.x();
  focus_point_marker.pose.position.y = target_world.y();
  focus_point_marker.pose.position.z = target_world.z();
  focus_point_marker.scale.x = 0.5;
  focus_point_marker.scale.y = 0.5;
  focus_point_marker.scale.z = 0.5;
  focus_point_marker.color.a = 1.0;
  focus_point_marker.color.r = 1.0;
  focus_point_marker.color.g = 0.0;
  focus_point_marker.color.b = 0.0;

  visualization_msgs::msg::Marker target_point_marker;
  target_point_marker.header.frame_id = "map";
  target_point_marker.header.stamp = node_->now();
  target_point_marker.ns = "target_point";
  target_point_marker.id = 0;
  target_point_marker.type = visualization_msgs::msg::Marker::SPHERE;
  target_point_marker.action = visualization_msgs::msg::Marker::ADD;
  target_point_marker.pose.position.x = exec_info_.target_point.x();
  target_point_marker.pose.position.y = exec_info_.target_point.y();
  target_point_marker.pose.position.z = exec_info_.target_point.z();
  target_point_marker.scale.x = 0.3;
  target_point_marker.scale.y = 0.3;
  target_point_marker.scale.z = 0.3;
  target_point_marker.color.a = 1.0;
  target_point_marker.color.r = 0.0;
  target_point_marker.color.g = 1.0;
  target_point_marker.color.b = 0.0;

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.push_back(focus_point_marker);
  marker_array.markers.push_back(target_point_marker);
  pub_focus_point_->publish(marker_array);
}

void D20PtzBridge::PublishPtzCmd(const Eigen::Vector3d &ptz_cmd) {
  rte_interface::msg::CurrentPTZ cmd_msg;
  cmd_msg.header.stamp = node_->now();
  cmd_msg.header.frame_id = "d20_ptz_bridge";
  cmd_msg.p = wrap_angle(ptz_cmd.x()) / M_PI * -180.0;
  cmd_msg.t = ptz_cmd.y() / M_PI * 180.0;
  cmd_msg.z = ptz_cmd.z();
  pub_ptz_cmd_->publish(cmd_msg);
}

} // namespace communication
} // namespace planning
