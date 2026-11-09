/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 13:59:55
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-02-13 13:15:18
 */
#include "src/runnable/mission/tracking/tracking.h"
#include "src/common/util/math.h"
#include "src/common/util/tf_buffer.h"
#include "src/runnable/entry/planning_entry.h"

using planning::common::util::normalize_angle;
using planning::common::util::TfBuffer;
namespace planning {
namespace mission {

Tracking::Tracking(const MissionConf::Conf &conf) : mission_fsm_ctx_(this) {
  conf_ = conf;
}

Tracking::~Tracking() {}

bool Tracking::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  ACHECK(planning_entry_ != nullptr) << "planning_entry_ is nullptr";
  grid_path_planner_ = std::make_shared<GridPathPlanner>(
      conf_.tracking_conf().grid_path_planner_conf());

  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm) {
    grid_path_planner_->SetGridMap(gmm->global_map());
  } else {
    if (!grid_path_planner_->initGridMap()) {
      AERROR << "grid_path_planner_ initGridMap failed";
      return false;
    }
  }
  auto cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  teb_planner_ = std::make_shared<teb_planner::TebPlanner>(
      conf_.tracking_conf().teb_planner_conf(),
      cost_map_production->GetIsObstacleFunction(),
      cost_map_production->GetObstacleDistanceFunction(),
      cost_map_production->GetCompositeCostFunction(),
      cost_map_production->GetIsCollisionFunction());
  init_flag_ = true;
  return true;
}

bool Tracking::Proc() {
  mission_fsm_ctx_.Proc();
  mission_state_ = mission_fsm_ctx_.current_state();
  AINFO << "mission_state: " << mission_fsm_ctx_.current_state_name();
  return true;
}

void Tracking::spin_s_off() {}
void Tracking::spin_s_passive() {}
void Tracking::spin_s_standby() {}

void Tracking::spin_s_active() {
  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  const auto &odom = main_bridge->odometry();
  Pose2d start_pose;
  start_pose.position.x() = odom->pose.position.x();
  start_pose.position.y() = odom->pose.position.y();
  start_pose.theta = odom->pose.euler().z();

  planning_entry_->preprocess_entry_ptr()
      ->cost_map_production()
      ->SetGlobalReferencePath(global_path_2d_);

  auto end_pose2d = global_path_2d_.poses().back();
  double dist_to_end = (start_pose.position - end_pose2d.position).norm();
  bool constant_final_theta = (dist_to_end <= 2.0);

  auto local_poses = global_path_2d_.pose_subset(start_pose.position, 5);
  std::vector<Eigen::Vector3d> local_path_2d;
  Transform3d tf;
  TfBuffer::Instance()->get("map", "base_link", tf);
  for (const auto &pose : local_poses) {
    auto base_pt =
        tf.inverse() * Eigen::Vector3d(pose.position.x(), pose.position.y(), 0);
    auto base_theta = normalize_angle(pose.theta - odom->pose.euler().z());
    local_path_2d.emplace_back(
        Eigen::Vector3d(base_pt.x(), base_pt.y(), base_theta));
  }
  if (local_path_2d.empty()) {
    AERROR << "local_path_2d is empty";
    return;
  }

  Eigen::Vector3d current_pose(0.0, 0.0, 0.0);
  Eigen::Vector3d current_velocity(
      odom->twist.linear.x(), odom->twist.linear.y(), odom->twist.angular.z());
  if (!teb_planner_->solver(local_path_2d, current_pose, current_velocity,
                            local_path_2d.back(), constant_final_theta, tf)) {
    AERROR << "teb_planner_ solver failed";
    twist_.timestamp = TimeKeeper::TimeNowSec();
    twist_.twist.linear.x() = 0.0;
    twist_.twist.linear.y() = 0.0;
    twist_.twist.angular.z() = 0.0;
    return;
  }

  local_path_2d_.clear();
  auto coarse_path = teb_planner_->getOptimizePath();
  for (const auto &pose : coarse_path) {
    auto pt = tf * pose;
    auto theta = normalize_angle(odom->pose.euler().z() + pose.z());
    Pose2d pose2d;
    pose2d.position.x() = pt.x();
    pose2d.position.y() = pt.y();
    pose2d.theta = theta;
    local_path_2d_.emplace_back(pose2d);
  }

  auto cmd = teb_planner_->getVelocityCmd(
      conf_.tracking_conf().teb_planner_conf().velocity_cmd_time());
  twist_.timestamp = TimeKeeper::TimeNowSec();
  twist_.twist.linear.x() = cmd.x();
  twist_.twist.linear.y() = cmd.y();
  twist_.twist.angular.z() = cmd.z();
  AINFO << "CMD: " << cmd.transpose();
}

void Tracking::spin_s_standstill() {}

void Tracking::spin_s_fault() {}

bool Tracking::check_e_off_passive() {
  RETURN_VAL_IF2(!init_flag_, false);
  RETURN_VAL_IF2(!need_start(), false)
  return true;
}

bool Tracking::check_e_off_fault() { return false; }
bool Tracking::check_e_passive_standby() { return true; }
bool Tracking::check_e_passive_fault() { return false; }
bool Tracking::check_e_standby_active() {
  RETURN_VAL_IF(!global_plan_flag_, false);
  // auto d20_ptz_bridge =
  //     planning_entry_->communication_entry_ptr()->d20_ptz_bridge();
  // if (d20_ptz_bridge) {
  //   if (target_pose_.position.head(2).isApprox(
  //           Eigen::Vector2d(154.11, 483.57), 1.0)) {
  //     d20_ptz_bridge->PublishPtzCmd(Eigen::Vector3d(-0.41, 0.0, 1.0));
  //   } else if (target_pose_.position.head(2).isApprox(
  //                  Eigen::Vector2d(-615.07, 729.70), 1.0)) {
  //     d20_ptz_bridge->PublishPtzCmd(Eigen::Vector3d(0.1, 0.0, 1.0));
  //   }
  // }
  return true;
}
bool Tracking::check_e_standby_passive() { return false; }
bool Tracking::check_e_standby_fault() {
  RETURN_VAL_IF(!global_plan_flag_, true);
  return false;
}
bool Tracking::check_e_standby_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Tracking::check_e_active_standstill() {
  RETURN_VAL_IF(need_pause(), true);
  return false;
}
bool Tracking::check_e_active_standby() { return false; }
bool Tracking::check_e_active_fault() { return false; }
bool Tracking::check_e_active_off() {
  RETURN_VAL_IF(need_stop(), true);
  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  const auto &odom = main_bridge->odometry();
  auto end_pose2d = global_path_2d_.poses().back();
  double dist_to_end =
      (odom->pose.position.head(2) - end_pose2d.position).norm();
  double yaw_to_end = normalize_angle(end_pose2d.theta - odom->pose.euler()[2]);
  double distance_threshold = conf_.tracking_conf().off_distance_threshold();
  double yaw_threshold = conf_.tracking_conf().off_yaw_threshold();
  double velocity_threshold = conf_.tracking_conf().off_velocity_threshold();
  double angular_velocity_threshold =
      conf_.tracking_conf().off_angular_velocity_threshold();
  double duration_threshold = conf_.tracking_conf().off_duration_threshold();
  bool error_met = abs(twist_.twist.linear.x()) < velocity_threshold &&
                   abs(twist_.twist.linear.y()) < velocity_threshold &&
                   abs(twist_.twist.angular.z()) < angular_velocity_threshold &&
                   abs(dist_to_end) < distance_threshold &&
                   abs(yaw_to_end) < yaw_threshold;
  AINFO << "dist_to_end: " << dist_to_end << ", yaw_to_end: " << yaw_to_end
        << ", error_met: " << error_met;
  if (error_met) {
    if (off_condition_met_time_ < 0) {
      off_condition_met_time_ = TimeKeeper::TimeNowSec();
    } else if (TimeKeeper::TimeNowSec() - off_condition_met_time_ >=
               duration_threshold) {
      off_condition_met_time_ = -1;
      return true;
    }
  } else {
    off_condition_met_time_ = -1;
  }
  return false;
}
bool Tracking::check_e_standstill_active() {
  RETURN_VAL_IF(need_resume(), true);
  return false;
}
bool Tracking::check_e_standstill_standby() { return false; }
bool Tracking::check_e_standstill_fault() { return false; }
bool Tracking::check_e_standstill_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Tracking::check_e_fault_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Tracking::check_e_fault_passive() { return false; }

void Tracking::on_entry_s_off() {
  global_path_2d_.clear();
  local_path_2d_.clear();
  twist_.twist.reset();
}

void Tracking::on_entry_s_passive() {}
void Tracking::on_entry_s_standby() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm) {
    grid_path_planner_->SetGridMap(gmm->global_map());
  }

  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  auto odom = main_bridge->odometry();
  Point2d start_pt(odom->pose.position.x(), odom->pose.position.y());
  Point2d end_pt(target_pose_.position.x(), target_pose_.position.y());
  std::vector<Eigen::Vector2d> path_pts;
  global_plan_flag_ = grid_path_planner_->plan(start_pt, end_pt, path_pts);
  if (global_plan_flag_) {
    while (path_pts.size() < 2) {
      path_pts.emplace_back(Eigen::Vector2d(target_pose_.position.x(),
                                            target_pose_.position.y()));
    }
    global_path_2d_ = std::move(
        Path2d(std::vector<Point2d>(path_pts.begin(), path_pts.end())));
    if (constraint_theta_) {
      global_path_2d_.poses_ptr()->back().theta = target_pose_.euler()[2];
    }
  }
}

void Tracking::on_entry_s_active() {
  if (teb_planner_ != nullptr) {
    teb_planner_->reset();
  }
}

void Tracking::on_entry_s_standstill() { twist_.twist.reset(); }
void Tracking::on_entry_s_fault() { twist_.twist.reset(); }
void Tracking::on_exit_s_off() { twist_.twist.reset(); }

void Tracking::on_exit_s_passive() {}
void Tracking::on_exit_s_standby() {}
void Tracking::on_exit_s_active() {}
void Tracking::on_exit_s_standstill() {}
void Tracking::on_exit_s_fault() {}

void Tracking::transition_e_off_passive() { reset_trigger(); }
void Tracking::transition_e_off_fault() { reset_trigger(); }
void Tracking::transition_e_passive_standby() { reset_trigger(); }
void Tracking::transition_e_passive_fault() { reset_trigger(); }
void Tracking::transition_e_standby_active() { reset_trigger(); }
void Tracking::transition_e_standby_passive() { reset_trigger(); }
void Tracking::transition_e_standby_fault() { reset_trigger(); }
void Tracking::transition_e_active_standstill() { reset_trigger(); }
void Tracking::transition_e_active_standby() { reset_trigger(); }
void Tracking::transition_e_active_fault() { reset_trigger(); }
void Tracking::transition_e_standstill_active() { reset_trigger(); }
void Tracking::transition_e_standstill_standby() { reset_trigger(); }
void Tracking::transition_e_standstill_fault() { reset_trigger(); }
void Tracking::transition_e_fault_off() { reset_trigger(); }
void Tracking::transition_e_fault_passive() { reset_trigger(); }

} // namespace mission
} // namespace planning
