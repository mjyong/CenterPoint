/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 13:59:55
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-02-13 13:15:18
 */
#include "src/runnable/mission/tracking/tracking.h"
#include "src/common/util/debug_datas.h"
#include "src/common/util/diagnostic.h"
#include "src/common/util/math.h"
#include "src/common/util/tf_buffer.h"
#include "src/runnable/entry/planning_entry.h"

using planning::common::util::DebugDatas;
using planning::common::util::Diagnostic;
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

  auto cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  if (conf_.tracking_conf().planner_type() == MissionConf::TrackingConf::MPPI) {
    // Device distance field (plus grid geometry) consumed by the MPPI CUDA
    // backend each cycle; empty when the costmap CUDA pipeline is off, which
    // makes the backend fall back to the CPU scoring path.
    std::function<mppi::cuda::MppiCudaMapField()> cuda_map_field_func;
    if (cost_map_production->UseCuda()) {
      cuda_map_field_func = [cost_map_production]() {
        mppi::cuda::MppiCudaMapField field{};
        const auto &query_ctx = cost_map_production->GetCudaQueryContext();
        const auto *grid_map = cost_map_production->GridMap();
        field.distance_field = query_ctx.distance_field;
        field.rows = query_ctx.map_rows;
        field.cols = query_ctx.map_cols;
        field.resolution = grid_map->getResolution();
        field.map_position_x = grid_map->getPosition().x();
        field.map_position_y = grid_map->getPosition().y();
        field.length_x = grid_map->getLength().x();
        field.length_y = grid_map->getLength().y();
        return field;
      };
    }
    mppi_planner_ = std::make_shared<algorithm::MppiPlanner>(
        conf_.tracking_conf().mppi_planner_conf(),
        cost_map_production->GetIsCompositeObstacleFunction(),
        cost_map_production->GetCompositeObstacleDistanceFunction(),
        std::move(cuda_map_field_func));
  } else {
    teb_planner_ = std::make_shared<teb_planner::TebPlanner>(
        conf_.tracking_conf().teb_planner_conf(),
        cost_map_production->GetIsObstacleFunction(),
        cost_map_production->GetObstacleDistanceFunction(),
        cost_map_production->GetCompositeCostFunction(),
        cost_map_production->GetIsCollisionFunction());
  }
  init_flag_ = true;
  return true;
}

bool Tracking::Proc() {
  mission_fsm_ctx_.Proc();
  mission_state_ = mission_fsm_ctx_.current_state();
  AINFO << "mission_state: " << mission_fsm_ctx_.current_state_name();
  return true;
}

const std::vector<Pose2d> *Tracking::global_path_2d() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm == nullptr) {
    return nullptr;
  }
  return &gmm->global_path().poses();
}

void Tracking::spin_s_off() {}
void Tracking::spin_s_passive() {}
void Tracking::spin_s_standby() {}

void Tracking::spin_s_active() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm == nullptr) {
    AERROR << "Tracking: global_map_manager is nullptr";
    return;
  }
  const auto &global_path_2d = gmm->global_path();

  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  const auto &odom = main_bridge->odometry();
  Pose2d start_pose;
  start_pose.position.x() = odom->pose.position.x();
  start_pose.position.y() = odom->pose.position.y();
  start_pose.theta = odom->pose.euler().z();

  auto end_pose2d = global_path_2d.poses().back();
  double dist_to_end = (start_pose.position - end_pose2d.position).norm();
  bool constant_final_theta = (dist_to_end <= 2.0);

  auto local_poses = gmm->ExtractLocalPath(
      start_pose.position,
      conf_.tracking_conf().local_reference_path_length());
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

  auto *cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  if (cost_map_production != nullptr) {
    cost_map_production->SetLocalReferencePath(local_path_2d);
  }

  Eigen::Vector3d current_pose(0.0, 0.0, 0.0);
  Eigen::Vector3d current_velocity(
      odom->twist.linear.x(), odom->twist.linear.y(), odom->twist.angular.z());
  bool solver_ok = false;
  if (mppi_planner_ != nullptr) {
    solver_ok =
        mppi_planner_->solver(local_path_2d, current_pose, current_velocity);
  } else {
    solver_ok = teb_planner_->solver(
        local_path_2d, current_pose, current_velocity, local_path_2d.back(),
        constant_final_theta, tf, constant_final_theta);
  }
  if (!solver_ok) {
    AERROR << "local planner solver failed";
    twist_.timestamp = TimeKeeper::TimeNowSec();
    twist_.twist.linear.x() = 0.0;
    twist_.twist.linear.y() = 0.0;
    twist_.twist.angular.z() = 0.0;
    return;
  }

  local_path_2d_.clear();
  auto coarse_path = (mppi_planner_ != nullptr)
                         ? mppi_planner_->getOptimizePath()
                         : teb_planner_->getOptimizePath();
  for (const auto &pose : coarse_path) {
    auto pt = tf * pose;
    auto theta = normalize_angle(odom->pose.euler().z() + pose.z());
    Pose2d pose2d;
    pose2d.position.x() = pt.x();
    pose2d.position.y() = pt.y();
    pose2d.theta = theta;
    local_path_2d_.emplace_back(pose2d);
  }

  Eigen::Vector3d cmd;
  if (mppi_planner_ != nullptr) {
    cmd = mppi_planner_->getVelocityCmd(
        conf_.tracking_conf().mppi_planner_conf().velocity_cmd_time());
  } else {
    cmd = teb_planner_->getVelocityCmd(
        conf_.tracking_conf().teb_planner_conf().velocity_cmd_time());
  }
  twist_.timestamp = TimeKeeper::TimeNowSec();
  twist_.twist.linear.x() = cmd.x();
  twist_.twist.linear.y() = cmd.y();
  twist_.twist.angular.z() = cmd.z();
  AINFO << "CMD: " << cmd.transpose();

  UpdateNavBlocked(twist_.timestamp, dist_to_end);
}

void Tracking::UpdateNavBlocked(double time_now, double dist_to_end) {
  const auto &tconf = conf_.tracking_conf();
  if (!tconf.nav_blocked_enable()) {
    return;
  }

  nav_dist_history_.emplace_back(time_now, dist_to_end);
  const double window = tconf.nav_blocked_window();
  while (nav_dist_history_.size() > 2 &&
         time_now - nav_dist_history_.front().first > window) {
    nav_dist_history_.pop_front();
  }

  const auto &newest = nav_dist_history_.back();
  const auto &oldest = nav_dist_history_.front();
  if (newest.first - oldest.first < window) {
    return;
  }

  const double progress_rate =
      (oldest.second - newest.second) / (newest.first - oldest.first);
  const double epsilon = tconf.nav_blocked_progress_epsilon();

  if (nav_blocked_) {
    if (progress_rate > epsilon) {
      nav_blocked_ = false;
      nav_blocked_met_time_ = -1;
      AINFO << "nav_blocked cleared, progress_rate: " << progress_rate;
    }
  } else {
    if (dist_to_end > tconf.off_distance_threshold() &&
        progress_rate < epsilon) {
      if (nav_blocked_met_time_ < 0) {
        nav_blocked_met_time_ = time_now;
      } else if (time_now - nav_blocked_met_time_ >=
                 tconf.nav_blocked_enter_duration()) {
        nav_blocked_ = true;
        AINFO << "nav_blocked raised, progress_rate: " << progress_rate
              << ", dist_to_end: " << dist_to_end;
      }
    } else {
      nav_blocked_met_time_ = -1;
    }
  }

  DebugDatas::Instance()->emplace("nav_progress_rate", progress_rate);
  DebugDatas::Instance()->emplace("nav_blocked", nav_blocked_ ? 1 : 0);
}

void Tracking::ResetNavBlocked() {
  nav_dist_history_.clear();
  nav_blocked_met_time_ = -1;
  nav_blocked_ = false;
}

void Tracking::spin_s_standstill() {
  if (nav_blocked_) {
    Diagnostic::Instance()->emplace("0x04040007");
  }
}

void Tracking::spin_s_fault() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm == nullptr || !gmm->global_plan_flag()) {
    Diagnostic::Instance()->emplace("0x04040005");
  }
}

bool Tracking::check_e_off_passive() {
  RETURN_VAL_IF2(!init_flag_, false);
  RETURN_VAL_IF2(!need_start(), false)
  return true;
}

bool Tracking::check_e_off_fault() { return false; }
bool Tracking::check_e_passive_standby() { return true; }
bool Tracking::check_e_passive_fault() { return false; }
bool Tracking::check_e_standby_active() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  RETURN_VAL_IF(gmm == nullptr || !gmm->global_plan_flag(), false);
  return true;
}
bool Tracking::check_e_standby_passive() { return false; }
bool Tracking::check_e_standby_fault() {
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  RETURN_VAL_IF(gmm == nullptr || !gmm->global_plan_flag(), true);
  return false;
}
bool Tracking::check_e_standby_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Tracking::check_e_active_standstill() {
  RETURN_VAL_IF(need_pause(), true);
  RETURN_VAL_IF(nav_blocked_, true);
  return false;
}
bool Tracking::check_e_active_standby() { return false; }
bool Tracking::check_e_active_fault() { return false; }
bool Tracking::check_e_active_off() {
  RETURN_VAL_IF(need_stop(), true);
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  RETURN_VAL_IF(gmm == nullptr, false);
  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  const auto &odom = main_bridge->odometry();
  auto end_pose2d = gmm->global_path().poses().back();
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
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm != nullptr) {
    gmm->ClearGlobalPath();
  }
  auto *cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  if (cost_map_production != nullptr) {
    cost_map_production->SetLocalReferencePath({});
  }
  local_path_2d_.clear();
  twist_.twist.reset();
}

void Tracking::on_entry_s_passive() {}
void Tracking::on_entry_s_standby() {
  ResetNavBlocked();
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (gmm == nullptr) {
    AERROR << "Tracking: global_map_manager is nullptr, global plan skipped";
    return;
  }

  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  auto odom = main_bridge->odometry();
  Point2d start_pt(odom->pose.position.x(), odom->pose.position.y());
  Point2d end_pt(target_pose_.position.x(), target_pose_.position.y());
  std::optional<double> end_theta;
  if (constraint_theta_) {
    end_theta = target_pose_.euler()[2];
  }
  gmm->PlanGlobalPath(start_pt, end_pt, end_theta);
}

void Tracking::on_entry_s_active() {
  ResetNavBlocked();
  if (teb_planner_ != nullptr) {
    teb_planner_->reset();
  }
  if (mppi_planner_ != nullptr) {
    mppi_planner_->reset();
  }
}

void Tracking::on_entry_s_standstill() { twist_.twist.reset(); }
void Tracking::on_entry_s_fault() {
  auto *cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  if (cost_map_production != nullptr) {
    cost_map_production->SetLocalReferencePath({});
  }
  twist_.twist.reset();
}
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
void Tracking::transition_e_active_off() {
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}

void Tracking::transition_e_standstill_active() { reset_trigger(); }
void Tracking::transition_e_standstill_standby() { reset_trigger(); }
void Tracking::transition_e_standstill_fault() { reset_trigger(); }
void Tracking::transition_e_fault_off() { reset_trigger(); }
void Tracking::transition_e_fault_passive() { reset_trigger(); }

} // namespace mission
} // namespace planning
