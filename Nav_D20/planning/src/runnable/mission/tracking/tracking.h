/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 10:45:30
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-02-13 10:45:23
 */
#pragma once
#include <deque>
#include <optional>
#include <string_view>

#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"

#include "src/algorithm/mppi_planner/mppi_planner.h"
#include "src/algorithm/teb_planner/teb_planner.h"

namespace planning {
namespace mission {
class Tracking : public MissionBase {

public:
  Tracking(const MissionConf::Conf &conf);
  ~Tracking();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;

  void setTargetPoint(const Point3d &target_point) {
    target_pose_.position = target_point;
    target_pose_.orientation = Eigen::Quaterniond::Identity();
    constraint_theta_ = false;
  }
  void setTargetPoint(const Point2d &target_point) {
    target_pose_.position.x() = target_point.x();
    target_pose_.position.y() = target_point.y();
    target_pose_.position.z() = 0.0;
    target_pose_.orientation = Eigen::Quaterniond::Identity();
    constraint_theta_ = false;
  }
  void setTargetPose(const Pose3d &target_pose, bool constraint_theta = false) {
    target_pose_ = target_pose;
    constraint_theta_ = constraint_theta;
  }
  void setTargetPose(const Pose2d &target_pose, bool constraint_theta = false) {
    target_pose_.position.x() = target_pose.position.x();
    target_pose_.position.y() = target_pose.position.y();
    target_pose_.position.z() = 0.0;
    Eigen::Vector3d euler(target_pose.theta, 0, 0);
    Eigen::AngleAxisd Rx(euler(2), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(euler(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(euler(0), Eigen::Vector3d::UnitZ());
    target_pose_.orientation = Rz * Ry * Rx;
    constraint_theta_ = constraint_theta;
  }
  void setTargetSpeed(double target_speed) {
    if (mppi_planner_ != nullptr) {
      mppi_planner_->set_target_speed(target_speed);
      return;
    }
    if (teb_planner_ == nullptr) {
      return;
    }
    teb_planner_->set_target_speed(target_speed);
  }
  const Pose3d &target_pose() const { return target_pose_; }
  bool constraint_theta() const { return constraint_theta_; }

  // Global path is owned by GlobalMapManager; forward for the communication
  // bridges that consume it via the MissionBase interface.
  const std::vector<Pose2d> *global_path_2d() override;
  const std::vector<Pose2d> *local_path_2d() override {
    return &local_path_2d_;
  }

  const MissionFSMBase::StateType &mission_status() override {
    return mission_state_;
  }

  bool nav_blocked() const { return nav_blocked_; }

  teb_planner::TebPlanner *teb_planner() const { return teb_planner_.get(); }
  algorithm::MppiPlanner *mppi_planner() const { return mppi_planner_.get(); }

private:
  // friend class MissionFSMBaseCtx;
  bool check_e_off_passive() override;
  bool check_e_off_fault() override;
  bool check_e_passive_standby() override;
  bool check_e_passive_fault() override;
  bool check_e_standby_active() override;
  bool check_e_standby_passive() override;
  bool check_e_standby_fault() override;
  bool check_e_standby_off() override;
  bool check_e_active_standstill() override;
  bool check_e_active_standby() override;
  bool check_e_active_fault() override;
  bool check_e_active_off() override;
  bool check_e_standstill_active() override;
  bool check_e_standstill_standby() override;
  bool check_e_standstill_fault() override;
  bool check_e_standstill_off() override;
  bool check_e_fault_off() override;
  bool check_e_fault_passive() override;
  void transition_e_off_passive() override;
  void transition_e_off_fault() override;
  void transition_e_passive_standby() override;
  void transition_e_passive_fault() override;
  void transition_e_standby_active() override;
  void transition_e_standby_passive() override;
  void transition_e_standby_fault() override;
  void transition_e_active_standstill() override;
  void transition_e_active_standby() override;
  void transition_e_active_fault() override;
  void transition_e_active_off() override;
  void transition_e_standstill_active() override;
  void transition_e_standstill_standby() override;
  void transition_e_standstill_fault() override;
  void transition_e_fault_off() override;
  void transition_e_fault_passive() override;
  void spin_s_off() override;
  void spin_s_passive() override;
  void spin_s_standby() override;
  void spin_s_active() override;
  void spin_s_standstill() override;
  void spin_s_fault() override;
  void on_entry_s_off() override;
  void on_entry_s_passive() override;
  void on_entry_s_standby() override;
  void on_entry_s_active() override;
  void on_entry_s_standstill() override;
  void on_entry_s_fault() override;
  void on_exit_s_off() override;
  void on_exit_s_passive() override;
  void on_exit_s_standby() override;
  void on_exit_s_active() override;
  void on_exit_s_standstill() override;
  void on_exit_s_fault() override;

private:
  bool init_flag_{false};
  MissionFSMBaseCtx mission_fsm_ctx_;
  MissionFSMBase::StateType mission_state_;
  std::shared_ptr<teb_planner::TebPlanner> teb_planner_;
  std::shared_ptr<algorithm::MppiPlanner> mppi_planner_;

  Pose3d target_pose_;
  bool constraint_theta_{false};
  std::vector<Pose2d> local_path_2d_;

  bool update_fail_{false};
  double accumulate_s_{0.0};
  double target_speed_{1.5};
  int run_status_{-1};
  int start_idx_{0};
  int motion_s_idx_{1};
  double global_remain_dis_{0.0};
  double off_condition_met_time_{-1};

  std::deque<std::pair<double, double>> nav_dist_history_;
  double nav_blocked_met_time_{-1};
  bool nav_blocked_{false};

  void UpdateNavBlocked(double time_now, double dist_to_end);
  void ResetNavBlocked();
};

} // namespace mission
} // namespace planning
