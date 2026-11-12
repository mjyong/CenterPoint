#include "src/runnable/mission/following/following.h"
#include "src/runnable/entry/planning_entry.h"
namespace planning {
namespace mission {

Following::Following(const MissionConf::Conf &conf) : mission_fsm_ctx_(this) {
  conf_ = conf;
}

Following::~Following() {}

bool Following::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  ACHECK(planning_entry_ != nullptr) << "planning_entry_ is nullptr";

  const auto &tf_conf = conf_.following_conf().target_follow_conf();
  algorithm::TargetFollowParams tf_params;
  tf_params.headway_time = tf_conf.headway_time();
  tf_params.spacing_gain = tf_conf.spacing_gain();
  tf_params.max_follow_dist = tf_conf.max_follow_dist();
  tf_params.min_follow_dist = tf_conf.min_follow_dist();
  tf_params.speed_filter_window = tf_conf.speed_filter_window();
  tf_params.min_goal_ahead = tf_conf.min_goal_ahead();
  tf_params.arrive_dist_thresh = tf_conf.arrive_dist_thresh();
  tf_params.arrive_theta_thresh = tf_conf.arrive_theta_thresh();
  algorithm::TargetPathExtractParams extract_params;
  const auto &pe_conf = tf_conf.target_path_extract_conf();
  extract_params.filter_window = pe_conf.filter_window();
  extract_params.jump_reset_thresh = pe_conf.jump_reset_thresh();
  extract_params.max_consecutive_jumps = pe_conf.max_consecutive_jumps();
  extract_params.accept_thresh = pe_conf.accept_thresh();
  extract_params.max_path_length = pe_conf.max_path_length();
  extract_params.smooth_resample_interval =
      pe_conf.smooth_resample_interval();

  auto cost_map_production =
      planning_entry_->preprocess_entry_ptr()->cost_map_production();
  target_follow_planner_ =
      std::make_unique<algorithm::TargetFollowPlanner>(
          tf_params, extract_params, tf_conf.teb_planner_conf(),
          cost_map_production->GetIsObstacleFunction(),
          cost_map_production->GetObstacleDistanceFunction(),
          cost_map_production->GetCompositeCostFunction(),
          cost_map_production->GetIsCollisionFunction());
  init_flag_ = true;
  return true;
}

bool Following::Proc() {
  mission_fsm_ctx_.Proc();
  mission_state_ = mission_fsm_ctx_.current_state();
  AINFO << "mission_state: " << mission_fsm_ctx_.current_state_name();
  return true;
}

void Following::spin_s_off() {}
void Following::spin_s_passive() {}
void Following::spin_s_standby() {}
void Following::spin_s_active() { FollowPlan(); }
void Following::spin_s_standstill() { FollowPlan(); }
void Following::spin_s_fault() {}

void Following::FollowPlan() {
  if (!target_position_valid_ || !target_follow_planner_) {
    return;
  }
  auto odometry =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  if (odometry == nullptr) {
    return;
  }
  const Pose2d robot_pose(
      Point2d(odometry->pose.position.x(), odometry->pose.position.y()),
      odometry->pose.euler().z());
  follow_cmd_ = target_follow_planner_->Plan(
      Point2d(target_position_.x(), target_position_.y()),
      Point2d(target_velocity_.x(), target_velocity_.y()), robot_pose,
      Eigen::Vector3d(odometry->twist.linear.x(), odometry->twist.linear.y(),
                      odometry->twist.angular.z()));
  AINFO << "follow_cmd: " << follow_cmd_.value().transpose();
  twist_.timestamp = TimeKeeper::TimeNowSec();
  twist_.twist.linear.x() = follow_cmd_.value().x();
  twist_.twist.linear.y() = follow_cmd_.value().y();
  twist_.twist.angular.z() = follow_cmd_.value().z();
}

bool Following::check_e_off_passive() {
  RETURN_VAL_IF(!init_flag_, false);
  RETURN_VAL_IF(!need_start(), false)
  return true;
}

bool Following::check_e_off_fault() { return false; }
bool Following::check_e_passive_standby() {
  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  RETURN_VAL_IF(!main_bridge->odometry(), false)
  return true;
}
bool Following::check_e_passive_fault() { return false; }
bool Following::check_e_standby_active() { return true; }
bool Following::check_e_standby_passive() { return false; }
bool Following::check_e_standby_fault() { return false; }
bool Following::check_e_standby_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Following::check_e_active_standstill() {
  RETURN_VAL_IF(need_pause(), true);
  return false;
}
bool Following::check_e_active_standby() { return false; }
bool Following::check_e_active_fault() { return false; }
bool Following::check_e_active_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Following::check_e_standstill_active() {
  RETURN_VAL_IF(need_resume(), true);
  return false;
}
bool Following::check_e_standstill_standby() { return false; }
bool Following::check_e_standstill_fault() { return false; }
bool Following::check_e_standstill_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Following::check_e_fault_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Following::check_e_fault_passive() { return false; }

void Following::on_entry_s_off() {
  if (target_follow_planner_) {
    target_follow_planner_->Reset();
  }
  follow_cmd_.reset();
  target_position_valid_ = false;
  twist_.twist.linear.x() = 0.0;
  twist_.twist.linear.y() = 0.0;
  twist_.twist.angular.z() = 0.0;
}
void Following::on_entry_s_passive() {}
void Following::on_entry_s_standby() {}
void Following::on_entry_s_active() {}
void Following::on_entry_s_standstill() {}
void Following::on_entry_s_fault() {}

void Following::on_exit_s_off() {}
void Following::on_exit_s_passive() {}
void Following::on_exit_s_standby() {}
void Following::on_exit_s_active() {}
void Following::on_exit_s_standstill() {}
void Following::on_exit_s_fault() {}

void Following::transition_e_off_passive() {reset_trigger();}
void Following::transition_e_off_fault() {reset_trigger();}
void Following::transition_e_passive_standby() {reset_trigger();}
void Following::transition_e_passive_fault() {reset_trigger();}
void Following::transition_e_standby_active() {reset_trigger();}
void Following::transition_e_standby_passive() {reset_trigger();}
void Following::transition_e_standby_fault() {reset_trigger();}
void Following::transition_e_active_standstill() {reset_trigger();}
void Following::transition_e_active_standby() {reset_trigger();}
void Following::transition_e_active_fault() {reset_trigger();}
void Following::transition_e_active_off() { 
  reset_trigger(); 
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Following::transition_e_standstill_active() {reset_trigger();}
void Following::transition_e_standstill_standby() {reset_trigger();}
void Following::transition_e_standstill_fault() {reset_trigger();}
void Following::transition_e_fault_off() {reset_trigger();}
void Following::transition_e_fault_passive() {reset_trigger();}

} // namespace mission
} // namespace planning
