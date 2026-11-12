#include "src/runnable/mission/skill/skill.h"
#include "src/runnable/entry/planning_entry.h"
namespace planning {
namespace mission {

Skill::Skill(const MissionConf::Conf &conf) : mission_fsm_ctx_(this) {
  conf_ = conf;
}

Skill::~Skill() {}

bool Skill::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  ACHECK(planning_entry_ != nullptr) << "planning_entry_ is nullptr";
  init_flag_ = true;
  return true;
}

bool Skill::Proc() {
  mission_fsm_ctx_.Proc();
  mission_state_ = mission_fsm_ctx_.current_state();
  AINFO << "mission_state: " << mission_fsm_ctx_.current_state_name();
  return true;
}

Twist Skill::ProcRotatingSkill() {
  Twist twist;
  const auto *odom =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  const double yaw = odom->pose.euler().z();
  skill_info_.rotating_skill.accumulated_yaw +=
      common::util::normalize_angle(yaw - skill_info_.rotating_skill.last_yaw);
  skill_info_.rotating_skill.last_yaw = yaw;
  const double yaw_error = skill_info_.rotating_skill.requested_rotation -
                           skill_info_.rotating_skill.accumulated_yaw;
  skill_info_.complete =
      std::abs(yaw_error) <= conf_.skill_conf().yaw_tolerance();
  twist.angular.z() =
      std::clamp(1.5 * yaw_error, -conf_.skill_conf().max_omega(),
                 conf_.skill_conf().max_omega());
  AINFO << "yaw_error: " << yaw_error
        << " accumulated_yaw: " << skill_info_.rotating_skill.accumulated_yaw
        << " requested_rotation: "
        << skill_info_.rotating_skill.requested_rotation;
  return twist;
}

Twist Skill::ProcMovingSkill() {
  Twist twist;
  const auto *odom =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  const double yaw = odom->pose.euler().z();
  Point2d target_position_(skill_info_.moving_skill.odom_x,
                           skill_info_.moving_skill.odom_y);
  const Eigen::Vector2d error_map =
      target_position_ - odom->pose.position.head<2>();
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const Eigen::Vector2d error_base(c * error_map.x() + s * error_map.y(),
                                   -s * error_map.x() + c * error_map.y());
  const double yaw_error =
      common::util::normalize_angle(skill_info_.moving_skill.odom_yaw - yaw);
  skill_info_.complete = error_map.norm() <= conf_.skill_conf().position_tolerance() &&
             std::abs(yaw_error) <= conf_.skill_conf().yaw_tolerance();
  twist.linear.x() = std::clamp(1.5 * error_base.x(), -conf_.skill_conf().max_vx(),
                                conf_.skill_conf().max_vx());
  twist.linear.y() = std::clamp(1.5 * error_base.y(), -conf_.skill_conf().max_vy(),
                                conf_.skill_conf().max_vy());
  twist.angular.z() = std::clamp(1.5 * yaw_error, -conf_.skill_conf().max_omega(),
                                 conf_.skill_conf().max_omega());
  AINFO << "error_base: " << error_base.transpose() << " yaw_error: " << yaw_error;
  return twist;
}

void Skill::spin_s_off() {}
void Skill::spin_s_passive() {}
void Skill::spin_s_standby() {}
void Skill::spin_s_active() {
  Twist twist;
  if (skill_info_.skill_type == SkillType::kMoving) {
    twist = ProcMovingSkill();
  } else if (skill_info_.skill_type == SkillType::kRotating) {
    twist = ProcRotatingSkill();
  }
  double now = TimeKeeper::TimeNowSec();
  const double dt = now - twist_.timestamp;
  const double linear_delta = conf_.skill_conf().max_accel() * dt;
  const double angular_delta = conf_.skill_conf().max_angular_accel() * dt;
  twist.linear.x() =
      std::clamp(twist.linear.x(), twist_.twist.linear.x() - linear_delta,
                 twist_.twist.linear.x() + linear_delta);
  twist.linear.y() =
      std::clamp(twist.linear.y(), twist_.twist.linear.y() - linear_delta,
                 twist_.twist.linear.y() + linear_delta);
  twist.angular.z() =
      std::clamp(twist.angular.z(), twist_.twist.angular.z() - angular_delta,
                 twist_.twist.angular.z() + angular_delta);
  twist_.twist = twist;
  twist_.timestamp = now;
}
void Skill::spin_s_standstill() {}
void Skill::spin_s_fault() {}

bool Skill::check_e_off_passive() {
  RETURN_VAL_IF(!init_flag_, false);
  RETURN_VAL_IF(!need_start(), false)
  return true;
}

bool Skill::check_e_off_fault() { return false; }
bool Skill::check_e_passive_standby() { return true; }
bool Skill::check_e_passive_fault() { return false; }
bool Skill::check_e_standby_active() { return true; }
bool Skill::check_e_standby_passive() { return false; }
bool Skill::check_e_standby_fault() { return false; }
bool Skill::check_e_standby_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Skill::check_e_active_standstill() {
  RETURN_VAL_IF(need_pause(), true);
  return false;
}
bool Skill::check_e_active_standby() { return false; }
bool Skill::check_e_active_fault() { return false; }
bool Skill::check_e_active_off() {
  RETURN_VAL_IF(need_stop(), true);
  RETURN_VAL_IF(skill_info_.complete, true);
  return false;
}
bool Skill::check_e_standstill_active() {
  RETURN_VAL_IF(need_resume(), true);
  return false;
}
bool Skill::check_e_standstill_standby() { return false; }
bool Skill::check_e_standstill_fault() { return false; }
bool Skill::check_e_standstill_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Skill::check_e_fault_off() {
  RETURN_VAL_IF(need_stop(), true);
  return false;
}
bool Skill::check_e_fault_passive() { return false; }

void Skill::on_entry_s_off() { twist_.twist.reset(); }
void Skill::on_entry_s_passive() {}
void Skill::on_entry_s_standby() {}
void Skill::on_entry_s_active() {}
void Skill::on_entry_s_standstill() { twist_.twist.reset(); }
void Skill::on_entry_s_fault() { twist_.twist.reset(); }

void Skill::on_exit_s_off() {}
void Skill::on_exit_s_passive() {}
void Skill::on_exit_s_standby() {}
void Skill::on_exit_s_active() {}
void Skill::on_exit_s_standstill() {}
void Skill::on_exit_s_fault() {}

void Skill::transition_e_off_passive() { reset_trigger(); }
void Skill::transition_e_off_fault() { reset_trigger(); }
void Skill::transition_e_passive_standby() { reset_trigger(); }
void Skill::transition_e_passive_fault() { reset_trigger(); }
void Skill::transition_e_standby_active() { reset_trigger(); }
void Skill::transition_e_standby_passive() { reset_trigger(); }
void Skill::transition_e_standby_fault() { reset_trigger(); }
void Skill::transition_e_active_standstill() { reset_trigger(); }
void Skill::transition_e_active_standby() { reset_trigger(); }
void Skill::transition_e_active_fault() { reset_trigger(); }
void Skill::transition_e_active_off() {
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
  skill_info_.complete = false;
}
void Skill::transition_e_standstill_active() { reset_trigger(); }
void Skill::transition_e_standstill_standby() { reset_trigger(); }
void Skill::transition_e_standstill_fault() { reset_trigger(); }
void Skill::transition_e_fault_off() { reset_trigger(); }
void Skill::transition_e_fault_passive() { reset_trigger(); }

} // namespace mission
} // namespace planning
