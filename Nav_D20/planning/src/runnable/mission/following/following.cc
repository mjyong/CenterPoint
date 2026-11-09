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
void Following::spin_s_active() {}
void Following::spin_s_standstill() {}
void Following::spin_s_fault() {}

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

void Following::on_entry_s_off() {}
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

void Following::transition_e_off_passive() {}
void Following::transition_e_off_fault() {}
void Following::transition_e_passive_standby() {}
void Following::transition_e_passive_fault() {}
void Following::transition_e_standby_active() {}
void Following::transition_e_standby_passive() {}
void Following::transition_e_standby_fault() {}
void Following::transition_e_active_standstill() {}
void Following::transition_e_active_standby() {}
void Following::transition_e_active_fault() {}
void Following::transition_e_standstill_active() {}
void Following::transition_e_standstill_standby() {}
void Following::transition_e_standstill_fault() {}
void Following::transition_e_fault_off() {}
void Following::transition_e_fault_passive() {}

} // namespace mission
} // namespace planning
