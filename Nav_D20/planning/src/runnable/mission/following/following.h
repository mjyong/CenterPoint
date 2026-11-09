#pragma once

#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"

namespace planning {
namespace mission {
class Following : public MissionBase {
public:
  Following(const MissionConf::Conf &conf);
  ~Following();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  const MissionFSMBase::StateType &mission_status() override {
    return mission_state_;
  }
  void set_target_position(const double x, const double y) {
    target_position_.x() = x;
    target_position_.y() = y;
  }

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
  Eigen::Vector2d target_position_;
};
} // namespace mission
} // namespace planning