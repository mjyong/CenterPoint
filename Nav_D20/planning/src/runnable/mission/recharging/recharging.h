/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 10:45:30
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */
#pragma once
#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"
#include "src/runnable/mission/recharging/fsm/recharging_fsm_base_ctx.h"
#include "src/runnable/mission/recharging/recharging_types.h"

#include <vector>

namespace planning {
namespace mission {
//   内层 recharging_fsm_ctx_ (RechargingFSMBaseCtx)：充电执行子状态机
//     标志(back_stake_)区分两条流程：
//       进桩充电: searching -> aligning -> inserting -> sitting -> done
//                 失败经 standing_up -> backing_off -> aligning 重试
//       退桩:     standing_up -> backing_off -> done
class Recharging : public MissionBase {

public:
  Recharging(const MissionConf::Conf &conf);
  ~Recharging();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;

  const MissionFSMBase::StateType &mission_status() override {
    return mission_state_;
  }
  RechargingFSMBase::StateType recharging_status() const {
    return recharging_fsm_ctx_.current_state();
  }

  void SetChargingStation(const Pose3dStamped &charging_station) {
    charging_station_ = charging_station;
  }
  const Pose3dStamped &charging_station() const { return charging_station_; }
  void SetBackStake(bool back_stake) { back_stake_ = back_stake; }
  bool IsBackStake() const { return back_stake_; }
  double charging_station_offset_x() const {
    return conf_.recharging_conf().charging_station_offset_x();
  }

  void UpdateBatteryState(const BatteryState &state);
  void UpdateMotionStates(const std::vector<MotionState> &states);
  BatteryState battery_state() const;

  bool IsMotionDone(uint8_t motion_id) const;
  void QueueMotionSkill(uint8_t motion, const std::vector<float> &para = {});
  bool TakePendingMotion(uint8_t &motion, std::vector<float> &para);

  // PID controller helpers
  void ResetPid() {
    pid_vx_integral_ = 0.0;   pid_vx_prev_error_ = 0.0;
    pid_vy_integral_ = 0.0;   pid_vy_prev_error_ = 0.0;
    pid_yaw_integral_ = 0.0;  pid_yaw_prev_error_ = 0.0;
  }

private:
  Pose3dStamped ReadyPosition();
  bool IsArrived(const Pose3d &relative_target, double pos_x_tolerance = 0.1,
                 double pos_y_tolerance = 0.1, double euler_tolerance = 0.1,
                 double stable_time = 1e-3);
  bool IsDetected();
  TwistStamped FineTuning(const Pose3d &relative_target);
  double InnerStateElapsed() const;
  bool TakeRetry();

private:
  friend class RechargingFSMBaseCtx;

  // ==================== 外层状态机钩子（MissionBase 虚函数，薄壳） ====================
  bool check_e_off_passive() override;
  bool check_e_passive_standby() override;
  bool check_e_passive_fault() override;
  bool check_e_passive_off() override;
  bool check_e_standby_active() override;
  bool check_e_standby_fault() override;
  bool check_e_standby_off() override;
  bool check_e_active_standstill() override;
  bool check_e_active_off() override;
  bool check_e_active_fault() override;
  bool check_e_standstill_active() override;
  bool check_e_standstill_off() override;
  bool check_e_fault_off() override;
  bool check_e_fault_passive() override;

  void transition_e_off_passive() override;
  void transition_e_passive_standby() override;
  void transition_e_passive_fault() override;
  void transition_e_passive_off() override;
  void transition_e_standby_active() override;
  void transition_e_standby_off() override;
  void transition_e_standby_fault() override;
  void transition_e_active_standstill() override;
  void transition_e_active_off() override;
  void transition_e_active_fault() override;
  void transition_e_standstill_active() override;
  void transition_e_standstill_off() override;
  void transition_e_fault_off() override;
  void transition_e_fault_passive() override;

  void spin_s_off() override;
  void spin_s_passive() override;
  void spin_s_standby() override;
  void spin_s_active() override;
  void spin_s_standstill() override;
  void spin_s_fault() override;

  void on_entry_s_passive() override;
  void on_entry_s_active() override;
  void on_entry_s_standstill() override;
  void on_entry_s_fault() override;

  // ==================== 内层子状态机钩子（由 RechargingFSMBaseCtx 回调） ====================
  bool check_e_detected();
  bool check_e_searching_timeout();
  bool check_e_aligning_arrived();
  bool check_e_aligning_timeout();
  bool check_e_inserting_arrived();
  bool check_e_inserting_timeout();
  bool check_e_charge_ok();
  bool check_e_charge_fail();
  bool check_e_motion_done();
  bool check_e_backoff_arrived();
  bool check_e_standing_up_timeout();
  bool check_e_backoff_timeout();

  void transition_e_start_charge();
  void transition_e_start_back_stake();
  void transition_e_detected();
  void transition_e_arrived_ready();
  void transition_e_arrived_station();
  void transition_e_retry();
  void transition_e_charge_ok();
  void transition_e_charge_fail();
  void transition_e_motion_done();
  void transition_e_backoff_retry_done();
  void transition_e_backoff_finish();
  void transition_e_reset();

  void spin_s_idle();
  void spin_s_searching();
  void spin_s_aligning();
  void spin_s_inserting();
  void spin_s_sitting();
  void spin_s_standing_up();
  void spin_s_backing_off();
  void spin_s_done();

  void on_entry_s_idle();
  void on_entry_s_searching();
  void on_entry_s_aligning();
  void on_entry_s_inserting();
  void on_entry_s_sitting();
  void on_entry_s_standing_up();
  void on_entry_s_backing_off();
  void on_entry_s_done();

  void on_exit_s_idle();
  void on_exit_s_searching();
  void on_exit_s_aligning();
  void on_exit_s_inserting();
  void on_exit_s_sitting();
  void on_exit_s_standing_up();
  void on_exit_s_backing_off();
  void on_exit_s_done();

private:
  bool init_flag_{false};
  // 外层任务状态机 / 内层充电执行子状态机
  MissionFSMBaseCtx mission_fsm_ctx_;
  RechargingFSMBaseCtx recharging_fsm_ctx_;
  MissionFSMBase::StateType mission_state_{MissionFSMBase::StateType::off};

  Pose3dStamped charging_station_;
  Pose3dStamped ready_position_;
  bool back_stake_{false};

  BatteryState battery_state_{};
  std::vector<MotionState> motion_states_;
  double last_motion_send_time_{0.0};
  bool has_pending_motion_{false};
  uint8_t pending_motion_{0};
  std::vector<float> pending_para_;
  bool lie_cmd_sent_{false};

  int retry_count_{0};
  int max_retry_count_{3};

  double inner_state_enter_time_{0.0};
  bool pause_by_data_loss_{false};
  double data_lost_start_{0.0};

  // PID controller state
  double pid_vx_integral_{0.0};
  double pid_vx_prev_error_{0.0};
  double pid_vy_integral_{0.0};
  double pid_vy_prev_error_{0.0};
  double pid_yaw_integral_{0.0};
  double pid_yaw_prev_error_{0.0};

  static constexpr double kChargeWaitTimeout = 30.0;
  static constexpr double kDataLostPauseTime = 2.0;
};

} // namespace mission
} // namespace planning
