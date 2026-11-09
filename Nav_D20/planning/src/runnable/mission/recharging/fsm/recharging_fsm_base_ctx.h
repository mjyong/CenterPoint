/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:44:10
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */
#pragma once

#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/recharging/fsm/recharging_fsm_base.h"

#include <boost/sml.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

namespace sml = boost::sml;

namespace planning {
namespace mission {
class Recharging;
// Recharging 内层子状态机上下文：
// 1. 在 Recharging 外层状态机的 passive/standby/active 的 spin 中被驱动(Proc)
// 2. 业务检测/动作钩子直接回调 Recharging(友元)
// 3. 通过 outer request 标志触发外层迁移，外层 check_e_* 读取、transition_e_* 清除
class RechargingFSMBaseCtx {
public:
  explicit RechargingFSMBaseCtx(Recharging *mission);
  void Proc();
  // 复位到 idle 并按业务标志启动：进桩充电 / 退桩
  void Start();
  // 复位到 idle（清执行状态与全部外层请求标志）
  void Reset();
  RechargingFSMBase::StateType current_state() const;
  std::string state_name(RechargingFSMBase::StateType state_type) const;
  std::string current_state_name() const;

  // ---- 外层迁移请求标志（外层 check_e_* 读取） ----
  bool passive_to_standby() const { return passive_to_standby_; }
  bool standby_to_active() const { return standby_to_active_; }
  bool active_to_off() const { return active_to_off_; }
  bool to_fault() const { return to_fault_; }
  // ---- 外层迁移完成后清除对应标志 ----
  void ClearPassiveToStandby() { passive_to_standby_ = false; }
  void ClearStandbyToActive() { standby_to_active_ = false; }

private:
  friend class RechargingFSMBase;
  void transition_action(RechargingFSMBase::EventType event);
  void spin_action(RechargingFSMBase::StateType state);
  void on_entry(RechargingFSMBase::StateType state);
  void on_exit(RechargingFSMBase::StateType state);
  bool init_flag_{false};

  void set_outer_request(RechargingFSMBase::EventType event);
  void clear_outer_requests();

private:
  // clang-format off
  // check：业务检测（含重试计数决策），由 Proc 按当前状态调用
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
  // clang-format on

private:
  RechargingFSMBase fsm_;
  sml::sm<RechargingFSMBase> fsm_manager_;
  Recharging *mission_;

  // 外层迁移请求标志
  bool passive_to_standby_{false};
  bool standby_to_active_{false};
  bool active_to_off_{false};
  bool to_fault_{false};
};
} // namespace mission
} // namespace planning
