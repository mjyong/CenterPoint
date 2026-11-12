/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:48:56
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */
#include "src/runnable/mission/recharging/fsm/recharging_fsm_base_ctx.h"
#include "src/runnable/mission/recharging/recharging.h"
namespace planning {
namespace mission {
RechargingFSMBaseCtx::RechargingFSMBaseCtx(Recharging *mission)
    : mission_(mission), fsm_(*this), fsm_manager_(fsm_) {
  init_flag_ = true;
}

RechargingFSMBase::StateType RechargingFSMBaseCtx::current_state() const {
  if (fsm_manager_.is(sml::state<RechargingFSMBase::s_idle>)) {
    return RechargingFSMBase::StateType::idle;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_searching>)) {
    return RechargingFSMBase::StateType::searching;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_aligning>)) {
    return RechargingFSMBase::StateType::aligning;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_inserting>)) {
    return RechargingFSMBase::StateType::inserting;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_sitting>)) {
    return RechargingFSMBase::StateType::sitting;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_standing_up>)) {
    return RechargingFSMBase::StateType::standing_up;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_backing_off>)) {
    return RechargingFSMBase::StateType::backing_off;
  } else if (fsm_manager_.is(sml::state<RechargingFSMBase::s_done>)) {
    return RechargingFSMBase::StateType::done;
  }
  return RechargingFSMBase::StateType::idle;
}

std::string
RechargingFSMBaseCtx::state_name(RechargingFSMBase::StateType state_type) const {
  if (fsm_.state_names.find(state_type) != fsm_.state_names.end()) {
    return fsm_.state_names.at(state_type);
  }
  return "";
}

std::string RechargingFSMBaseCtx::current_state_name() const {
  return state_name(current_state());
}

void RechargingFSMBaseCtx::Start() {
  if (current_state() != RechargingFSMBase::StateType::idle) {
    fsm_manager_.process_event(RechargingFSMBase::e_reset{});
  }
  if (mission_->IsBackStake()) {
    fsm_manager_.process_event(RechargingFSMBase::e_start_back_stake{});
  } else {
    fsm_manager_.process_event(RechargingFSMBase::e_start_charge{});
  }
}

void RechargingFSMBaseCtx::Reset() {
  if (current_state() != RechargingFSMBase::StateType::idle) {
    fsm_manager_.process_event(RechargingFSMBase::e_reset{});
  } else {
    clear_outer_requests();
  }
}

void RechargingFSMBaseCtx::Proc() {
  // clang-format off
  switch (current_state()) {
  case RechargingFSMBase::StateType::searching:
    if (check_e_detected())          {fsm_manager_.process_event(RechargingFSMBase::e_detected{});}
    else if (check_e_searching_timeout()) {to_fault_ = true;}
    break;
  case RechargingFSMBase::StateType::aligning:
    if (check_e_aligning_arrived())  {fsm_manager_.process_event(RechargingFSMBase::e_arrived_ready{});}
    else if (check_e_aligning_timeout())   {to_fault_ = true;}
    break;
  case RechargingFSMBase::StateType::inserting:
    if (check_e_inserting_arrived()) {fsm_manager_.process_event(RechargingFSMBase::e_arrived_station{});}
    else if (check_e_inserting_timeout()) {
      if (mission_->TakeRetry())     {fsm_manager_.process_event(RechargingFSMBase::e_retry{});}
      else                           {to_fault_ = true;}
    }
    break;
  case RechargingFSMBase::StateType::sitting:
    if (check_e_charge_ok())         {fsm_manager_.process_event(RechargingFSMBase::e_charge_ok{});}
    else if (check_e_charge_fail()) {
      if (mission_->TakeRetry())     {fsm_manager_.process_event(RechargingFSMBase::e_charge_fail{});}
      else                           {to_fault_ = true;}
    }
    break;
  case RechargingFSMBase::StateType::standing_up:
    if (check_e_motion_done())       {fsm_manager_.process_event(RechargingFSMBase::e_motion_done{});}
    else if (check_e_standing_up_timeout()) {to_fault_ = true;}
    break;
  case RechargingFSMBase::StateType::backing_off:
    if (check_e_backoff_arrived())   {
      if (mission_->IsBackStake())   {fsm_manager_.process_event(RechargingFSMBase::e_backoff_finish{});}
      else                           {fsm_manager_.process_event(RechargingFSMBase::e_backoff_retry_done{});}
    }
    else if (check_e_backoff_timeout())     {to_fault_ = true;}
    break;
  case RechargingFSMBase::StateType::idle:
  case RechargingFSMBase::StateType::done:
  default:
    break;
  }
  fsm_manager_.process_event(RechargingFSMBase::e_spin{});
  // clang-format on
}

void RechargingFSMBaseCtx::spin_action(RechargingFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case RechargingFSMBase::StateType::idle:        spin_s_idle();        break;
  case RechargingFSMBase::StateType::searching:   spin_s_searching();   break;
  case RechargingFSMBase::StateType::aligning:    spin_s_aligning();    break;
  case RechargingFSMBase::StateType::inserting:   spin_s_inserting();   break;
  case RechargingFSMBase::StateType::sitting:     spin_s_sitting();     break;
  case RechargingFSMBase::StateType::standing_up: spin_s_standing_up(); break;
  case RechargingFSMBase::StateType::backing_off: spin_s_backing_off(); break;
  case RechargingFSMBase::StateType::done:        spin_s_done();        break;
  default:                                                              break;
  }
  // clang-format on
}

void RechargingFSMBaseCtx::on_entry(RechargingFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case RechargingFSMBase::StateType::idle:        on_entry_s_idle();        break;
  case RechargingFSMBase::StateType::searching:   on_entry_s_searching();   break;
  case RechargingFSMBase::StateType::aligning:    on_entry_s_aligning();    break;
  case RechargingFSMBase::StateType::inserting:   on_entry_s_inserting();   break;
  case RechargingFSMBase::StateType::sitting:     on_entry_s_sitting();     break;
  case RechargingFSMBase::StateType::standing_up: on_entry_s_standing_up(); break;
  case RechargingFSMBase::StateType::backing_off: on_entry_s_backing_off(); break;
  case RechargingFSMBase::StateType::done:        on_entry_s_done();        break;
  default:                                                                  break;
  }
  // clang-format on
}

void RechargingFSMBaseCtx::on_exit(RechargingFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case RechargingFSMBase::StateType::idle:        on_exit_s_idle();        break;
  case RechargingFSMBase::StateType::searching:   on_exit_s_searching();   break;
  case RechargingFSMBase::StateType::aligning:    on_exit_s_aligning();    break;
  case RechargingFSMBase::StateType::inserting:   on_exit_s_inserting();   break;
  case RechargingFSMBase::StateType::sitting:     on_exit_s_sitting();     break;
  case RechargingFSMBase::StateType::standing_up: on_exit_s_standing_up(); break;
  case RechargingFSMBase::StateType::backing_off: on_exit_s_backing_off(); break;
  case RechargingFSMBase::StateType::done:        on_exit_s_done();        break;
  default:                                                                break;
  }
  // clang-format on
}

void RechargingFSMBaseCtx::transition_action(
    RechargingFSMBase::EventType event) {
  AINFO << "[RechargingFSM] transition: " << fsm_.event_names[event];
  // clang-format off
  switch (event) {
  case RechargingFSMBase::EventType::start_charge:       transition_e_start_charge();      break;
  case RechargingFSMBase::EventType::start_back_stake:   transition_e_start_back_stake();  break;
  case RechargingFSMBase::EventType::detected:           transition_e_detected();          break;
  case RechargingFSMBase::EventType::arrived_ready:      transition_e_arrived_ready();     break;
  case RechargingFSMBase::EventType::arrived_station:    transition_e_arrived_station();   break;
  case RechargingFSMBase::EventType::retry:              transition_e_retry();             break;
  case RechargingFSMBase::EventType::charge_ok:          transition_e_charge_ok();         break;
  case RechargingFSMBase::EventType::charge_fail:        transition_e_charge_fail();       break;
  case RechargingFSMBase::EventType::motion_done:        transition_e_motion_done();       break;
  case RechargingFSMBase::EventType::backoff_retry_done: transition_e_backoff_retry_done();break;
  case RechargingFSMBase::EventType::backoff_finish:     transition_e_backoff_finish();    break;
  case RechargingFSMBase::EventType::reset:              transition_e_reset();             break;
  default:                                                                                 break;
  }
  set_outer_request(event);
  // clang-format on
}

void RechargingFSMBaseCtx::set_outer_request(
    RechargingFSMBase::EventType event) {
  // 内层迁移 -> 外层迁移请求
  switch (event) {
  case RechargingFSMBase::EventType::start_charge:
    // 充电流程启动（idle -> searching）：与退桩一致，
    // 外层 passive -> standby -> active 空拍快速通过，
    // searching/aligning/inserting/sitting 全部在 active 下执行
  case RechargingFSMBase::EventType::start_back_stake:
    // 退桩流程启动（idle -> standing_up，不经过 searching）：
    // 外层 passive -> standby 空拍快速通过
    passive_to_standby_ = true;
    break;
  case RechargingFSMBase::EventType::charge_ok:
  case RechargingFSMBase::EventType::backoff_finish:
    // 充电成功 / 退桩完成，外层 active -> off
    active_to_off_ = true;
    break;
  default:
    break;
  }
}

void RechargingFSMBaseCtx::clear_outer_requests() {
  passive_to_standby_ = false;
  standby_to_active_ = false;
  active_to_off_ = false;
  to_fault_ = false;
}

// ==================== check functions ====================
bool RechargingFSMBaseCtx::check_e_detected()          { return mission_->check_e_detected(); }
bool RechargingFSMBaseCtx::check_e_searching_timeout() { return mission_->check_e_searching_timeout(); }
bool RechargingFSMBaseCtx::check_e_aligning_arrived()  { return mission_->check_e_aligning_arrived(); }
bool RechargingFSMBaseCtx::check_e_aligning_timeout()  { return mission_->check_e_aligning_timeout(); }
bool RechargingFSMBaseCtx::check_e_inserting_arrived() { return mission_->check_e_inserting_arrived(); }
bool RechargingFSMBaseCtx::check_e_inserting_timeout() { return mission_->check_e_inserting_timeout(); }
bool RechargingFSMBaseCtx::check_e_charge_ok()         { return mission_->check_e_charge_ok(); }
bool RechargingFSMBaseCtx::check_e_charge_fail()       { return mission_->check_e_charge_fail(); }
bool RechargingFSMBaseCtx::check_e_motion_done()       { return mission_->check_e_motion_done(); }
bool RechargingFSMBaseCtx::check_e_backoff_arrived()   { return mission_->check_e_backoff_arrived(); }
bool RechargingFSMBaseCtx::check_e_standing_up_timeout() { return mission_->check_e_standing_up_timeout(); }
bool RechargingFSMBaseCtx::check_e_backoff_timeout()   { return mission_->check_e_backoff_timeout(); }

// ==================== transition functions ====================
void RechargingFSMBaseCtx::transition_e_start_charge()       { mission_->transition_e_start_charge(); }
void RechargingFSMBaseCtx::transition_e_start_back_stake()   { mission_->transition_e_start_back_stake(); }
void RechargingFSMBaseCtx::transition_e_detected()           { mission_->transition_e_detected(); }
void RechargingFSMBaseCtx::transition_e_arrived_ready()      { mission_->transition_e_arrived_ready(); }
void RechargingFSMBaseCtx::transition_e_arrived_station()    { mission_->transition_e_arrived_station(); }
void RechargingFSMBaseCtx::transition_e_retry()              { mission_->transition_e_retry(); }
void RechargingFSMBaseCtx::transition_e_charge_ok()          { mission_->transition_e_charge_ok(); }
void RechargingFSMBaseCtx::transition_e_charge_fail()        { mission_->transition_e_charge_fail(); }
void RechargingFSMBaseCtx::transition_e_motion_done()        { mission_->transition_e_motion_done(); }
void RechargingFSMBaseCtx::transition_e_backoff_retry_done() { mission_->transition_e_backoff_retry_done(); }
void RechargingFSMBaseCtx::transition_e_backoff_finish()     { mission_->transition_e_backoff_finish(); }
void RechargingFSMBaseCtx::transition_e_reset()              { mission_->transition_e_reset(); }

// ==================== spin functions ====================
void RechargingFSMBaseCtx::spin_s_idle()        { mission_->spin_s_idle(); }
void RechargingFSMBaseCtx::spin_s_searching()   { mission_->spin_s_searching(); }
void RechargingFSMBaseCtx::spin_s_aligning()    { mission_->spin_s_aligning(); }
void RechargingFSMBaseCtx::spin_s_inserting()   { mission_->spin_s_inserting(); }
void RechargingFSMBaseCtx::spin_s_sitting()     { mission_->spin_s_sitting(); }
void RechargingFSMBaseCtx::spin_s_standing_up() { mission_->spin_s_standing_up(); }
void RechargingFSMBaseCtx::spin_s_backing_off() { mission_->spin_s_backing_off(); }
void RechargingFSMBaseCtx::spin_s_done()        { mission_->spin_s_done(); }

// ==================== on_entry functions ====================
void RechargingFSMBaseCtx::on_entry_s_idle()        { clear_outer_requests(); if (init_flag_) mission_->on_entry_s_idle(); }
void RechargingFSMBaseCtx::on_entry_s_searching() {
  // 充电流程首个执行状态：请求外层 standby -> active（standby 空拍快速通过），
  // searching 及后续 aligning/inserting/sitting 全部在 active 下执行，
  // 与退桩流程（on_entry_s_standing_up）保持一致
  standby_to_active_ = true;
  mission_->on_entry_s_searching();
}
void RechargingFSMBaseCtx::on_entry_s_aligning()    { mission_->on_entry_s_aligning(); }
void RechargingFSMBaseCtx::on_entry_s_inserting()   { mission_->on_entry_s_inserting(); }
void RechargingFSMBaseCtx::on_entry_s_sitting()     { mission_->on_entry_s_sitting(); }
void RechargingFSMBaseCtx::on_entry_s_standing_up() {
  // 退桩流程首个执行状态：请求外层 standby -> active（standby 空拍快速通过）。
  // 充电失败重试路径外层已在 active，不置位。
  if (mission_->IsBackStake()) {
    standby_to_active_ = true;
  }
  mission_->on_entry_s_standing_up();
}
void RechargingFSMBaseCtx::on_entry_s_backing_off() {
  mission_->on_entry_s_backing_off();
}
void RechargingFSMBaseCtx::on_entry_s_done()        { mission_->on_entry_s_done(); }

// ==================== on_exit functions ====================
void RechargingFSMBaseCtx::on_exit_s_idle()        { mission_->on_exit_s_idle(); }
void RechargingFSMBaseCtx::on_exit_s_searching()   { mission_->on_exit_s_searching(); }
void RechargingFSMBaseCtx::on_exit_s_aligning()    { mission_->on_exit_s_aligning(); }
void RechargingFSMBaseCtx::on_exit_s_inserting()   { mission_->on_exit_s_inserting(); }
void RechargingFSMBaseCtx::on_exit_s_sitting()     { mission_->on_exit_s_sitting(); }
void RechargingFSMBaseCtx::on_exit_s_standing_up() { mission_->on_exit_s_standing_up(); }
void RechargingFSMBaseCtx::on_exit_s_backing_off() { mission_->on_exit_s_backing_off(); }
void RechargingFSMBaseCtx::on_exit_s_done()        { mission_->on_exit_s_done(); }

} // namespace mission
} // namespace planning
