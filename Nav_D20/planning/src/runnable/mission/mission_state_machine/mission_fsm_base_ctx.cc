/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:48:56
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-07 16:24:59
 */
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"
namespace planning {
namespace mission {
MissionFSMBaseCtx::MissionFSMBaseCtx(MissionBase *mission)
    : mission_(mission), fsm_(*this), fsm_manager_(fsm_) {
  init_flag_ = true;
}

MissionFSMBase::StateType MissionFSMBaseCtx::current_state() const {
  if (fsm_manager_.is(sml::state<MissionFSMBase::s_off>)) {
    return MissionFSMBase::StateType::off;
  } else if (fsm_manager_.is(sml::state<MissionFSMBase::s_passive>)) {
    return MissionFSMBase::StateType::passive;
  } else if (fsm_manager_.is(sml::state<MissionFSMBase::s_standby>)) {
    return MissionFSMBase::StateType::standby;
  } else if (fsm_manager_.is(sml::state<MissionFSMBase::s_active>)) {
    return MissionFSMBase::StateType::active;
  } else if (fsm_manager_.is(sml::state<MissionFSMBase::s_standstill>)) {
    return MissionFSMBase::StateType::standstill;
  } else {
    return MissionFSMBase::StateType::fault;
  }
}

std::string
MissionFSMBaseCtx::state_name(MissionFSMBase::StateType state_type) const {
  if(fsm_.state_names.find(state_type) != fsm_.state_names.end()){
    return fsm_.state_names.at(state_type);
  }
  return "";
}

std::string MissionFSMBaseCtx::current_state_name() const{
  return state_name(current_state());
}

void MissionFSMBaseCtx::Proc() {
  // clang-format off
  switch (current_state()) {
  case MissionFSMBase::StateType::off:
    if (check_e_off_passive())        {fsm_manager_.process_event(MissionFSMBase::e_off_passive{});}
    if (check_e_off_fault())          {fsm_manager_.process_event(MissionFSMBase::e_off_fault{});}
    break;
  case MissionFSMBase::StateType::passive:
    if (check_e_passive_off())        {fsm_manager_.process_event(MissionFSMBase::e_passive_off{});}
    if (check_e_passive_standby())    {fsm_manager_.process_event(MissionFSMBase::e_passive_standby{});}
    if (check_e_passive_fault())      {fsm_manager_.process_event(MissionFSMBase::e_passive_fault{});}
    break;
  case MissionFSMBase::StateType::standby:
    if (check_e_standby_active())     {fsm_manager_.process_event(MissionFSMBase::e_standby_active{});}
    if (check_e_standby_passive())    {fsm_manager_.process_event(MissionFSMBase::e_standby_passive{});}
    if (check_e_standby_fault())      {fsm_manager_.process_event(MissionFSMBase::e_standby_fault{});}
    if (check_e_standby_off())        {fsm_manager_.process_event(MissionFSMBase::e_standby_off{});}
    break;
  case MissionFSMBase::StateType::active:
    if (check_e_active_standstill())  {fsm_manager_.process_event(MissionFSMBase::e_active_standstill{});}
    if (check_e_active_standby())     {fsm_manager_.process_event(MissionFSMBase::e_active_standby{});}
    if (check_e_active_fault())       {fsm_manager_.process_event(MissionFSMBase::e_active_fault{});}
    if (check_e_active_off())         {fsm_manager_.process_event(MissionFSMBase::e_active_off{});}
    break;
  case MissionFSMBase::StateType::standstill:
    if (check_e_standstill_active())  {fsm_manager_.process_event(MissionFSMBase::e_standstill_active{});}
    if (check_e_standstill_standby()) {fsm_manager_.process_event(MissionFSMBase::e_standstill_standby{});}
    if (check_e_standstill_fault())   {fsm_manager_.process_event(MissionFSMBase::e_standstill_fault{});}
    if (check_e_standstill_off())     {fsm_manager_.process_event(MissionFSMBase::e_standstill_off{});}
    break;
  case MissionFSMBase::StateType::fault:
    if (check_e_fault_off())          {fsm_manager_.process_event(MissionFSMBase::e_fault_off{});}
    if (check_e_fault_passive())      {fsm_manager_.process_event(MissionFSMBase::e_fault_passive{});}
    break;
  default:
    break;
  }
  fsm_manager_.process_event(MissionFSMBase::e_spin{});
  // clang-format on
}

void MissionFSMBaseCtx::spin_action(MissionFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case MissionFSMBase::StateType::off:        spin_s_off();         break;
  case MissionFSMBase::StateType::passive:    spin_s_passive();     break;
  case MissionFSMBase::StateType::standby:    spin_s_standby();     break;
  case MissionFSMBase::StateType::active:     spin_s_active();      break;
  case MissionFSMBase::StateType::standstill: spin_s_standstill();  break;
  case MissionFSMBase::StateType::fault:      spin_s_fault();       break;
  default:                                                          break;
  }
  // clang-format on
}

void MissionFSMBaseCtx::on_entry(MissionFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case MissionFSMBase::StateType::off:        on_entry_s_off();         break;
  case MissionFSMBase::StateType::passive:    on_entry_s_passive();     break;
  case MissionFSMBase::StateType::standby:    on_entry_s_standby();     break;
  case MissionFSMBase::StateType::active:     on_entry_s_active();      break;
  case MissionFSMBase::StateType::standstill: on_entry_s_standstill();  break;
  case MissionFSMBase::StateType::fault:      on_entry_s_fault();       break;
  default:                                                              break;
  }
  // clang-format on
}

void MissionFSMBaseCtx::on_exit(MissionFSMBase::StateType state) {
  // clang-format off
  switch (state) {
  case MissionFSMBase::StateType::off:        on_exit_s_off();         break;
  case MissionFSMBase::StateType::passive:    on_exit_s_passive();     break;
  case MissionFSMBase::StateType::standby:    on_exit_s_standby();     break;
  case MissionFSMBase::StateType::active:     on_exit_s_active();      break;
  case MissionFSMBase::StateType::standstill: on_exit_s_standstill();  break;
  case MissionFSMBase::StateType::fault:      on_exit_s_fault();       break;
  default:                                                             break;
  }
  // clang-format on
}
void MissionFSMBaseCtx::transition_action(MissionFSMBase::EventType event) {
  // clang-format off
  switch (event) {
  case MissionFSMBase::EventType::off_passive:          transition_e_off_passive();         break;
  case MissionFSMBase::EventType::off_fault:            transition_e_off_fault();           break;
  case MissionFSMBase::EventType::passive_standby:      transition_e_passive_standby();     break;
  case MissionFSMBase::EventType::passive_fault:        transition_e_passive_fault();       break;
  case MissionFSMBase::EventType::passive_off:          transition_e_passive_off();         break;
  case MissionFSMBase::EventType::standby_active:       transition_e_standby_active();      break;
  case MissionFSMBase::EventType::standby_passive:      transition_e_standby_passive();     break;
  case MissionFSMBase::EventType::standby_fault:        transition_e_standby_fault();       break;
  case MissionFSMBase::EventType::standby_off:          transition_e_standby_off();         break;
  case MissionFSMBase::EventType::active_standstill:    transition_e_active_standstill();   break;
  case MissionFSMBase::EventType::active_standby:       transition_e_active_standby();      break;
  case MissionFSMBase::EventType::active_fault:         transition_e_active_fault();        break;
  case MissionFSMBase::EventType::active_off:           transition_e_active_off();          break;
  case MissionFSMBase::EventType::standstill_active:    transition_e_standstill_active();   break;
  case MissionFSMBase::EventType::standstill_standby:   transition_e_standstill_standby();  break;
  case MissionFSMBase::EventType::standstill_fault:     transition_e_standstill_fault();    break;
  case MissionFSMBase::EventType::standstill_off:       transition_e_standstill_off();      break;
  case MissionFSMBase::EventType::fault_off:            transition_e_fault_off();           break;
  case MissionFSMBase::EventType::fault_passive:        transition_e_fault_passive();       break;
  default:                                                                                  break;
  }
  // clang-format on
}

} // namespace mission
} // namespace planning
