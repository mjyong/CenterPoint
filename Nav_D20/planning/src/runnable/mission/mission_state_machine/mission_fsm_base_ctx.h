/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-29 20:44:10
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-07 16:24:53
 */
#pragma once

#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"

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
class MissionFSMBaseCtx {
public:
  explicit MissionFSMBaseCtx(MissionBase *mission);
  void Proc();
  MissionFSMBase::StateType current_state() const;
  std::string state_name(MissionFSMBase::StateType state_type) const;
  std::string current_state_name() const;

private:
  friend class MissionFSMBase;
  void transition_action(MissionFSMBase::EventType event);
  void spin_action(MissionFSMBase::StateType state);
  void on_entry(MissionFSMBase::StateType state);
  void on_exit(MissionFSMBase::StateType state);
  bool init_flag_{false};

private:
  // clang-format off
  bool check_e_off_passive()         { return mission_->check_e_off_passive(); }
  bool check_e_off_fault()           { return mission_->check_e_off_fault(); }
  bool check_e_passive_standby()     { return mission_->check_e_passive_standby(); }
  bool check_e_passive_fault()       { return mission_->check_e_passive_fault(); }
  bool check_e_passive_off()         { return mission_->check_e_passive_off(); }
  bool check_e_standby_active()      { return mission_->check_e_standby_active(); }
  bool check_e_standby_passive()     { return mission_->check_e_standby_passive(); }
  bool check_e_standby_fault()       { return mission_->check_e_standby_fault(); }
  bool check_e_standby_off()         { return mission_->check_e_standby_off(); }
  bool check_e_active_standstill()   { return mission_->check_e_active_standstill(); }
  bool check_e_active_standby()      { return mission_->check_e_active_standby(); }
  bool check_e_active_fault()        { return mission_->check_e_active_fault(); }
  bool check_e_active_off()          { return mission_->check_e_active_off(); }
  bool check_e_standstill_active()   { return mission_->check_e_standstill_active(); }
  bool check_e_standstill_standby()  { return mission_->check_e_standstill_standby(); }
  bool check_e_standstill_fault()    { return mission_->check_e_standstill_fault(); }
  bool check_e_standstill_off()      { return mission_->check_e_standstill_off(); }
  bool check_e_fault_off()           { return mission_->check_e_fault_off(); }
  bool check_e_fault_passive()       { return mission_->check_e_fault_passive(); }

  void transition_e_off_passive()       {mission_->transition_e_off_passive();}
  void transition_e_off_fault()         {mission_->transition_e_off_fault();}
  void transition_e_passive_standby()   {mission_->transition_e_passive_standby();}
  void transition_e_passive_fault()     {mission_->transition_e_passive_fault();}
  void transition_e_passive_off()       {mission_->transition_e_passive_off();}
  void transition_e_standby_active()    {mission_->transition_e_standby_active();}
  void transition_e_standby_passive()   {mission_->transition_e_standby_passive();}
  void transition_e_standby_fault()     {mission_->transition_e_standby_fault();}
  void transition_e_standby_off()       {mission_->transition_e_standby_off();}
  void transition_e_active_standstill() {mission_->transition_e_active_standstill();}
  void transition_e_active_standby()    {mission_->transition_e_active_standby();}
  void transition_e_active_fault()      {mission_->transition_e_active_fault();}
  void transition_e_active_off()        {mission_->transition_e_active_off();}
  void transition_e_standstill_active() {mission_->transition_e_standstill_active();}
  void transition_e_standstill_standby(){mission_->transition_e_standstill_standby();}
  void transition_e_standstill_fault()  {mission_->transition_e_standstill_fault();}
  void transition_e_standstill_off()    {mission_->transition_e_standstill_off();}
  void transition_e_fault_off()         {mission_->transition_e_fault_off();}
  void transition_e_fault_passive()     {mission_->transition_e_fault_passive();}

  void spin_s_off()             {mission_->spin_s_off();}
  void spin_s_passive()         {mission_->spin_s_passive();}
  void spin_s_standby()         {mission_->spin_s_standby();}
  void spin_s_active()          {mission_->spin_s_active();}
  void spin_s_standstill()      {mission_->spin_s_standstill();}
  void spin_s_fault()           {mission_->spin_s_fault();}

  void on_entry_s_off()         {if(init_flag_) mission_->on_entry_s_off();}
  void on_entry_s_passive()     {mission_->on_entry_s_passive();}
  void on_entry_s_standby()     {mission_->on_entry_s_standby();}
  void on_entry_s_active()      {mission_->on_entry_s_active();}
  void on_entry_s_standstill()  {mission_->on_entry_s_standstill();}
  void on_entry_s_fault()       {mission_->on_entry_s_fault();}

  void on_exit_s_off()          {mission_->on_exit_s_off();}
  void on_exit_s_passive()      {mission_->on_exit_s_passive();}
  void on_exit_s_standby()      {mission_->on_exit_s_standby();}
  void on_exit_s_active()       {mission_->on_exit_s_active();}
  void on_exit_s_standstill()   {mission_->on_exit_s_standstill();}
  void on_exit_s_fault()        {mission_->on_exit_s_fault();}
  // clang-format on

private:
  MissionFSMBase fsm_;
  sml::sm<MissionFSMBase> fsm_manager_;
  MissionBase *mission_;
};
} // namespace mission
} // namespace planning