/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2023-11-09 10:22:04
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2024-12-24 09:45:23
 */
#pragma once
#include "mission_conf.pb.h"
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"
using planning::config::MissionConf;

namespace planning {
class PlanningEntry;
namespace mission {

class MissionBase {
  enum class Trigger {
    kNone,
    kStartAtOff,         // Off -> passive
    kPauseAtActive,      // Active -> Standstill
    kResumeAtStandstill, // Standstill -> Active
    kStopAtAny           // Any -> Off
  };

public:
  virtual ~MissionBase() = default;
  virtual bool Init(PlanningEntry *planning_entry) = 0;
  virtual bool Proc() = 0;

  const TwistStamped &GetTwist() const { return twist_; }
  virtual const std::vector<Pose2d> *global_path_2d() { return nullptr; }
  virtual const std::vector<Pose2d> *local_path_2d() { return nullptr; }

  const std::string &GetName() const {
    return MissionConf::Type_Name(conf_.type());
  }
  virtual const MissionFSMBase::StateType &mission_status() = 0;

  void start() { external_trigger_ = Trigger::kStartAtOff; }
  void pause() { external_trigger_ = Trigger::kPauseAtActive; }
  void resume() { external_trigger_ = Trigger::kResumeAtStandstill; }
  void stop() { external_trigger_ = Trigger::kStopAtAny; }
  void reset_trigger() { external_trigger_ = Trigger::kNone; }

private:
  friend class MissionFSMBaseCtx;
  // clang-format off
  virtual bool check_e_off_passive()         {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_off_fault()           {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_passive_standby()     {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_passive_fault()       {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_passive_off()         {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standby_active()      {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standby_passive()     {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standby_fault()       {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standby_off()         {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_active_standstill()   {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_active_standby()      {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_active_fault()        {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_active_off()          {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standstill_active()   {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standstill_standby()  {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standstill_fault()    {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_standstill_off()      {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_fault_off()           {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }
  virtual bool check_e_fault_passive()       {ADEBUG << GetName() << "default func, " << __FUNCTION__; return false; }

  virtual void transition_e_off_passive()                                 {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_off_fault()                                   {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_passive_standby()                             {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_passive_fault()                               {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_passive_off()                                 {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standby_active()                              {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standby_passive()                             {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standby_fault()                               {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standby_off()                                 {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_active_standstill()                           {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_active_standby()                              {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_active_fault()                                {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_active_off()                                  {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standstill_active()                           {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standstill_standby()                          {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standstill_fault()                            {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_standstill_off()                              {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_fault_off()                                   {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void transition_e_fault_passive()                               {ADEBUG << GetName() << "default func, " << __FUNCTION__;}

  virtual void spin_s_off()                                               {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void spin_s_passive()                                           {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void spin_s_standby()                                           {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void spin_s_active()                                            {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void spin_s_standstill()                                        {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void spin_s_fault()                                             {ADEBUG << GetName() << "default func, " << __FUNCTION__;}

  virtual void on_entry_s_off()                                           {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_entry_s_passive()                                       {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_entry_s_standby()                                       {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_entry_s_active()                                        {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_entry_s_standstill()                                    {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_entry_s_fault()                                         {ADEBUG << GetName() << "default func, " << __FUNCTION__;}

  virtual void on_exit_s_off()                                            {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_exit_s_passive()                                        {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_exit_s_standby()                                        {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_exit_s_active()                                         {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_exit_s_standstill()                                     {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  virtual void on_exit_s_fault()                                          {ADEBUG << GetName() << "default func, " << __FUNCTION__;}
  // clang-format on

protected:
  Trigger external_trigger_{Trigger::kNone};
  // clang-format off
  bool need_start() const {return external_trigger_ == Trigger::kStartAtOff; }
  bool need_stop() const {return external_trigger_ == Trigger::kStopAtAny; }
  bool need_pause() const {return external_trigger_ == Trigger::kPauseAtActive; }
  bool need_resume() const {return external_trigger_ == Trigger::kResumeAtStandstill; }
  // clang-format on

  MissionConf::Conf conf_;
  PlanningEntry *planning_entry_{nullptr};
  TwistStamped twist_{};
};
} // namespace mission
} // namespace planning
