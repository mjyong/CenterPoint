/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 13:59:55
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-16 11:43:50
 */
#include "src/runnable/mission/mission_entry.h"
#include "src/common/util/time_keeper.h"
#include "src/runnable/entry/planning_entry.h"
using planning::common::util::TimeKeeper;
namespace planning {

MissionEntry::MissionEntry(PlanningEntry &planning_entry)
    : planning_entry_(&planning_entry) {}

MissionEntry::~MissionEntry() {}

bool MissionEntry::Init(const MissionConf &conf) {
  reset_idle_time_ = conf.reset_idle_time();
  for (const auto &conf : conf.confs()) {
    mission_conf_map_[conf.type()] = &conf;
  }
  for (size_t i = 0; i < conf.types_size(); ++i) {
    auto type = conf.types(i);
    ACHECK(mission_conf_map_.find(type) != mission_conf_map_.end())
        << "mission: " << MissionConf::Type_Name(type)
        << " used but not configured";
    auto iter = mission_map_.find(type);
    if (iter == mission_map_.end()) {
      auto ptr = mission_factory_.CreateMission(*mission_conf_map_[type]);
      AINFO << "Create mission: " << MissionConf::Type_Name(type);
      ptr->Init(planning_entry_);
      mission_list_.push_back(ptr.get());
      mission_map_[type] = std::move(ptr);
    } else {
      mission_list_.push_back(iter->second.get());
    }
  }

  return true;
}
bool MissionEntry::Proc() {
  if (current_mission_type_ != MissionConf::Idle) {
    TimeKeeper timekeeper("MissionEntry::Process");
    auto current_mission = mission_map_[current_mission_type_].get();
    ACHECK(current_mission != nullptr)
        << MissionConf::Type_Name(current_mission_type_) << " is nullptr";
    timekeeper.RecordTime(current_mission->GetName() + " Process");
    if (!current_mission->Proc()) {
      AERROR << "mission : " << current_mission->GetName() << ", Got ERROR!";
    }

    if (current_mission->mission_status() == MissionFSMBase::StateType::off) {
      if (off_start_time_ < 0) {
        off_start_time_ = TimeKeeper::TimeNowSec();
        AINFO << "Mission " << current_mission->GetName()
              << " switched to off state";
      } else if (TimeKeeper::TimeNowSec() - off_start_time_ >=
                 reset_idle_time_) {
        AINFO << reset_idle_time_ << " seconds elapsed after mission "
              << current_mission->GetName()
              << " switched to off, switching to Idle";
        current_mission_type_ = MissionConf::Idle;
        off_start_time_ = -1;
      }
    } else {
      off_start_time_ = -1;
    }

    timekeeper.RecordTime("STOP");
    ADEBUG << timekeeper.DumpString();
  } else {
    AINFO_EVERY(20) << "Current mission type is Idle";
  }
  return true;
}

} // namespace planning