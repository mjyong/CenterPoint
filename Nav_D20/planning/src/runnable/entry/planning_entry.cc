/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-03-04 14:25:51
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-27 09:25:23
 */
#include "src/runnable/entry/planning_entry.h"
#include "src/common/util/time_keeper.h"
using planning::common::util::TimeKeeper;

namespace planning {
PlanningEntry::PlanningEntry()
    : communication_entry_(*this), preprocess_entry_(*this),
      mission_entry_(*this) {}

PlanningEntry::~PlanningEntry() {}

bool PlanningEntry::Init(const CommunicationConf &communication_conf,
                         const PreprocessConf &preprocess_conf,
                         const MissionConf &mission_conf) {
  if (!communication_entry_.Init(communication_conf)) {
    AERROR << "CommunicationEntry init failed.";
    return false;
  }
  if (!preprocess_entry_.Init(preprocess_conf)) {
    AERROR << "PreprocessEntry init failed.";
    return false;
  }
  if (!mission_entry_.Init(mission_conf)) {
    AERROR << "MissionEntry init failed.";
    return false;
  }

  return true;
}

bool PlanningEntry::Proc() {
  ADEBUG << "\n\n\n#############################################################"
           "##########";
  TimeKeeper timekeeper("PlanningEntry::Process");
  timekeeper.RecordTime("MAIN");

  communication_entry_.Proc();
  preprocess_entry_.Proc();
  mission_entry_.Proc();
  communication_entry_.Publish();

  timekeeper.RecordTime("STOP");
  ADEBUG << timekeeper.DumpString();
  return true;
}

} // namespace planning