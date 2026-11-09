/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 09:15:12
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-05 11:00:37
 */
#pragma once
#include "src/common/util/time_keeper.h"
#include "src/runnable/communication/communication_entry.h"
#include "src/runnable/mission/mission_entry.h"
#include "src/runnable/preprocess/preprocess_entry.h"
using planning::CommunicationEntry;
using planning::MissionEntry;
using planning::PreprocessEntry;
using planning::common::util::TimeKeeper;
using namespace planning::config;
namespace planning {
class PlanningEntry {
public:
  PlanningEntry();
  ~PlanningEntry();

  bool Init(const CommunicationConf &communication_conf,
            const PreprocessConf &preprocess_conf,
            const MissionConf &mission_conf);
  bool Proc();

  CommunicationEntry *communication_entry_ptr() {
    return &communication_entry_;
  }
  MissionEntry *mission_entry_ptr() { return &mission_entry_; }
  PreprocessEntry *preprocess_entry_ptr() { return &preprocess_entry_; }
  double timestamp() { return TimeKeeper::TimeNowSec(); }

private:
  CommunicationEntry communication_entry_;
  MissionEntry mission_entry_;
  PreprocessEntry preprocess_entry_;
};
} // namespace planning
