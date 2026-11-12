/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 09:17:42
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-16 11:38:24
 */
#pragma once
#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_factory.h"
namespace planning {
class PlanningEntry;
class MissionEntry {
public:
  MissionEntry(PlanningEntry &planning_entry);
  ~MissionEntry();
  bool Init(const MissionConf &conf);
  bool Proc();
  MissionConf::Type GetCurrentMissionType() { return current_mission_type_; }
  void SetCurrentMissionType(const MissionConf::Type &type) {
    current_mission_type_ = type;
  }
  MissionBase *GetMissionPtr(const MissionConf::Type &type) {
    return mission_map_[type].get();
  }
  MissionBase *GetCurrentMissionPtr() {
    return mission_map_[current_mission_type_].get();
  }

private:
  std::unordered_map<MissionConf::Type, const MissionConf::Conf *,
                     std::hash<int>>
      mission_conf_map_;
  std::map<MissionConf::Type, std::unique_ptr<MissionBase>> mission_map_;
  MissionFactory mission_factory_;

  std::vector<MissionBase *> mission_list_;

  PlanningEntry *planning_entry_{nullptr};
  MissionConf::Type current_mission_type_{MissionConf::Idle};
  double off_start_time_{-1};
  double reset_idle_time_{3.0};
};

} // namespace planning