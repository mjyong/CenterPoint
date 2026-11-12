/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:32:34
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-12-23 17:22:42
 */
#pragma once
#include "src/common/structures.h"
#include "src/runnable/communication/communication_factory.h"
#include "src/runnable/communication/d20_ptz_bridge/d20_ptz_bridge.h"
#include "src/runnable/communication/r11_head_bridge/r11_head_bridge.h"
using namespace planning::common;

namespace planning {
class PlanningEntry;
class CommunicationEntry {

public:
  CommunicationEntry(PlanningEntry &planning_entry);
  ~CommunicationEntry();
  bool Init(const CommunicationConf &conf);
  bool Proc();

  void Publish();
  CommunicationBase *MainBridge() { return main_bridge_; }
  CommunicationBase *DebugBridge() { return debug_bridge_; }
  bool ok() { return main_bridge_->ok(); }

  R11HeadBridge *r11_head_bridge() {
    if (communication_map_.find(CommunicationConf::R11HeadBridge) !=
        communication_map_.end()) {
      return static_cast<R11HeadBridge *>(
          communication_map_[CommunicationConf::R11HeadBridge].get());
    } else {
      return nullptr;
    }
  }
  D20PtzBridge *d20_ptz_bridge() {
    if (communication_map_.find(CommunicationConf::D20PtzBridge) !=
        communication_map_.end()) {
      return static_cast<D20PtzBridge *>(
          communication_map_[CommunicationConf::D20PtzBridge].get());
    } else {
      return nullptr;
    }
  }

private:
  std::unordered_map<CommunicationConf::Type, const CommunicationConf::Conf *,
                     std::hash<int>>
      communication_conf_map_;
  std::map<CommunicationConf::Type, std::unique_ptr<CommunicationBase>>
      communication_map_;
  CommunicationFactory communication_factory_;

  std::vector<CommunicationBase *> communication_list_;

  PlanningEntry *planning_entry_{nullptr};
  CommunicationBase *main_bridge_{nullptr};
  CommunicationBase *debug_bridge_{nullptr};
};

} // namespace planning
