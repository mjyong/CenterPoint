/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 13:59:55
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-12-27 09:43:21
 */
#include "src/runnable/communication/communication_entry.h"
#include "src/common/util/time_keeper.h"
#include "src/runnable/entry/planning_entry.h"
using planning::common::util::TimeKeeper;
namespace planning {

CommunicationEntry::CommunicationEntry(PlanningEntry &planning_entry)
    : planning_entry_(&planning_entry) {}

CommunicationEntry::~CommunicationEntry() {}

bool CommunicationEntry::Init(const CommunicationConf &conf) {
  for (const auto &conf : conf.confs()) {
    communication_conf_map_[conf.type()] = &conf;
  }
  for (size_t i = 0; i < conf.types_size(); ++i) {
    auto type = conf.types(i);
    ACHECK(communication_conf_map_.find(type) != communication_conf_map_.end())
        << "communication: " << CommunicationConf::Type_Name(type)
        << " used but not configured";
    auto iter = communication_map_.find(type);
    if (conf.main_type() != type && conf.debug_type() != type &&
        !communication_conf_map_[type]->assist()) {
      continue;
    }
    if (iter == communication_map_.end()) {
      auto ptr = communication_factory_.CreateCommunication(
          *communication_conf_map_[type]);
      AINFO << "Create communication: " << CommunicationConf::Type_Name(type);
      ptr->Init(planning_entry_);
      communication_list_.push_back(ptr.get());
      communication_map_[type] = std::move(ptr);
    } else {
      communication_list_.push_back(iter->second.get());
    }
  }
  if (communication_map_.find(conf.main_type()) != communication_map_.end()) {
    main_bridge_ = communication_map_[conf.main_type()].get();
  }
  ACHECK(main_bridge_ != nullptr) << "main_bridge_ is nullptr";

  if (communication_map_.find(conf.debug_type()) != communication_map_.end()) {
    debug_bridge_ = communication_map_[conf.debug_type()].get();
  }
  ACHECK(debug_bridge_ != nullptr) << "debug_bridge_ is nullptr";
  return true;
}

bool CommunicationEntry::Proc() {
  TimeKeeper timekeeper("CommunicationEntry::Process");
  for (auto *communication : communication_list_) {
    timekeeper.RecordTime(communication->GetName() + " Process");
    if (!communication->Proc()) {
      AERROR << "communication : " << communication->GetName()
             << ", Got ERROR!";
    }
  }
  timekeeper.RecordTime("STOP");
  ADEBUG << timekeeper.DumpString();
  return true;
}

void CommunicationEntry::Publish() {
  TimeKeeper timekeeper("CommunicationEntry::Publish");
  for (auto *communication : communication_list_) {
    timekeeper.RecordTime(communication->GetName() + " Publish");
    if (!communication->Publish()) {
      AERROR << "communication : " << communication->GetName()
             << ", Got ERROR!";
    }
  }
  timekeeper.RecordTime("STOP");
  ADEBUG << timekeeper.DumpString();
}

} // namespace planning
