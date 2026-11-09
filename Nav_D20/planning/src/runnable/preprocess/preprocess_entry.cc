/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:37:51
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-03-07 19:20:55
 */
#include "src/runnable/preprocess/preprocess_entry.h"
#include "src/common/util/time_keeper.h"
#include "src/runnable/entry/planning_entry.h"
using planning::common::util::TimeKeeper;

namespace planning {

PreprocessEntry::PreprocessEntry(PlanningEntry &planning_entry)
    : planning_entry_(&planning_entry) {}

PreprocessEntry::~PreprocessEntry() {}

bool PreprocessEntry::Init(const PreprocessConf &conf) {
  for (const auto &conf : conf.confs()) {
    preprocess_conf_map_[conf.type()] = &conf;
  }
  for (size_t i = 0; i < conf.types_size(); ++i) {
    auto type = conf.types(i);
    ACHECK(preprocess_conf_map_.find(type) != preprocess_conf_map_.end())
        << "preprocess: " << PreprocessConf::Type_Name(type)
        << " used but not configured";
    auto iter = preprocess_map_.find(type);
    if (iter == preprocess_map_.end()) {
      auto ptr =
          preprocess_factory_.CreatePreprocess(*preprocess_conf_map_[type]);
      AINFO << "Create preprocess: " << PreprocessConf::Type_Name(type);
      ptr->Init(planning_entry_);
      preprocess_list_.push_back(ptr.get());
      preprocess_map_[type] = std::move(ptr);
    } else {
      preprocess_list_.push_back(iter->second.get());
    }
  }

  return true;
}
bool PreprocessEntry::Proc() {
  TimeKeeper timekeeper("PreprocessEntry::Process");
  for (auto *preprocess : preprocess_list_) {
    timekeeper.RecordTime(preprocess->GetName() + " Process");
    if (!preprocess->Proc()) {
      AERROR << "preprocess : " << preprocess->GetName() << ", Got ERROR!";
    }
  }
  timekeeper.RecordTime("STOP");
  ADEBUG << timekeeper.DumpString();
  return true;
}
} // namespace planning