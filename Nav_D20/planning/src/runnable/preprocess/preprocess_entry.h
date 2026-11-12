/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:37:32
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-15 18:52:11
 */

#pragma once
#include "src/common/util/time_keeper.h"
#include "src/runnable/preprocess/collision_checker/collision_checker.h"
#include "src/runnable/preprocess/cost_map_production/cost_map_production.h"
#include "src/runnable/preprocess/global_map_manager/global_map_manager.h"
#include "src/runnable/preprocess/preprocess_base.h"
#include "src/runnable/preprocess/preprocess_factory.h"
#include <unordered_map>

namespace planning {

class PlanningEntry;
class PreprocessEntry {

public:
  PreprocessEntry(PlanningEntry &planning_entry);
  ~PreprocessEntry();
  bool Init(const PreprocessConf &conf);
  bool Proc();

  CostMapProduction *cost_map_production() {
    if (preprocess_map_.find(PreprocessConf::CostMapProduction) !=
        preprocess_map_.end()) {
      return static_cast<CostMapProduction *>(
          preprocess_map_[PreprocessConf::CostMapProduction].get());
    } else {
      return nullptr;
    }
  }

  CollisionChecker *collision_checker() {
    if (preprocess_map_.find(PreprocessConf::CollisionChecker) !=
        preprocess_map_.end()) {
      return static_cast<CollisionChecker *>(
          preprocess_map_[PreprocessConf::CollisionChecker].get());
    } else {
      return nullptr;
    }
  }

  GlobalMapManager *global_map_manager() {
    if (preprocess_map_.find(PreprocessConf::GlobalMapManager) !=
        preprocess_map_.end()) {
      return static_cast<GlobalMapManager *>(
          preprocess_map_[PreprocessConf::GlobalMapManager].get());
    } else {
      return nullptr;
    }
  }

private:
  PreprocessBase *GetPreprocessPtr(const PreprocessConf::Type &type) {
    return preprocess_map_[type].get();
  }

private:
  std::unordered_map<PreprocessConf::Type, const PreprocessConf::Conf *,
                     std::hash<int>>
      preprocess_conf_map_;
  std::map<PreprocessConf::Type, std::unique_ptr<PreprocessBase>>
      preprocess_map_;
  PreprocessFactory preprocess_factory_;

  std::vector<PreprocessBase *> preprocess_list_;

  PlanningEntry *planning_entry_{nullptr};
};

} // namespace planning
