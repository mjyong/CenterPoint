/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:37:58
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-15 18:53:42
 */
#include "src/runnable/preprocess/preprocess_factory.h"
#include "src/runnable/preprocess/collision_checker/collision_checker.h"
#include "src/runnable/preprocess/cost_map_production/cost_map_production.h"
#include "src/runnable/preprocess/global_map_manager/global_map_manager.h"

namespace planning {
Factory<
    PreprocessConf::Type, PreprocessBase,
    PreprocessBase *(*)(const PreprocessConf::Conf &conf),
    std::unordered_map<PreprocessConf::Type,
                       PreprocessBase *(*)(const PreprocessConf::Conf &conf),
                       std::hash<int>>>
    PreprocessFactory::preprocess_factory_;

PreprocessFactory::PreprocessFactory() {
  preprocess_factory_.Clear();
  // TODO: regist more class
  preprocess_factory_.Register(
      PreprocessConf::CostMapProduction,
      [](const PreprocessConf::Conf &conf) -> PreprocessBase * {
        return new CostMapProduction(conf);
      });
  preprocess_factory_.Register(
      PreprocessConf::CollisionChecker,
      [](const PreprocessConf::Conf &conf) -> PreprocessBase * {
        return new CollisionChecker(conf);
      });
  preprocess_factory_.Register(
      PreprocessConf::GlobalMapManager,
      [](const PreprocessConf::Conf &conf) -> PreprocessBase * {
        return new GlobalMapManager(conf);
      });
}

std::unique_ptr<PreprocessBase>
PreprocessFactory::CreatePreprocess(const PreprocessConf::Conf &conf) {
  return preprocess_factory_.CreateObject(conf.type(), conf);
}

} // namespace planning
