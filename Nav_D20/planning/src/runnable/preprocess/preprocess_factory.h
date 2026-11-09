/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:37:44
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-02-23 09:31:44
 */
#pragma once
#include "src/common/util/factory.h"
#include "src/runnable/preprocess/preprocess_base.h"
using planning::common::util::Factory;
using namespace planning::preprocess;

namespace planning {
class PreprocessFactory {

public:
  PreprocessFactory();
  virtual ~PreprocessFactory() = default;
  std::unique_ptr<PreprocessBase>
  CreatePreprocess(const PreprocessConf::Conf &conf);

private:
  static Factory<
      PreprocessConf::Type, PreprocessBase,
      PreprocessBase *(*)(const PreprocessConf::Conf &conf),
      std::unordered_map<PreprocessConf::Type,
                         PreprocessBase *(*)(const PreprocessConf::Conf &conf),
                         std::hash<int>>>
      preprocess_factory_;
};
} // namespace planning
