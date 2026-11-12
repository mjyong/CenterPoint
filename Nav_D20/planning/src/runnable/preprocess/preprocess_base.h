/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:37:20
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-05-05 09:02:31
 */

#pragma once
#include "preprocess_conf.pb.h"
using planning::config::PreprocessConf;

namespace planning {
class PlanningEntry;
namespace preprocess {

class PreprocessBase {

public:
  virtual ~PreprocessBase() = default;
  virtual bool Init(PlanningEntry *planning_entry) = 0;
  virtual bool Proc() = 0;

  const std::string &GetName() const {
    return PreprocessConf::Type_Name(conf_.type());
  }

protected:
  PreprocessConf::Conf conf_;
  PlanningEntry *planning_entry_{nullptr};

private:
  /* data */
};
} // namespace preprocess
} // namespace planning