/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 09:17:42
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-02-23 11:02:14
 */
#pragma once
#include "src/common/util/factory.h"
#include "src/runnable/mission/mission_base.h"
#include <memory>
using planning::common::util::Factory;
using namespace planning::mission;

namespace planning {
class MissionFactory {

public:
  MissionFactory();
  virtual ~MissionFactory() = default;
  std::unique_ptr<MissionBase> CreateMission(const MissionConf::Conf &conf);

private:
  static Factory<
      MissionConf::Type, MissionBase,
      MissionBase *(*)(const MissionConf::Conf &conf),
      std::unordered_map<MissionConf::Type,
                         MissionBase *(*)(const MissionConf::Conf &conf),
                         std::hash<int>>>
      mission_factory_;
};
} // namespace planning