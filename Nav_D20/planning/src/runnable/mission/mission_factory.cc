/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-03-04 14:25:51
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-06-23 15:58:11
 */

#include "src/runnable/mission/mission_factory.h"
#include "src/runnable/mission/following/following.h"
#include "src/runnable/mission/tracking/tracking.h"
#include "src/runnable/mission/recharging/recharging.h"

namespace planning {
Factory<MissionConf::Type, MissionBase,
        MissionBase *(*)(const MissionConf::Conf &conf),
        std::unordered_map<MissionConf::Type,
                           MissionBase *(*)(const MissionConf::Conf &conf),
                           std::hash<int>>>
    MissionFactory::mission_factory_;

MissionFactory::MissionFactory() {
  mission_factory_.Clear();
  // TODO: regist more class
  mission_factory_.Register(MissionConf::Tracking,
                            [](const MissionConf::Conf &conf) -> MissionBase * {
                              return new Tracking(conf);
                            });
  mission_factory_.Register(MissionConf::Following,
                            [](const MissionConf::Conf &conf) -> MissionBase * {
                              return new Following(conf);
                            });
  mission_factory_.Register(MissionConf::Recharging,
                            [](const MissionConf::Conf &conf) -> MissionBase * {
                              return new Recharging(conf);
                            });
}

std::unique_ptr<MissionBase>
MissionFactory::CreateMission(const MissionConf::Conf &conf) {
  return mission_factory_.CreateObject(conf.type(), conf);
}

} // namespace planning