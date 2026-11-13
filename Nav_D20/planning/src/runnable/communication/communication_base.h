/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 10:08:29
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-06-23 15:45:15
 */
#pragma once
#include "communication_conf.pb.h"
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/algorithm/yield_gate/yield_gate.h"
#include "grid_map_core/grid_map_core.hpp"
#include <cstdint>
#include <mutex>
#include <pcl/point_types.h>
using planning::config::CommunicationConf;
namespace planning {
class PlanningEntry;

namespace communication {

class CommunicationBase {

public:
  virtual ~CommunicationBase() = default;
  virtual bool Init(PlanningEntry *planning_entry) = 0;
  virtual bool Proc() = 0;
  virtual bool Publish() { return true; }
  virtual bool ok() { return true; }

  virtual const Odometry *odometry() { return nullptr; };
  virtual const grid_map::GridMap *occupied_map_2d() { return nullptr; };
  virtual const Gait *current_gait() { return nullptr; };
  virtual const Pose3dStamped *charging_pose() { return nullptr; };
  // perception's latest tracked objects (in its own frame) with their stamp and a count of the messages so far;
  // false before the first one (crosswalks, T51 review)
  virtual bool lidar_objects(double *stamp, uint64_t *seq, std::vector<algorithm::yield::Object> *objects) {
    (void)stamp;
    (void)seq;
    (void)objects;
    return false;
  }

  const std::string &GetName() const {
    return CommunicationConf::Type_Name(conf_.type());
  }
protected:
  CommunicationConf::Conf conf_;
  PlanningEntry *planning_entry_{nullptr};
};

} // namespace communication
} // namespace planning
