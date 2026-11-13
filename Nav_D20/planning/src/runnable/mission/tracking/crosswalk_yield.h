// Crossing at the map's crosswalks (docs/PLANNING_INTEGRATION.md section 3, decided 2026-11-12/13): when the global
// path enters a crosswalk, the robot slows down, stops with its centre half a body length + stop_margin before the
// entry line, turns to face across, and asks the yield gate (src/algorithm/yield_gate/) every cycle; once released it
// goes, and on the crosswalk it does not stop for vehicles. No ROS here: Tracking feeds it each cycle and applies
// its output to the local planner's command; RosBridge turns perception's LidarObjectArray into yield::Object.
#pragma once

#include <deque>
#include <set>
#include <string>
#include <vector>

#include "src/algorithm/yield_gate/yield_gate.h"

namespace planning {
namespace mission {

struct CrosswalkYieldConfig {
  bool enable = false;
  double stop_margin = 0.3;          // [m] stop with the centre half a body length + this before the entry line
  double slow_distance = 3.0;        // [m] slow down over this distance before the stop point
  double min_speed_scale = 0.2;      // ... down to this share of the local planner's command
  double hold_distance = 0.1;        // [m] this close to the stop point (or past it): stop and turn
  double align_tolerance_deg = 10.0; // facing across within this: ask the gate
  double align_gain = 1.0;           // [1/s] turning in place
  double max_align_omega = 0.5;      // [rad/s]
  double check_beyond_exit = 1.0;    // [m] the gate checks the path up to the exit plus this
  double lookahead = 40.0;           // [m] crosswalks are looked for this far along the path
  double moving_speed = 0.1;         // [m/s] the robot counts as moving above this
  double max_pose_gap = 0.3;         // [s] perception further than this from the odometry history: stale
  // The robot on a crosswalk without the gate's go: a localization jump while waiting at the curb (the centre stops
  // 0.9 m before the entry), or the robot started on the road. Up to this far past the entry line: hold, face
  // across and ask the gate as at the curb. Deeper: cross (it is on the road; stopping there is worse).
  double max_unreleased_depth = 1.5; // [m]
  // the perception frame (on the ground under the lidar, x right, y forward, heading with the body) in base_link
  // (x forward, y left): D20 A2, lidar at (0.319, 0.043) m (T46)
  double det_to_base_x = 0.319;
  double det_to_base_y = 0.043;
  double det_to_base_yaw_deg = -90.0;
  algorithm::yield::YieldConfig gate;
};

struct CrosswalkYieldOutput {
  enum Mode { kNone = 0, kApproach = 1, kAlign = 2, kWait = 3, kGo = 4, kCross = 5 };
  Mode mode = kNone;
  bool hold = false;          // replace the command with (0, 0, hold_omega)
  double hold_omega = 0.0;
  double speed_scale = 1.0;   // otherwise scale the command
  int crosswalk = -1;         // index in the map's crosswalks
  double s_in = 0.0, s_out = 0.0, s_stop = 0.0;   // [m] along the path from the robot
  double heading_error = 0.0; // [rad] facing across
  algorithm::yield::Decision decision;
  int objects = 0;            // in the last perception message, in the classes the gate yields to
  bool stale = false;
  bool unreleased = false;    // on the crosswalk without the gate's go
  double depth = 0.0;         // [m] then: how far past the entry line (back along the crossing direction)
};

const char *ModeName(CrosswalkYieldOutput::Mode m);

class CrosswalkYield {
 public:
  using Vec2 = algorithm::yield::Vec2;
  void Configure(const CrosswalkYieldConfig &cfg);
  const CrosswalkYieldConfig &config() const { return cfg_; }
  // a new path / mission: forget the crosswalks already crossed and the gate's timers
  void Reset();
  void SetCrosswalks(const std::vector<std::vector<Vec2>> &polygons) { crosswalks_ = polygons; }
  // robot pose history (map frame), for placing perception at its own stamp
  void AddOdom(double stamp, const Vec2 &xy, double yaw);
  // the latest perception message: objects in the perception frame (det), at `stamp` (same clock as the odometry)
  void SetObjects(double stamp, const std::vector<algorithm::yield::Object> &objects_det);
  // one planning cycle; `now` on the odometry's clock. On a crosswalk the robot does not stop once released (go);
  // without a go it holds at most max_unreleased_depth past the entry (see the config)
  CrosswalkYieldOutput Step(double now, const Vec2 &robot_xy, double robot_yaw, double robot_speed,
                            const std::vector<Vec2> &global_path);
  // perception placed in the map at its stamp; false when the pose there is unknown
  bool ObjectsInMap(std::vector<algorithm::yield::Object> *out) const;

 private:
  struct OdomSample {
    double t;
    Vec2 xy;
    double yaw;
  };
  bool PoseAt(double t, Vec2 *xy, double *yaw) const;
  CrosswalkYieldConfig cfg_;
  algorithm::yield::YieldGate gate_;
  std::vector<std::vector<Vec2>> crosswalks_;
  std::deque<OdomSample> odom_;
  double objects_stamp_ = -1.0;
  std::vector<algorithm::yield::Object> objects_det_;
  CrosswalkYieldOutput::Mode mode_ = CrosswalkYieldOutput::kNone;
  int current_ = -1;
  std::set<int> passed_;
};

}  // namespace mission
}  // namespace planning
