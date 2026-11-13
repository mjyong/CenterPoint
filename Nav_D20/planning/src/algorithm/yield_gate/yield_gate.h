// Yield gate: may the robot start across (or keep going along) its path now, given tracked vehicles and their
// predictions? C++ port of dog_perception/yield_gate.py (the reference; docs/PLANNING_INTEGRATION.md section 4),
// compared with it frame by frame (tools/yield_gate_parity/). No ROS here: the planner feeds it.
//
// Frame agnostic: the path, the objects and their predictions are in one 2D frame (the planner: map). Two parts:
//   Assessment a = Assess(path_xy, objects, now, stamp, cfg);   // stateless: what is in the way, and when
//   Decision d = gate.Update(a, now, dog_moving, on_crosswalk);  // YieldGate: hysteresis, timers -> go / wait
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace planning {
namespace algorithm {
namespace yield {

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct YieldConfig {
  std::vector<std::string> classes{"vehicle", "cyclist"};  // the robot yields to these
  double dog_speed = 1.0;        // [m/s] along the path while crossing
  double dog_length = 1.0;       // [m] D20 robot_param.pb.txt
  double dog_width = 0.4;
  double lateral_margin = 0.5;   // [m] each side of the robot
  double vehicle_margin = 0.5;   // [m] each side of a vehicle
  double sigma_k = 1.0;          // the vehicle box also grows by k * sigma of its predicted position
  double min_mode_prob = 0.1;    // prediction modes below this are dropped (the most likely one never)
  double horizon = 8.0;          // [s] vehicles are followed at least this far
  double sigma_growth = 0.3;     // [m/s] position sigma growth past the prediction
  double sigma_max = 1.0;        // [m] cap
  double dt = 0.1;               // [s] time step of the check
  double path_step = 0.1;        // [m] the path is resampled at this
  double path_length = 10.0;     // [m] of the path ahead that is checked
  double pet = 2.0;              // [s] gap rule
  double clear_time = 2.0;       // [s] gap / stop rules: no conflict this long before going again
  double stopped_speed = 0.5;    // [m/s] slower (and not "moving" in the tracker): a standing vehicle
  double stopped_wait = 3.0;     // [s] a standing vehicle near the path is watched this long
  double blocked_wait = 15.0;    // [s] ... and this long when its body is on the path
  double start_zone = 10.0;      // [m] a standing vehicle this close to the path, nose or tail towards it
  double stale_after = 0.3;      // [s] perception older than this: wait
  double young_age = 1.0;        // [s] a track younger than this ...
  double young_range = 30.0;     // [m] ... this close to the path: wait (0: off)
  double turn_range = 25.0;      // [m] a moving vehicle this close may also turn (gap / stop rules; 0: off)
  double turn_radius = 10.0;     // [m]
  double turn_max_deg = 90.0;
  std::string yield_rule = "wait";  // wait (decided 2026-11-12) / gap / stop
  double reaction_time = 1.5;    // [s] stop rule
  double brake_decel = 3.0;      // [m/s^2] stop rule
  double stop_margin = 1.0;      // [s] stop rule
  double wait_reach = 60.0;      // [m] wait rule: a moving vehicle is followed this far ...
  double wait_max_horizon = 30.0;  // [s] ... for at most this long
  double wait_margin = 1.5;      // [m] wait rule: each side of a vehicle, on top of vehicle_margin
  double wait_clear = 3.0;       // [s] wait rule: a vehicle waited for must stay clear this long
  double lost_hold = 5.0;        // [s] wait rule: a vehicle waited for and then lost is waited for this long
  double min_watch = 0.0;        // [s] watch at the curb at least this long (T50: does not help; keep 0)
};

enum class Motion { kStatic, kMoving, kStopped };

struct PredictionMode {
  double probability = 1.0;
  std::vector<Vec2> positions;  // at Object::pred_times
  std::vector<Mat2> covariances;  // per position, or empty
};

struct Object {
  int id = 0;
  std::string label;           // vehicle / pedestrian / cyclist
  Motion motion = Motion::kStatic;
  Vec2 position = Vec2::Zero();
  Vec2 velocity = Vec2::Zero();
  double yaw = 0.0;
  double length = 0.0, width = 0.0;
  bool has_age = false;
  double age = 0.0;            // [s]
  bool has_covariance = false;
  Mat2 covariance = Mat2::Zero();  // [x, y] block of the track covariance
  bool has_standing_for = false;
  double standing_for = 0.0;   // [s] how long the tracker has seen it standing
  std::vector<double> pred_times;     // [s] after the perception stamp; empty: constant velocity
  std::vector<PredictionMode> modes;  // same length as pred_times each
};

struct Conflict {
  int id;
  double gap;   // [s] NaN: young; stop rule: slack; wait rule: when it reaches the path
  double s;     // [m] along the path
  std::string kind;  // predicted / turn / young
};

struct Standing {
  int id;
  bool has_standing_for;
  double standing_for;
  bool blocking;  // its body is on the path
};

struct Assessment {
  double now = 0.0;
  bool stale = false;
  std::vector<Conflict> conflicts;
  std::vector<Standing> standing;
  std::set<int> approaching;
  std::set<int> present;
  bool in_lane = false;  // the robot's footprint is in a moving vehicle's swept body
  double min_gap = kInf;
};

struct Decision {
  bool go = true;
  std::string reason;  // clear / committed / on_crosswalk / vehicle / young / standing / blocked / clearing /
                       // not_passed / lost / stale / watching
  std::vector<int> ids;
  double gap = kInf;
};

// Points every `step` metres along the polyline from its start, at most `length`; and their s.
void ResamplePath(const std::vector<Vec2> &path, double step, double length, std::vector<Vec2> *pts,
                  std::vector<double> *s);
// Even-odd test of a point against a polygon (open or closed).
bool InsidePolygon(const Vec2 &p, const std::vector<Vec2> &polygon);
// Where the path first enters the polygon and leaves it again, [m] along the path; false when it never does.
bool CrosswalkSpan(const std::vector<Vec2> &path, const std::vector<Vec2> &polygon, double *s_in, double *s_out,
                   double step = 0.1);

// `stamp` NaN: no perception at all (stale).
Assessment Assess(const std::vector<Vec2> &path, const std::vector<Object> &objects, double now, double stamp,
                  const YieldConfig &cfg);

class YieldGate {
 public:
  explicit YieldGate(const YieldConfig &cfg = YieldConfig()) : cfg_(cfg) { Reset(); }
  void Reset();
  // on_crosswalk: the robot's centre is on a crosswalk of the map: it does not stop there for vehicles.
  Decision Update(const Assessment &a, double now, bool dog_moving = false, bool on_crosswalk = false);
  const YieldConfig &config() const { return cfg_; }

 private:
  Decision Decide(const Assessment &a, double now, bool dog_moving, bool on_crosswalk);
  YieldConfig cfg_;
  bool paused_ = false;
  double last_conflict_ = -kInf;
  std::set<int> yielded_;
  std::map<int, double> still_since_;
  std::map<int, double> conflict_last_;
  std::map<int, double> seen_last_;
  bool watch_set_ = false;
  double watch_since_ = 0.0;
};

}  // namespace yield
}  // namespace algorithm
}  // namespace planning
