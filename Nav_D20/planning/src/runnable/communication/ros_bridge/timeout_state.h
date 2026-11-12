#pragma once

#include <functional>
#include <string>
#include <unordered_map>

namespace planning {
namespace communication {

// Shared per-source timeout bookkeeping: communication sources (odom,
// occupancy grid, joint states and the mission-specific subscriptions owned
// by the bridges) register entries in one map owned by RosBridge;
// RosBridge::checkTimeout() scans it every cycle.
struct TimeoutState {
  double last_time = 0.0;
  bool timed_out = true;
  double threshold = 0.0;
  std::function<bool()> is_active;
};

using TimeoutStates = std::unordered_map<std::string, TimeoutState>;

} // namespace communication
} // namespace planning
