/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-12-11 11:21:47
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-13 10:10:55
 */
#pragma once
#include "src/common/util/log.h"
#include "src/common/util/macros.h"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_ROS
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#endif

namespace planning {
namespace common {
namespace util {
class Diagnostic : public std::set<std::string> {
  struct DiagnosticInfo {
    std::string code;
    std::string module;
    std::string type;
    std::string level;
    std::string name;
    std::string describe;
  };

public:
  // Edge-detected fault transition: emitted once when a fault code appears
  // in the active set and once when it disappears.
  struct FaultEvent {
    DiagnosticInfo info;
    bool active;        // true = raised, false = cleared
    uint64_t timestamp; // milliseconds since epoch, edge observed
  };

#ifdef ENABLE_ROS
  void ConvertToRosMsg(diagnostic_msgs::msg::DiagnosticArray &msg);
#endif
  bool Init(const std::string name, const std::string &filename);

  // Hides std::set::clear: the periodic clear() is the only point where both
  // raising and clearing edges are observable, so the diff against the
  // previous cycle's active set is done here. Callers keep calling clear().
  void clear();

  // Swap out the pending edge events for the publisher. At most one cycle
  // of latency between the edge and the drain.
  std::vector<FaultEvent> DrainPendingEvents();

private:
  static std::string name_;
  static std::unordered_map<std::string, DiagnosticInfo> diagnostic_map_;
  std::set<std::string> prev_active_codes_;
  std::vector<FaultEvent> pending_events_;
  Diagnostic() = default;
  BASE_DECLARE_SINGLETON(Diagnostic)
};
} // namespace util
} // namespace common
} // namespace planning