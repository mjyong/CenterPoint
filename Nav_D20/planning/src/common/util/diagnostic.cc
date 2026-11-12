/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-12-12 15:55:34
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-13 10:05:06
 */
#include "src/common/util/diagnostic.h"
#include <chrono>
namespace planning {
namespace common {
namespace util {
std::string Diagnostic::name_;
std::unordered_map<std::string, Diagnostic::DiagnosticInfo> Diagnostic::diagnostic_map_;

bool Diagnostic::Init(const std::string name, const std::string &filename) {
  name_ = name;
  std::ifstream file(filename);
  if (!file.is_open()) {
    AERROR << "Error: Could not open file: " << filename;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<std::string> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, '\t')) {
      row.push_back(cell);
    }
    if (row.size() != 6) {
      AERROR << "Error: Invalid row size: " << row.size();
      return false;
    }
    DiagnosticInfo info;
    info.code = row[0];
    info.module = row[1];
    info.type = row[2];
    info.level = row[3];
    info.name = row[4];
    info.describe = row[5];
    diagnostic_map_[info.code] = info;
  }

  file.close();
  return true;
}

void Diagnostic::clear() {
  // Edge detection at the cycle boundary: diff the active set against the
  // previous cycle. Raising = in current but not in previous; clearing = in
  // previous but not in current. Timestamp granularity is the cycle period.
  const uint64_t now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  for (const auto &code : *this) {
    if (prev_active_codes_.count(code) == 0) {
      auto itr = diagnostic_map_.find(code);
      if (itr == diagnostic_map_.end()) {
        AWARN << "Unknown diagnostic code raised: " << code;
        continue;
      }
      pending_events_.push_back(FaultEvent{itr->second, true, now_ms});
    }
  }
  for (const auto &code : prev_active_codes_) {
    if (count(code) == 0) {
      auto itr = diagnostic_map_.find(code);
      if (itr == diagnostic_map_.end()) {
        continue;
      }
      pending_events_.push_back(FaultEvent{itr->second, false, now_ms});
    }
  }
  prev_active_codes_ = *this;
  std::set<std::string>::clear();
}

std::vector<Diagnostic::FaultEvent> Diagnostic::DrainPendingEvents() {
  std::vector<FaultEvent> events;
  pending_events_.swap(events);
  return events;
}

#ifdef ENABLE_ROS
void Diagnostic::ConvertToRosMsg(diagnostic_msgs::msg::DiagnosticArray &msg) {
  msg.header.stamp = rclcpp::Clock().now();
  msg.header.frame_id = name_;
  if (this->empty()) {
    msg.status.resize(1);
    msg.status[0].name = name_;
    msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    msg.status[0].message = "OK";
  } else {
    msg.status.resize(1);
    msg.status[0].name = name_;
    msg.status[0].level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    msg.status[0].message = "ERROR";
    for (auto itr = this->begin(); itr != this->end(); itr++) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = *itr;
      kv.value = diagnostic_map_[*itr].describe;
      msg.status[0].values.push_back(kv);
    }
  }
}
#endif
} // namespace util
} // namespace common
} // namespace planning