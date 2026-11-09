/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-12-12 15:55:34
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-13 10:05:06
 */
#include "src/common/util/diagnostic.h"
namespace planning {
namespace common {
namespace util {
std::string Diagnostic::name_;
std::unordered_map<std::string, Diagnostic::FaultInfo> Diagnostic::fault_map_;

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
    if (row.size() != 7) {
      AERROR << "Error: Invalid row size: " << row.size();
      return false;
    }
    FaultInfo info;
    info.index = row[0];
    info.code = row[1];
    info.module = row[2];
    info.fault_type = row[3];
    info.fault_level = row[4];
    info.fault_index = row[5];
    info.fault_describe = row[6];
    fault_map_[info.code] = info;
  }

  file.close();
  return true;
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
      kv.value = fault_map_[*itr].fault_describe;
      msg.status[0].values.push_back(kv);
    }
  }
}
#endif
} // namespace util
} // namespace common
} // namespace planning