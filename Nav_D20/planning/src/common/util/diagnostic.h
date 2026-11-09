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
#include <vector>

#ifdef ENABLE_ROS
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#endif

namespace planning {
namespace common {
namespace util {
class Diagnostic : public std::set<std::string> {
  struct FaultInfo {
    std::string index;
    std::string code;
    std::string module;
    std::string fault_type;
    std::string fault_level;
    std::string fault_index;
    std::string fault_describe;
  };

public:
#ifdef ENABLE_ROS
  void ConvertToRosMsg(diagnostic_msgs::msg::DiagnosticArray &msg);
#endif
  bool Init(const std::string name, const std::string &filename);

private:
  static std::string name_;
  static std::unordered_map<std::string, FaultInfo> fault_map_;
  Diagnostic() = default;
  BASE_DECLARE_SINGLETON(Diagnostic)
};
} // namespace util
} // namespace common
} // namespace planning