/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-05-18 10:58:09
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-07-23 10:14:40
 */

#pragma once

#include "src/common/util/macros.h"
#include "src/common/util/time_keeper.h"
#ifdef ENABLE_ROS
#include <plotjuggler_msgs/msg/data_points.hpp>
#include <plotjuggler_msgs/msg/dictionary.hpp>
#endif
#include <random>
#include <unordered_map>
namespace planning {
namespace common {
namespace util {

class DebugDatas : public std::unordered_map<std::string, double> {
public:
#ifdef ENABLE_ROS
  static void ConvertToRosMsg(plotjuggler_msgs::msg::Dictionary &dictionary,
                              plotjuggler_msgs::msg::DataPoints &data_points);
  static std::string ConvertToString(std::string header_name = "timestamp");
#endif

private:
  static uint32_t uuid_;
  DebugDatas() = default;
  BASE_DECLARE_SINGLETON(DebugDatas)
};
} // namespace util
} // namespace common
} // namespace planning