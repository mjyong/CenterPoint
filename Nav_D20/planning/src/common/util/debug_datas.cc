/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-05-20 10:28:07
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-07-23 10:14:35
 */
#include "src/common/util/debug_datas.h"

namespace planning {
namespace common {
namespace util {

uint32_t DebugDatas::uuid_ = 121111;
#ifdef ENABLE_ROS
void DebugDatas::ConvertToRosMsg(
    plotjuggler_msgs::msg::Dictionary &dictionary,
    plotjuggler_msgs::msg::DataPoints &data_points) {
  dictionary.names.clear();
  data_points.samples.clear();
  uint32_t index = 0;
  dictionary.dictionary_uuid = uuid_;
  data_points.dictionary_uuid = uuid_;
  const double time_stamp = TimeKeeper::TimeNowSec();
  for (auto it = DebugDatas::Instance()->begin();
       it != DebugDatas::Instance()->end(); ++it) {
    dictionary.names.emplace_back(it->first);
    plotjuggler_msgs::msg::DataPoint data_point;
    data_point.value = it->second;
    data_point.name_index = index++;
    data_point.stamp = time_stamp;
    data_points.samples.emplace_back(data_point);
  }
}

std::string DebugDatas::ConvertToString(std::string header_name) {
  std::string str;
  str = header_name + ": " + std::to_string(TimeKeeper::TimeNowSec()) + "\n";
  for (auto it = DebugDatas::Instance()->begin();
       it != DebugDatas::Instance()->end(); ++it) {
    str += it->first + ": " + std::to_string(it->second) + "\n";
  }
  return str;
}
#endif
} // namespace util
} // namespace common
} // namespace planning