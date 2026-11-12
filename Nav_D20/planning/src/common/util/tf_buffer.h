#pragma once
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/common/util/macros.h"
#include <unordered_map>
#ifdef ENABLE_ROS
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#endif

namespace planning {
namespace common {
namespace util {
class TfBuffer {
public:
  static void add(const TransformStamped &tf);

  static bool get(const std::string &target_frame,
                  const std::string &source_frame, Transform3d &result,
                  bool timeout = false); // TODO: timeout true
#ifdef ENABLE_ROS
  static std::vector<geometry_msgs::msg::TransformStamped> toTfMessage();
#endif

private:
  static std::unordered_map<std::string, TransformStamped> tf_buffer_;
  static std::mutex mutex_;
  TfBuffer() = default;
  BASE_DECLARE_SINGLETON(TfBuffer)
};
} // namespace util
} // namespace common
} // namespace planning