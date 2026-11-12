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
  // Stores tf and its inverse for get(). With publish (the default) tf is also an edge of the tree that
  // toTfMessage() sends on /tf; a derived shortcut (e.g. map -> ptz_visible next to base_link -> ptz_visible)
  // passes publish = false. A child frame keeps the first parent it was published with (2026-11-09).
  static void add(const TransformStamped &tf, bool publish = true);

  static bool get(const std::string &target_frame,
                  const std::string &source_frame, Transform3d &result,
                  bool timeout = false); // TODO: timeout true
#ifdef ENABLE_ROS
  // One transform per published child frame (parent -> child), never the inverses: /tf must be a tree.
  static std::vector<geometry_msgs::msg::TransformStamped> toTfMessage();
#endif

private:
  static std::unordered_map<std::string, TransformStamped> tf_buffer_;
  // child frame -> tf_buffer_ key of the edge published for it
  static std::unordered_map<std::string, std::string> published_;
  static std::mutex mutex_;
  TfBuffer() = default;
  BASE_DECLARE_SINGLETON(TfBuffer)
};
} // namespace util
} // namespace common
} // namespace planning