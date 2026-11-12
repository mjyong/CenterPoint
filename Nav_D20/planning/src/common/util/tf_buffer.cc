/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-05-20 10:28:07
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2024-07-23 10:14:35
 */
#include "src/common/util/tf_buffer.h"
#include "src/common/util/time_keeper.h"
namespace planning {
namespace common {
namespace util {

std::unordered_map<std::string, TransformStamped> TfBuffer::tf_buffer_;
std::unordered_map<std::string, std::string> TfBuffer::published_;
std::mutex TfBuffer::mutex_;

void TfBuffer::add(const TransformStamped &tf, bool publish) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = tf.frame_id + tf.child_frame_id;
  tf_buffer_[key] = tf;
  if (publish) {
    // /tf used to carry every entry, inverses included (map -> base_link and base_link -> map, and map ->
    // ptz_visible next to base_link -> ptz_visible): frames with several parents and cycles, which tf2, RViz
    // and Foxglove cannot resolve.
    auto it = published_.find(tf.child_frame_id);
    if (it == published_.end()) {
      published_.emplace(tf.child_frame_id, key);
    } else if (it->second != key) {
      static std::unordered_map<std::string, bool> warned;
      if (!warned[tf.child_frame_id]) {
        warned[tf.child_frame_id] = true;
        AWARN << "TfBuffer: " << tf.child_frame_id << " already has the parent "
              << tf_buffer_.at(it->second).frame_id << " on /tf; "
              << tf.frame_id << " -> " << tf.child_frame_id
              << " is kept for lookups only";
      }
    }
  }

  TransformStamped inverse_tf;
  inverse_tf.timestamp = tf.timestamp;
  inverse_tf.frame_id = tf.child_frame_id;
  inverse_tf.child_frame_id = tf.frame_id;
  inverse_tf.transform = tf.transform.inverse();
  std::string inverse_key = inverse_tf.frame_id + inverse_tf.child_frame_id;
  tf_buffer_[inverse_key] = inverse_tf;
}

bool TfBuffer::get(const std::string &source_frame,
                   const std::string &target_frame, Transform3d &result,
                   bool timeout) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key = source_frame + target_frame;
  if (tf_buffer_.count(key) == 0)
    return false;
  const auto &tf = tf_buffer_.at(key);
  if (timeout && (fabs(TimeKeeper::TimeNowSec() - tf.timestamp) > 0.1)) {
    return false;
  }
  result = tf.transform;
  return true;
}
#ifdef ENABLE_ROS
std::vector<geometry_msgs::msg::TransformStamped> TfBuffer::toTfMessage() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<geometry_msgs::msg::TransformStamped> tf_msg;
  for (const auto &edge : published_) {
    const auto &tf = *tf_buffer_.find(edge.second);
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = tf.second.frame_id;
    int64_t total_nanoseconds = static_cast<int64_t>(tf.second.timestamp * 1e9);
    msg.header.stamp.sec = total_nanoseconds / 1000000000;
    msg.header.stamp.nanosec = total_nanoseconds % 1000000000;
    msg.child_frame_id = tf.second.child_frame_id;
    msg.transform.translation.x = tf.second.transform.translation().x();
    msg.transform.translation.y = tf.second.transform.translation().y();
    msg.transform.translation.z = tf.second.transform.translation().z();
    Eigen::Quaterniond quat(tf.second.transform.rotation());
    msg.transform.rotation.x = quat.x();
    msg.transform.rotation.y = quat.y();
    msg.transform.rotation.z = quat.z();
    msg.transform.rotation.w = quat.w();
    tf_msg.emplace_back(msg);
  }
  return tf_msg;
}
#endif
} // namespace util
} // namespace common
} // namespace planning