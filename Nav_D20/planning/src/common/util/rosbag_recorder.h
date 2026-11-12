#pragma once
#include "src/common/util/log.h"
#include "src/common/util/macros.h"
#include <fstream>
#include <iostream>

#include "config_helper.pb.h"
#include "src/common/util/config_helper.h"
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>

namespace planning {
namespace common {
namespace util {
using config::BagRecordConf;
class RosbagRecorder {
public:
  template <typename MessageT>
  static void add(const MessageT &message, const std::string &name,
                  const std::string &type) {
    if(!conf_.enable()){
      return;
    }
    rclcpp::SerializedMessage serialized_msg;
    rclcpp::Serialization<MessageT> serialization;
    serialization.serialize_message(&message, &serialized_msg);

    rosbag2_storage::TopicMetadata tm;
    tm.name = name;
    tm.type = type;
    tm.serialization_format = "cdr";
    writer_->create_topic(tm);

    auto bag_message =
        std::make_shared<rosbag2_storage::SerializedBagMessage>();
#ifndef ROS2_JAZZY
    auto ret = rcutils_system_time_now(&bag_message->time_stamp);
#else
    auto ret = rcutils_system_time_now(&bag_message->recv_timestamp);
    bag_message->send_timestamp = bag_message->recv_timestamp;
#endif
    if (ret != RCL_RET_OK) {
      RCLCPP_ERROR(rclcpp::get_logger("saveToBag"),
                   "couldn't assign time rosbag message");
    }

    bag_message->topic_name = tm.name;
    bag_message->serialized_data = std::shared_ptr<rcutils_uint8_array_t>(
        &serialized_msg.get_rcl_serialized_message(),
        [](rcutils_uint8_array_t *) {});
    writer_->write(bag_message);
    RestrictCheck();
  }
  ~RosbagRecorder();

private:
  RosbagRecorder();
  static void RestrictCheck();
  static std::unique_ptr<rosbag2_cpp::Writer> writer_;
  static double duration_;
  static double prev_check_time_;
  static std::string bag_dir_;
  static std::string bag_file_name_;
  static BagRecordConf conf_;
  BASE_DECLARE_SINGLETON(RosbagRecorder)
};
} // namespace util
} // namespace common
} // namespace planning