#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rhr_interface/msg/alarm_info.hpp>
#include <rhr_interface/msg/event_info.hpp>

#include "src/common/util/diagnostic.h"

namespace planning {
namespace communication {

/**
 * @brief Publishes edge-detected fault events as rhr_interface messages.
 *
 * Diagnostic (a std::set of active fault codes refilled every cycle) detects
 * raising/clearing edges in its periodic clear(). This bridge drains those
 * events and forwards each one as a single AlarmInfo or EventInfo message,
 * selected by the fault_info.csv `type` column:
 *   type == "alarm" -> /rhr/topic/alarm_info (rhr_interface/AlarmInfo)
 *   type == "event" -> /rhr/topic/event_info (rhr_interface/EventInfo)
 */
class DiagnosticBridge {
public:
  explicit DiagnosticBridge(const rclcpp::Node::SharedPtr &node);

  // Drains pending fault events from Diagnostic and publishes one message
  // per event. Called every communication cycle by RosBridge.
  void Publish();

private:
  rclcpp::Publisher<rhr_interface::msg::AlarmInfo>::SharedPtr pub_alarm_;
  rclcpp::Publisher<rhr_interface::msg::EventInfo>::SharedPtr pub_event_;
};

} // namespace communication
} // namespace planning
