#include "src/runnable/communication/ros_bridge/diagnostic_bridge.h"

#include <cstdlib>
#include <string>

namespace planning {
namespace communication {

using planning::common::util::Diagnostic;

namespace {
// Alarm/event levels share the same encoding: 1=general, 2=serious, 3=urgent.
uint8_t parseLevel(const std::string &level) {
  const int value = std::atoi(level.c_str());
  if (value < 1 || value > 3) {
    return 1; // general
  }
  return static_cast<uint8_t>(value);
}

// fault_info.csv codes are hex strings like "0x04040005".
bool parseCode(const std::string &code, uint32_t &id) {
  try {
    id = static_cast<uint32_t>(
        std::stoul(code, nullptr, 16));
    return true;
  } catch (...) {
    return false;
  }
}
} // namespace

DiagnosticBridge::DiagnosticBridge(const rclcpp::Node::SharedPtr &node) {
  // Fault transitions must not be dropped: reliable QoS with a small depth.
  pub_alarm_ = node->create_publisher<rhr_interface::msg::AlarmInfo>(
      "/rhr/topic/alarm_info", rclcpp::QoS(10).reliable());
  pub_event_ = node->create_publisher<rhr_interface::msg::EventInfo>(
      "/rhr/topic/event_info", rclcpp::QoS(10).reliable());
}

void DiagnosticBridge::Publish() {
  for (auto &event : Diagnostic::Instance()->DrainPendingEvents()) {
    uint32_t id = 0;
    if (!parseCode(event.info.code, id)) {
      AWARN << "Invalid diagnostic code format: " << event.info.code;
      continue;
    }
    const uint8_t level = parseLevel(event.info.level);
    const uint8_t state_on =
        event.active ? rhr_interface::msg::AlarmInfo::ALARM_ON
                     : rhr_interface::msg::AlarmInfo::ALARM_OFF;

    if (event.info.type == "alarm") {
      rhr_interface::msg::AlarmInfo msg;
      msg.id = id;
      msg.time = event.timestamp;
      msg.state = state_on;
      msg.level = level;
      msg.para = 0;
      msg.comment = event.info.describe;
      msg.name = event.info.name;
      pub_alarm_->publish(msg);
    } else if (event.info.type == "event") {
      rhr_interface::msg::EventInfo msg;
      msg.id = id;
      msg.time = event.timestamp;
      msg.state =
          event.active ? rhr_interface::msg::EventInfo::EVENT_ON
                       : rhr_interface::msg::EventInfo::EVENT_OFF;
      msg.level = level;
      msg.comment = event.info.describe;
      msg.name = event.info.name;
      pub_event_->publish(msg);
    } else {
      AWARN << "Unknown diagnostic type: " << event.info.type
            << " (code: " << event.info.code << ")";
    }
  }
}

} // namespace communication
} // namespace planning
