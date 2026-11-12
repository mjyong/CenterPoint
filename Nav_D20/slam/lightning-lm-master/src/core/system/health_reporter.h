//
// 健康上报器: 定位模块通过 rhr_interface 向 rhr 模块发布告警(AlarmInfo)与事件(EventInfo)
// 告警/事件的 ID、名称、等级定义见《事件和警告.xlsx》
//
// topic: /rhr/topic/alarm_info (告警)  /rhr/topic/event_info (事件)
// QoS: QoS(50).reliable()
//

#ifndef LIGHTNING_HEALTH_REPORTER_H
#define LIGHTNING_HEALTH_REPORTER_H

#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

#include "rhr_interface/msg/alarm_info.hpp"
#include "rhr_interface/msg/event_info.hpp"

namespace lightning {

class HealthReporter {
   public:
    using AlarmInfo = rhr_interface::msg::AlarmInfo;
    using EventInfo = rhr_interface::msg::EventInfo;

    // ===== 定位导航: 事件ID =====
    static constexpr uint32_t EVENT_LOC_LOST = 0x04040001;        // 定位丢失
    static constexpr uint32_t EVENT_RELOC_SUCCESS = 0x04040002;   // 重定位成功
    static constexpr uint32_t EVENT_MAP_MATCH_FAIL = 0x04040008;  // 地图匹配失败
    static constexpr uint32_t EVENT_RELOC_FAIL = 0x0404000D;      // 重定位失败

    // ===== 告警ID =====
    static constexpr uint32_t ALARM_MAP_CORRUPT = 0x04040003;       // 地图数据损坏
    static constexpr uint32_t ALARM_IMU_FRAME_RATE = 0x011A0102;    // 上装IMU帧率异常(NN=01)
    static constexpr uint32_t ALARM_LIDAR_CLOUD_DROP = 0x01033504;  // 点云数据骤降(35=导航激光雷达)
    static constexpr uint32_t ALARM_LIDAR_JITTER = 0x01033508;      // 帧间隔抖动(35=导航激光雷达)

    explicit HealthReporter(const rclcpp::Node::SharedPtr& node) {
        rclcpp::QoS qos(50);  // 协议要求 QoS(50).reliable()
        qos.reliable();
        alarm_pub_ = node->create_publisher<AlarmInfo>("/rhr/topic/alarm_info", qos);
        event_pub_ = node->create_publisher<EventInfo>("/rhr/topic/event_info", qos);
    }

    /// 发布一次性事件(state=EVENT_ON)
    void PublishEvent(uint32_t id, uint8_t level, const std::string& name, const std::string& comment) {
        EventInfo msg;
        msg.id = id;
        msg.time = NowMs();
        msg.state = EventInfo::EVENT_ON;
        msg.level = level;
        msg.comment = comment;
        msg.name = name;
        event_pub_->publish(msg);
    }

    /// 设置告警状态(带去重): 仅在告警状态发生变化时发布 ALARM_ON / ALARM_OFF
    void SetAlarm(uint32_t id, bool active, uint8_t level, const std::string& name,
                  const std::string& comment) {
        std::lock_guard<std::mutex> lock(mutex_);
        bool& cur = alarm_active_[id];
        if (cur == active) {
            return;  // 状态未变化, 不重复发布
        }
        cur = active;
        AlarmInfo msg;
        msg.id = id;
        msg.time = NowMs();
        msg.state = active ? AlarmInfo::ALARM_ON : AlarmInfo::ALARM_OFF;
        msg.level = level;
        msg.para = 0;
        msg.comment = comment;
        msg.name = name;
        alarm_pub_->publish(msg);
    }

   private:
    static uint64_t NowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    rclcpp::Publisher<AlarmInfo>::SharedPtr alarm_pub_;
    rclcpp::Publisher<EventInfo>::SharedPtr event_pub_;
    std::mutex mutex_;                          // 保护告警去重状态
    std::unordered_map<uint32_t, bool> alarm_active_;  // id -> 当前是否处于告警中
};

}  // namespace lightning

#endif  // LIGHTNING_HEALTH_REPORTER_H
