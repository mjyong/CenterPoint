#ifndef D10_APPLICATION_TARGET_FOLLLOW_NODE_TARGET_FOLLOW_STATE_H_
#define D10_APPLICATION_TARGET_FOLLLOW_NODE_TARGET_FOLLOW_STATE_H_

#include <cstdint>
#include "common_msg/msg/point.h"
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace target_folllow_node_msg {
namespace msg {
struct TargetFollowState {
    // 消息头
    common_msg::msg::Header header;
    // 目标点
    common_msg::msg::Point dst_point;
    // 跟随状态，枚举值
    uint8_t follow_state;

    // 定义枚举值常量
    static constexpr uint8_t FOLLOWING = 0;
    static constexpr uint8_t PAUSE = 1;
    static constexpr uint8_t STOP = 2;
};
} // namespace msg
} // namespace target_folllow_node_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_TARGET_FOLLLOW_NODE_TARGET_FOLLOW_STATE_H_
