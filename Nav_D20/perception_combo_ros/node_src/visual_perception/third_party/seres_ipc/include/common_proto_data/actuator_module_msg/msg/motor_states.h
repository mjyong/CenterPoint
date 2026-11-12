#ifndef D10_APPLICATION_ACTUATORMODULE_MOTORSTATES_H_
#define D10_APPLICATION_ACTUATORMODULE_MOTORSTATES_H_

#include <cstdint>

#include "common_msg/msg/header.h"
#include "motor_state.h"

#pragma pack(1)

namespace seres_ipc {
namespace actuator_module_msg {
namespace msg {
    /**
     * @struct MotorStates
     * @brief 电机状态集合消息，包含消息头和多个电机状态信息
     */
    struct MotorStates {
        common_msg::msg::Header header; ///< 消息头
        MotorState states[12]; ///< 电机状态数组
    };
}
}
} // namespace seres_ipc
#pragma pack()
#endif // D10_APPLICATION_ACTUATORMODULE_MOTORSTATES_H_
