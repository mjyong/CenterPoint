#ifndef D10_APPLICATION_ACTUATORMODULE_MOTORCMDS_H_
#define D10_APPLICATION_ACTUATORMODULE_MOTORCMDS_H_

#include <cstdint>
#include <vector>

#include "common_msg/msg/header.h"
#include "motor_cmd.h"

#pragma pack(1)

namespace seres_ipc {
namespace actuator_module_msg {
namespace msg {
    /**
     * @struct MotorCmds
     * @brief 表示一组电机控制指令，包含消息头和多个电机控制指令条目。
     */
    struct MotorCmds {
        common_msg::msg::Header header; ///< 消息头
        MotorCmd cmds[12]; ///< 电机控制指令数组
    };
} // namespace msg
} // namespace actuator_module_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_ACTUATORMODULE_MOTORCMDS_H_
