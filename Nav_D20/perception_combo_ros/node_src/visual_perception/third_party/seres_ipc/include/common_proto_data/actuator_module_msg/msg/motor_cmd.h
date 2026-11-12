#ifndef D10_APPLICATION_ACTUATORMODULE_MOTOR_CMD_H_
#define D10_APPLICATION_ACTUATORMODULE_MOTOR_CMD_H_

#include <stdint.h>

#pragma pack(1)

namespace seres_ipc {
namespace actuator_module_msg {
namespace msg {
    struct MotorCmd {
        uint8_t mode;       ///< 电机控制模式: 0x00(停止模式), 0x01(工作模式)
        // 关节状态反馈
        float q;            ///< 关节反馈位置 (单位: 弧度)
        float dq;           ///< 关节反馈速度 (单位: 弧度/秒)
        float tau;          ///< 关节力矩 (单位: N·m)
        // 阻抗控制参数
        float kp;           ///< 关节刚度系数 (单位: N·m/rad)
        float kd;           ///< 关节阻尼系数 (单位: N·m·s/rad)
        // 保留字段
        uint32_t reserve[3]; ///< 保留位 (预留未来扩展，默认全0)
    };
}
}
}
#pragma pack()
#endif
