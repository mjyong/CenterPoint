#ifndef D10_APPLICATION_ACTUATORMODULE_MOTORSTATE_H_
#define D10_APPLICATION_ACTUATORMODULE_MOTORSTATE_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace actuator_module_msg {
namespace msg {
    /**
     * @struct MotorState
     * @brief 电机状态信息结构体，包含电机控制模式、关节反馈信息、温度、丢包及错误标志位等信息
     */
    struct MotorState {
        uint8_t mode;  ///< 电机控制模式; 工作模式-> 0x01,停止模式-> 0x00
        float q;  ///< 关节反馈位置信息
        float dq;  ///< 关节反馈速度
        float ddq;  ///< 关节反馈加速度
        float tau_est;  ///< 关节反馈力矩
        float q_raw;  ///< 暂未不用
        float dq_raw;  ///< 暂未不用
        float ddq_raw;  ///< 暂未不用
        int8_t temperature;  ///< 电机温度信息
        uint32_t lost;  ///< 电机丢包信息
        uint32_t reserve[2];  ///< 电机错误标志位（高1字节）：
                              ///< 0x01 -过流
                              ///< 0x02 -过压
                              ///< 0x04 -驱动过热
                              ///< 0x08 -母线欠压
                              ///< 0x10 -绕组过热
                              ///< 0x20 -编码器异常
    };
} // namespace msg
} // namespace actuator_module_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_ACTUATORMODULE_MOTORSTATE_H_
