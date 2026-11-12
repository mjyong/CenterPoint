#ifndef D10_APPLICATION_ACTUATORMODULE_MOTIONSTATE_H_
#define D10_APPLICATION_ACTUATORMODULE_MOTIONSTATE_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace actuator_module_msg {
namespace msg {
    /**
     * @struct MotionState
     * @brief 表示执行器模块的运动状态
     */
    struct MotionState {
        double VX; ///< X 方向速度 (单位: m/s)
        double VY; ///< Y 方向速度
        double AX; ///< X 方向加速度 (单位: m/s²)
        double AY; ///< Y 方向加速度
        double JERK; ///< 加加速度 (单位: m/s³)
        double YAW_RATE; ///< 偏航角速度 (单位: rad/s)
    };
} // namespace msg
} // namespace actuator_module_msg
} // namespace seres_ipc
#pragma pack()
#endif // D10_APPLICATION_ACTUATORMODULE_MOTIONSTATE_H_


