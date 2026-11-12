#ifndef D10_APPLICATION_PREDICTION_TRAJECTORYPOINT_H_
#define D10_APPLICATION_PREDICTION_TRAJECTORYPOINT_H_

#include <cstdint>
#include "common_msg/msg/point.h"
#include "common_msg/msg/euler_angles.h"
#include "actuator_module_msg/msg/motion_state.h"
#include "common_msg/msg/time.h"

#pragma pack(1)

namespace seres_ipc {
namespace prediction_msg {
namespace msg {
struct TrajectoryPoint {
    // Position
    common_msg::msg::Point position;
    // Orientation
    common_msg::msg::EulerAngles orientation;
    // Motion state
    actuator_module_msg::msg::MotionState motion_state;
    // Time offset from the start of the trajectory, builtin_interfaces/Time
    common_msg::msg::Time time_from_start;
};
} //namespace msg
} // namespace prediction_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_PREDICTION_TRAJECTORYPOINT_H_
