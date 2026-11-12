#ifndef D10_APPLICATION_IMU_IMU_H_
#define D10_APPLICATION_IMU_IMU_H_

#include "common_msg/msg/header.h"
#include "common_msg/msg/quaternion.h"
#include "common_msg/msg/vector3.h"

#pragma pack(1)

namespace seres_ipc {
namespace imu_msg {
namespace msg {
struct IMU {
    // Message header
    common_msg::msg::Header header;
    // Orientation
    common_msg::msg::Quaternion orientation;
    // Angular velocity
    common_msg::msg::Vector3 angular_velocity;
    // Linear acceleration
    common_msg::msg::Vector3 linear_acceleration;
};
} // namespace msg
} // namespace imu_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_IMU_IMU_H_
