#ifndef D10_APPLICATION_ODOMETRY_ODOMETRY_H_
#define D10_APPLICATION_ODOMETRY_ODOMETRY_H_

#include "common_msg/msg/header.h"
#include "common_msg/msg/pose_with_covariance.h"
#include "common_msg/msg/twist_with_covariance.h"

#include <string>

#pragma pack(1)

namespace seres_ipc {
namespace odometry_msg {
namespace msg {
struct Odometry {
    // Message header
    common_msg::msg::Header header;
    // Child frame ID
    char child_frame_id[64];
    // Pose with covariance
    common_msg::msg::PoseWithCovariance pose;
    // Twist with covariance
    common_msg::msg::TwistWithCovariance twist;
};
} // namespace msg
} // namespace odometry_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_ODOMETRY_ODOMETRY_H_
