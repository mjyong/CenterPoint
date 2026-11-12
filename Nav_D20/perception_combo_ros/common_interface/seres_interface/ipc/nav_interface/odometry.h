#ifndef D10_1230_ODOMETRY_H_
#define D10_1230_ODOMETRY_H_

#include "common_interface/msg/header.h"
#include "common_interface/msg/pose_with_covariance.h"
#include "common_interface/msg/twist_with_covariance.h"
#include "utils/seres_string.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct Odometry {
        common_interface::msg::Header header;
        Seres::string<64> frame_id;
        common_interface::msg::PoseWithCovariance pose;
        common_interface::msg::TwistWithCovariance twist;
    };
}
}
}
#pragma pack()

#endif
