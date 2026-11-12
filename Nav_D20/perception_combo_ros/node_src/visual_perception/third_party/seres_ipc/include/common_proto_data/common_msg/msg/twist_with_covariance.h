#ifndef D10_APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_
#define D10_APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_

#include "vector3.h"

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct TwistWithCovariance {
    // 角速度
    Vector3 angular_velocity;
    // 线速度
    Vector3 linear_velocity;
    // 协方差
    double covariance[36];
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_
