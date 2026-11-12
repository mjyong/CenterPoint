#ifndef _APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_
#define _APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_

#include "utils/seres_vector.h"
#include "vector3.h"

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct TwistWithCovariance {
        // 角速度
        Vector3 angular_velocity;
        // 线速度
        Vector3 linear_velocity;
        // 协方差
        Seres::vector<double, 36> covariance;
    };

} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_MSG_TWIST_WITH_COVARIANCE_H_
