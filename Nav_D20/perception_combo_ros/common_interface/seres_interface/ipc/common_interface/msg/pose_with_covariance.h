#ifndef _APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_
#define _APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_

#include "pose.h"
#include "utils/seres_vector.h"

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct PoseWithCovariance {
        // Pose information
        Pose pose;
        // Covariance matrix, stored as a 36-element array
        Seres::vector<double, 36> covariance;
    };

} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_
