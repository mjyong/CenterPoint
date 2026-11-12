#ifndef D10_APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_
#define D10_APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_

#include "pose.h"

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct PoseWithCovariance {
    // Pose information
    Pose pose;
    // Covariance matrix, stored as a 36-element array
    double covariance[36];
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_POSE_WITH_COVARIANCE_H_
