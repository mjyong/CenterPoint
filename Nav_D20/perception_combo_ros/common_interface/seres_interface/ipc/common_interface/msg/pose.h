#ifndef _APPLICATION_COMMON_POSE_H_
#define _APPLICATION_COMMON_POSE_H_

#include "point.h"
#include "quaternion.h"

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Pose {
        // Position of the pose
        Point position;
        // Orientation of the pose represented as a quaternion
        Quaternion orientation;
    };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_POSE_H_
