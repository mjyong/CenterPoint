#ifndef D10_APPLICATION_COMMON_POSE_H_
#define D10_APPLICATION_COMMON_POSE_H_

#include "point.h"
#include "quaternion.h"

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct Pose {
    // Position of the pose
    Point position;
    // Orientation of the pose represented as a quaternion
    Quaternion orientation;
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_POSE_H_
