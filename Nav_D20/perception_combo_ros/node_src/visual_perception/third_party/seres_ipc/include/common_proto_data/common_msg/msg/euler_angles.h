#ifndef D10_APPLICATION_COMMON_EULERANGLES_H_
#define D10_APPLICATION_COMMON_EULERANGLES_H_

#include <cstdint>
#include "quaternion.h"

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct EulerAngles {
    // Yaw angle
    float yaw;
    // Pitch angle
    float pitch;
    // Roll angle
    float roll;
    // Corresponding quaternion
    Quaternion quaternion;
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_EULERANGLES_H_
