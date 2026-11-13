#ifndef _APPLICATION_COMMON_EULERANGLES_H_
#define _APPLICATION_COMMON_EULERANGLES_H_

#include "quaternion.h"
#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
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
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_EULERANGLES_H_
