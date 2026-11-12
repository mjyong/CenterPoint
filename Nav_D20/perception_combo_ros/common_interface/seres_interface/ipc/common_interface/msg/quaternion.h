#ifndef _APPLICATION_COMMON_QUATERNION_H_
#define _APPLICATION_COMMON_QUATERNION_H_

#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Quaternion {
        // X component of the quaternion
        double x;
        // Y component of the quaternion
        double y;
        // Z component of the quaternion
        double z;
        // W component of the quaternion
        double w;
    };

} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_QUATERNION_H_
