#ifndef D10_APPLICATION_COMMON_QUATERNION_H_
#define D10_APPLICATION_COMMON_QUATERNION_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
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
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_QUATERNION_H_


