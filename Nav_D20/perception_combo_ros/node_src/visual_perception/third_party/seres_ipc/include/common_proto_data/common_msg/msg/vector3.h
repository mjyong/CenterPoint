#ifndef D10_APPLICATION_COMMON_MSG_VECTOR3_H_
#define D10_APPLICATION_COMMON_MSG_VECTOR3_H_

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct Vector3 {
    // X - component of the 3D vector
    double x;
    // Y - component of the 3D vector
    double y;
    // Z - component of the 3D vector
    double z;
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_MSG_VECTOR3_H_
