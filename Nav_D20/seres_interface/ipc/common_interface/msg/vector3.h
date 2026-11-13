#ifndef _APPLICATION_COMMON_MSG_VECTOR3_H_
#define _APPLICATION_COMMON_MSG_VECTOR3_H_

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
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
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_MSG_VECTOR3_H_
