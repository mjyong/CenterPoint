#ifndef _APPLICATION_COMMON_POINT_H_
#define _APPLICATION_COMMON_POINT_H_
#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Point {
        // X-coordinate of the point
        double x;
        // Y-coordinate of the point
        double y;
        // Z-coordinate of the point
        double z;
    };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_POINT_H_
