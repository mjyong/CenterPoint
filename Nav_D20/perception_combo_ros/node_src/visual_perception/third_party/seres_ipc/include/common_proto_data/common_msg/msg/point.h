#ifndef D10_APPLICATION_COMMON_POINT_H_
#define D10_APPLICATION_COMMON_POINT_H_
#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
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
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_POINT_H_
