#ifndef D10_APPLICATION_LIDAR_POINT_CLOUD2_H_
#define D10_APPLICATION_LIDAR_POINT_CLOUD2_H_

#include <cstdint>
#include "common_msg/msg/header.h"
#include "point_field.h"

#pragma pack(1)

namespace seres_ipc {
namespace lidar_msg {
namespace msg {
struct PointCloud2 {
    // Message header
    common_msg::msg::Header header;
    // Height of the point cloud matrix
    uint32_t height;
    // Width of the point cloud matrix
    uint32_t width;
    // Whether the data is in big - endian format
    bool is_bigendian;
    // Number of bytes per point
    uint32_t point_step;
    // Number of bytes per row
    uint32_t row_step;
    // Raw point cloud data
    uint8_t data[1024000000000];
    // Whether the data is dense (no invalid points)
    bool is_dense;
    // Description of each point field
    PointField fields[1024000000000];
};
} // namespace msg
} // namespace lidar_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_LIDAR_POINT_CLOUD2_H_
