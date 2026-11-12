#ifndef D10_1230_WHOLE_POINT_CLOUD2_H_
#define D10_1230_WHOLE_POINT_CLOUD2_H_

#include "common_interface/msg/header.h"
#include "common_interface/msg/point_field.h"
#include "utils/seres_vector.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct WholePointCloud2 {
        common_interface::msg::Header header;
        uint32_t height;
        uint32_t width;
        bool is_dense;
        bool is_bigendian;
        uint32_t point_step;
        uint32_t row_step;
        Seres::vector<uint8_t, 10000000> data;
        Seres::vector<uint32_t, 10000> foreground_idx;
        Seres::vector<common_interface::msg::PointField, 10> fields;
    }; 
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif