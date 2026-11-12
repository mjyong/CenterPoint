#ifndef D10_1230_POINT_CLOUD2_H
#define D10_1230_POINT_CLOUD2_H

#include "common_interface/msg/header.h"
#include "utils/seres_vector.h"
#include "common_interface/msg/point_field.h"

#pragma pack(4)

namespace seres_ipc {
namespace rte_interface {
namespace msg {    
    struct PointCloud2 {
        seres_ipc::common_interface::msg::Header header;
        uint32_t height;
        uint32_t width;
        bool is_dense;
        bool is_bigendian;
        uint32_t point_step;
        uint32_t row_step;
        Seres::vector<uint8_t, 100000> data;
        Seres::vector<common_interface::msg::PointField, 10> fields;
    };
} // namespace msg
} // namespace rte_interface
} // namespace seres_ipc

#pragma pack()

#endif