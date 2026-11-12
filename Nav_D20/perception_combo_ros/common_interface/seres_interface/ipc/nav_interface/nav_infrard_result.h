#ifndef D10_1230_NAV_INFRARD_RESULT_H_
#define D10_1230_NAV_INFRARD_RESULT_H_

#include "common_interface/msg/header.h"
#include "utils/seres_string.h"
#include "utils/seres_vector.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct NavInfraredResult {
        common_interface::msg::Header header;
        bool is_specification;
        uint32_t height;
        uint32_t width;
        Seres::string<8> encoding;
        Seres::vector<uint8_t, 6320800> data;
        uint8_t is_bigendian;
        uint32_t step;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
