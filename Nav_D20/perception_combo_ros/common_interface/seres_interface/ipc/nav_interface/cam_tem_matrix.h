#ifndef D10_1230_CAM_TEM_MATRIX_H_
#define D10_1230_CAM_TEM_MATRIX_H_

#include "common_interface/msg/header.h"
#include "utils/seres_string.h"
#include "utils/seres_vector.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct CamTemMatrix {
        common_interface::msg::Header header;
        uint32_t height;
        uint32_t width;
        bool is_freezedata;
        Seres::vector<float, 1024> data;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
