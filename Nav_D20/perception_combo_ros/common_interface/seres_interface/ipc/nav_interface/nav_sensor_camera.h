#ifndef D10_1230_NAV_SENSOR_CAMERA_H
#define D10_1230_NAV_SENSOR_CAMERA_H

#include "common_interface/msg/header.h"
#include "utils/seres_string.h"
#include "utils/seres_vector.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct NavSensorCamera {
        common_interface::msg::Header header;
        uint32_t height;
        uint32_t width;
        Seres::string<8> encoding;
        Seres::vector<uint8_t, 1024> data;
    };
}
}
}
#pragma pack()

#endif
