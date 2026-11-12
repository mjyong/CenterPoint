#ifndef D10_1230_LINEAR_SCAN_H_
#define D10_1230_LINEAR_SCAN_H_

#include "common_interface/msg/point.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct LinearScan {
        int sensor_id; // 0: visible light, 1: infrared
        common_interface::msg::Point target_start_point;
        common_interface::msg::Point target_stop_point;
        double duration;
    };
}
}
}
#pragma pack()

#endif