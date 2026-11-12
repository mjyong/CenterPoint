#ifndef D10_1230_NAV_PLANNER_MOTION_H_
#define D10_1230_NAV_PLANNER_MOTION_H_

#include "common_interface/msg/header.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct NavPlannerMotion {
        common_interface::msg::Header header;
        float lx;
        float rx;
        float ry;
        float ly;
        uint16_t keys;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif