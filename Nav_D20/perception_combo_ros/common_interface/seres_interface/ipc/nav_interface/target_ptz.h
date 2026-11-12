#ifndef D10_1230_TARGET_PTZ_H_
#define D10_1230_TARGET_PTZ_H_

#include "common_interface/msg/header.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct TargetPtz {
        common_interface::msg::Header header;
        double p;
        double t;
        double z;
        bool ptz_in_place;
        uint8_t poi_id;
        uint8_t task_type;  //1 路灯检测（RGB图像）;2 建筑物外立面检测（RGB图像）;3 行人行为规范（RGB图像）;4 高温检测（红外图像）
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif