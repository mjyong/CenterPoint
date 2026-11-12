#ifndef D10_1230_NAV_PTZ_ANGLE_CTRL_H_
#define D10_1230_NAV_PTZ_ANGLE_CTRL_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {

    enum Angle {
        LIFT_MOVE = 1, //左平移
        RIGHT_MOVE = 2, //右平移
        ON_MOVE = 3, //俯仰上移
        BOTTOM_MOVE = 4  //俯仰下移
    };

struct NavPtzAngleCtrl {
    common_interface::msg::Header header;

    Angle angle;
};
}
}
}

#pragma pack()
#endif
