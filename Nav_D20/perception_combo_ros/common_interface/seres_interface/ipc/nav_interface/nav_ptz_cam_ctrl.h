#ifndef D10_1230_NAV_PTZ_CAM_CTRL_H_
#define D10_1230_NAV_PTZ_CAM_CTRL_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {

    enum FocusMode {
        AUTO_FOCUS = 1,  //自动聚焦
        NO_AUTO_FOCUS = 2 //手动聚焦
    };

    enum Focal {
        EXPAND = 1, //扩大
        SHRINK = 2  //缩小
    };

struct NavPtzCamCtrl {
    common_interface::msg::Header header;

    FocusMode focus_mode;
    Focal focal;
    uint32_t cameraId;
};
}
}
}

#pragma pack()
#endif
