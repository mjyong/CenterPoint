#ifndef D10_1230_NAV_PTZ_TASK_CTRL_H_
#define D10_1230_NAV_PTZ_TASK_CTRL_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {

struct NavPtzTakePhotoCtrl {
    common_interface::msg::Header header;

    uint32_t take_photo_id;
};
}
}
}

#pragma pack()
#endif
