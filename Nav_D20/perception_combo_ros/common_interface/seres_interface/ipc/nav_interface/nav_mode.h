#ifndef D10_1230_NAV_MODE_H_
#define D10_1230_NAV_MODE_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum MODE {
        IDLE_MODE = 0,          // 空闲模式
        WORK_MODE,       // 作业模式
        FAULT_MODE       // 故障模式
    };
    struct NavMode {
        common_interface::msg::Header header;
        MODE mode;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
