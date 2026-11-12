#ifndef D10_1230_NAV_REMOTE_CONTROLLER_REQUEST_H_
#define D10_1230_NAV_REMOTE_CONTROLLER_REQUEST_H_

#pragma pack(4)

#include "common_interface/msg/header.h"

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum request_type {
        REMOTE_DISCONNECT = 0,  // 遥控器退出
        REMOTE_CONNECT    = 1   // 遥控器接入
    };

    struct NavRemoteControllerRequest {
        common_interface::msg::Header header;
        request_type request;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif
