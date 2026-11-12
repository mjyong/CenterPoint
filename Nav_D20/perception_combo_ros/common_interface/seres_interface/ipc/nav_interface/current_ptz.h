#ifndef D10_1230_CURRENT_PTZ_H
#define D10_1230_CURRENT_PTZ_H

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct CurrentPtz {
        common_interface::msg::Header header;
        double p;
        double t;
        double z;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()
#endif 
