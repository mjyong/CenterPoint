#ifndef D10_1230_VISUAL_PERCEPTION_STATE_H
#define D10_1230_VISUAL_PERCEPTION_STATE_H

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct VisualPerceptionState {
        common_interface::msg::Header header;
        uint8_t state;
    };
}
}
}

#pragma pack()
#endif
