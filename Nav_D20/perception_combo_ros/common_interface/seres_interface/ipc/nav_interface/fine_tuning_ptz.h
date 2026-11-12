#ifndef D10_1230_FINE_TUNING_PTZ_H
#define D10_1230_FINE_TUNING_PTZ_H

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct FineTuningPtz {
        common_interface::msg::Header header;
        double p;
        double t;
        double z;
        bool fine_tuing;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif