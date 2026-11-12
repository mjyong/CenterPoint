#ifndef D10_1230_COST_MAP_H_
#define D10_1230_COST_MAP_H_

#include "utils/seres_vector.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct CostMap {
        Seres::vector<uint8_t, 100> data;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
