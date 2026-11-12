#ifndef D10_TRAVERSABILITY_GRID_H
#define D10_TRAVERSABILITY_GRID_H

#include "common_interface/msg/header.h"
#include "cost_map.h"
#include "common_interface/msg/pose.h"
#include "common_interface/msg/map_type.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct TraversabilityGrid {
        common_interface::msg::Header header;
        float resolution;
        uint32_t width;
        uint32_t height;
        common_interface::msg::Pose origin;
        CostMap costmap;
        common_interface::msg::MapType type;

    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()

#endif