#ifndef D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_TRAVERSABILITY_GRID_H_
#define D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_TRAVERSABILITY_GRID_H_

#include <cstdint>
#include "common_msg/msg/header.h"
#include "map_module_msg/msg/map_meta_data.h"
#include "cost_map.h"

#pragma pack(1)

namespace seres_ipc {
namespace point_cloud_to_height_map_msg {
namespace msg {
struct TraversabilityGrid {
    // Message header
    common_msg::msg::Header header;
    // Cost map
    CostMap costmap;
    // Map meta data
    map_module_msg::msg::MapMetaData mapdata;
};
} // namespace msg
} // namespace point_cloud_to_height_map_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_TRAVERSABILITY_GRID_H_
