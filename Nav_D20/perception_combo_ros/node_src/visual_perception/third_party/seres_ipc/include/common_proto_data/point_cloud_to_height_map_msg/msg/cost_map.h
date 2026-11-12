#ifndef D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_COST_MAP_H_
#define D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_COST_MAP_H_

#include <cstdint>
#include "common_msg/msg/header.h"
#include "common_msg/msg/map_data_2d.h"
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace point_cloud_to_height_map_msg {
namespace msg {
struct CostMap {
    // Message header
    common_msg::msg::Header header;
    // Array of map cost data
    common_msg::msg::MapData2D mapcost[1024];
};
}
}
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_POINT_CLOUD_TO_HEIGHT_MAP_COST_MAP_H_
