#ifndef D10_APPLICATION_MAP_MODULE_MAP_META_DATA_H_
#define D10_APPLICATION_MAP_MODULE_MAP_META_DATA_H_

#include <vector>
#include "common_msg/msg/header.h"
#include "common_msg/msg/pose.h"
#include "common_msg/msg/map_data_2d.h"
#include "common_msg/msg/map_type.h"
#include "common_msg/msg/time.h"

#pragma pack(1)

namespace seres_ipc {
namespace map_module_msg {
namespace msg {
struct MapMetaData {
    // Message header
    common_msg::msg::Header header;
    // Map load time
    common_msg::msg::Time map_load_time;
    // Map resolution (meters per pixel)
    float resolution;
    // Map width (number of pixels)
    uint32_t width;
    // Map height (number of pixels)
    uint32_t height;
    // Map depth (number of pixels)
    uint32_t depth;
    // Pose of the bottom - left corner of the map in the world coordinate system
    common_msg::msg::Pose origin;
    // Occupancy value array (row - major storage)
    common_msg::msg::MapData2D mapdata[1024];
    // Map type
    common_msg::msg::MapType type;
};
}
}
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_MAP_MODULE_MAP_META_DATA_H_
