#ifndef D10_1230_MAP_META_DATA_H_
#define D10_1230_MAP_META_DATA_H_

#include "common_interface/msg/header.h"
#include "utils/seres_vector.h"
#include "map_data_2d.h"
#include "common_interface/msg/pose.h"
#include "common_interface/msg/time.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    
   //地图类型 
    enum MapType {
        GLOBAL_MAP = 1,
        MAP_UPDATE = 2,
        LOCAL_MAP = 3,
        EXPLORE_MAP = 4
    };

    struct MapMetaData {
        common_interface::msg::Header header;
        common_interface::msg::Time time;
        float resolution;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        common_interface::msg::Pose origin;
        Seres::vector<MapData2D, 100> map_data;
        MapType type;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif
