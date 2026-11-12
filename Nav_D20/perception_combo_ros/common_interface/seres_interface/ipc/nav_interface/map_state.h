#ifndef D10_1230_MAP_STATE_H
#define D10_1230_MAP_STATE_H

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum MapStateStates{
        LOCALIZATION_MAP_MATCHING_SCORE = 0,     // 地图匹配得分太低
        LOCALIZATION_MAP_MISMATCH,           // 地图误匹配
        MAPPING_ODOMETRY_OPTIMIZATION,       // 优化不收敛
        MAPPING_ODOMETRY_MATCH_POINTS,       // 匹配点数太少
        MAPPING_ODOMETRY_MATCH_SCORE,        // 匹配得分低
        MAPPING_LOOP_CLOSURE,                // 回环异常
        MAPPING_BACKEND_OPTIMIZATION        // 优化不收敛
    };
    struct MapState {
        common_interface::msg::Header header;
        MapStateStates state;
    };
}
}
}
#pragma pack()

#endif
