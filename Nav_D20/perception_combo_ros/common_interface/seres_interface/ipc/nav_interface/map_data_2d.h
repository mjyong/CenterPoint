#ifndef D10_1230_2D_MAP_DATA_H_
#define D10_1230_2D_MAP_DATA_H_

#include "utils/seres_vector.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    // 占用值数组（行优先存储）：
    // -1：未知区域（未探索）
    // 0：自由空间（无障碍）
    // 100：障碍物
    // 0-100 之间：占用概率值（值越大表示障碍物概率越高）
    struct MapData2D {
        Seres::vector<uint8_t, 100000> data;
    };
}
}
}
#pragma pack()
#endif