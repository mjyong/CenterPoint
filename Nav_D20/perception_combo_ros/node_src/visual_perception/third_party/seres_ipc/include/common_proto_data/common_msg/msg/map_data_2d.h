#ifndef D10_APPLICATION_COMMON_MAP_DATA_2D_H_
#define D10_APPLICATION_COMMON_MAP_DATA_2D_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
// 待根据实际大小调整值
struct MapData2D {
    // 占用值数组（行优先存储）:
    // -1：未知区域（未探索）;
    // 0：自由空间（无障碍）;
    // 100：障碍物;
    // 0-100 之间：占用概率值（值越大表示障碍物概率越高）
    int8_t data[16106127360];
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_MAP_DATA_2D_H_
