#ifndef D10_APPLICATION_VISUAL_PERCEPTION_OBJECT_TYPE_H_
#define D10_APPLICATION_VISUAL_PERCEPTION_OBJECT_TYPE_H_
#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace visual_perception_msg {
namespace msg {
struct ObjectType {
    // 枚举字段
    uint8_t object_type;

    // 定义枚举值常量
    static constexpr uint8_t UNKNOWN = 0;
    static constexpr uint8_t UNKNOWN_MOVABLE = 1;
    static constexpr uint8_t UNKNOWN_UNMOVABLE = 2;
    static constexpr uint8_t PEDESTRIAN = 3;
    static constexpr uint8_t BICYCLE = 4;
    static constexpr uint8_t VEHICLE = 5;
    static constexpr uint8_t TRAFFIC_SIGN = 6;
};
} // namespace msg
} // namespace visual_perception_msg
} // namespace seres_ipc

#pragma pack()
#endif
