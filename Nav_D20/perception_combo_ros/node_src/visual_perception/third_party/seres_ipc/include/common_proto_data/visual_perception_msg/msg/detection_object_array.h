#ifndef D10_APPLICATION_VISUAL_PERCEPTION_DETECTION_OBJECT_ARRAY_H_
#define D10_APPLICATION_VISUAL_PERCEPTION_DETECTION_OBJECT_ARRAY_H_

#include <cstdint>
#include <vector>
#include "common_msg/msg/header.h"
#include "detection_object.h"

#pragma pack(1)

namespace seres_ipc {
namespace visual_perception_msg {
namespace msg {
struct DetectionObjectArray {
    common_msg::msg::Header header;
    DetectionObject det_obj_arr[64];
    uint8_t detect_type;

    // 定义枚举值常量
    static constexpr uint8_t DT_2D = 0;
    static constexpr uint8_t DT_3D = 1;
};
} // namespace msg
} // namespace visual_perception_msg
} // namespace seres_ipc

#pragma pack()
#endif
