#ifndef D10_APPLICATION_VISUAL_PERCEPTION_DETECTION_OBJECT_H_
#define D10_APPLICATION_VISUAL_PERCEPTION_DETECTION_OBJECT_H_

#include <cstdint>
#include <string>
#include <vector>
#include "common_msg/msg/header.h"
#include "common_msg/msg/pose.h"
#include "common_msg/msg/vector3.h"
#include "object_type.h"

#pragma pack(1)

namespace seres_ipc {
namespace visual_perception_msg {
namespace msg {
struct DetectionObject {
    // 消息头
    common_msg::msg::Header header;
    // 对象 ID
    char id[64];
    // 检测结果列表
    char results[102400];
    // 得分
    float score;
    // 中心点位置
    common_msg::msg::Pose center;
    // 尺寸
    common_msg::msg::Vector3 size;
    // 对象类型
    ObjectType type;
    // 存在概率
    float existence;
    // 综合置信度 [0 - 1]
    float confidence;
    // 数据来源类型
    uint8_t source_type;

    // 定义枚举值常量
    static constexpr uint8_t LIDAR = 1;  // 激光雷达
    static constexpr uint8_t CAMERA = 2; // 相机
    static constexpr uint8_t RADAR = 3;  // 毫米波雷达
    static constexpr uint8_t FUSION_FORWORDING = 4; // 前融合
    static constexpr uint8_t FUSION_MID = 5; // 中融合
    static constexpr uint8_t FUSION_LAT = 6; // 后融合
    static constexpr uint8_t DETECTION_2D = 7; // 2D 检测
    static constexpr uint8_t PREDICTION = 8; // 预测
};

} // namespace msg
} // namespace visual_perception_msg
} // namespace seres_ipc

#pragma pack()
#endif
