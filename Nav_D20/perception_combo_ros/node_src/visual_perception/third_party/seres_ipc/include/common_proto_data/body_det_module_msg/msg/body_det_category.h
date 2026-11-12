#ifndef D10_APPLICATION_BODYDETMODULE_BODYDETCATEGORY_H_
#define D10_APPLICATION_BODYDETMODULE_BODYDETCATEGORY_H_

#include <cstdint>

#include "common_msg/msg/header.h"
#include "camera_pre_process_module_msg/msg/compressed_image.h"
#include "common_msg/msg/pose_with_covariance.h"
#include "region_of_interest.h"

#pragma pack(1)

namespace seres_ipc {
namespace body_det_module_msg {
namespace msg {
    /**
     * @struct BodyDetCategory
     * @brief 人体检测类别信息
     */
    struct BodyDetCategory {
        common_msg::msg::Header header; ///< BodyDetCategory
        camera_pre_process_module_msg::msg::CompressedImage image; ///< 压缩图像
        common_msg::msg::PoseWithCovariance pose; ///< 3D姿态 (相机坐标系)
        RegionOfInterest roi; ///< 感兴趣区域

        uint8_t category; ///< 枚举值
        // 定义枚举值常量
        static constexpr uint8_t FACE_RECO = 0; ///< 人脸识别
        static constexpr uint8_t HAND_RECO = 1; ///< 手部识别
        static constexpr uint8_t POSE_RECO = 2; ///< 姿态识别
    };
}
}
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_BODYDETMODULE_BODYDETCATEGORY_H_
