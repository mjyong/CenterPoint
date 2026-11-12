#ifndef D10_APPLICATION_CAMERA_PRE_PROCESS_MODULE_COMPRESSED_IMAGE_H_
#define D10_APPLICATION_CAMERA_PRE_PROCESS_MODULE_COMPRESSED_IMAGE_H_

#include <cstdint>

#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace camera_pre_process_module_msg {
namespace msg {
    /**
     * @struct CompressedImage
     * @brief 该消息包含一张压缩图像。
     */
    struct CompressedImage {
        common_msg::msg::Header header; ///< 消息头
        char format[8]; ///< 图像格式
        uint8_t data[16106127360]; ///< 图像数据
    };
}
}
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_CAMERA_PRE_PROCESS_MODULE_COMPRESSED_IMAGE_H_
