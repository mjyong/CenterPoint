#ifndef D10_APPLICATION_BODYDETMODULE_REGION_OF_INTEREST_H_
#define D10_APPLICATION_BODYDETMODULE_REGION_OF_INTEREST_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace body_det_module_msg {
namespace msg {
    /**
     * @struct RegionOfInterest
     * @brief 定义感兴趣区域的相关信息
     */
    struct RegionOfInterest {
        uint32_t x_offset; ///< 感兴趣区域左上角的 x 轴偏移量
        uint32_t y_offset; ///< 感兴趣区域左上角的 y 轴偏移量
        uint32_t height;   ///< 感兴趣区域的高度
        uint32_t width;    ///< 感兴趣区域的宽度
        bool do_rectify;   ///< 是否对该区域进行校正
    };
}
}
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_BODYDETMODULE_REGION_OF_INTEREST_H_
