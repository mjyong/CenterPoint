#ifndef D10_1230_POINT_FIELD_H_
#define D10_1230_POINT_FIELD_H_

#include <cstdint>
#include <stdint.h>
#include "time.h"
#include "utils/seres_string.h"

#pragma pack(4)
namespace seres_ipc {
namespace common_interface {
namespace msg {
    enum DataType: uint8_t {
        INT8    = 1,
        UINT8   = 2,
        INT16  = 3,
        UINT16  = 4,
        INT32   = 5,
        UINT32  = 6,
        FLOAT32 = 7,
        FLOAT64 = 8
    };
    struct PointField {
        // 字段名称(x, y, z, intensity等)
        Seres::string<64> name;
        // 该字段在点数据中的字节偏移量
        uint32_t offset;
        // 数据类型
        DataType datatype;
        // 该字段的元素数量(通常为1)
        uint32_t count;
    };
} // namespace msg
} // namespace common_interface
} // namespace seres_ipc
#pragma pack()

#endif
