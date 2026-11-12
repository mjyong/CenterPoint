#ifndef D10_APPLICATION_LIDAR_POINT_FIELD_H_
#define D10_APPLICATION_LIDAR_POINT_FIELD_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace lidar_msg {
namespace msg {
struct PointField {
    // Name of the field
    char name[64];
    // Offset from the start of the point struct
    uint32_t offset;
    // Number of elements in the field
    uint32_t count;
    // Data type of the field
    uint8_t datatype;

    // Define enumeration constants
    static constexpr uint8_t INT8 = 1;
    static constexpr uint8_t UINT8 = 2;
    static constexpr uint8_t INT16 = 3;
    static constexpr uint8_t UINT16 = 4;
    static constexpr uint8_t INT32 = 5;
    static constexpr uint8_t UINT32 = 6;
    static constexpr uint8_t FLOAT32 = 7;
    static constexpr uint8_t FLOAT64 = 8;
};
} // namespace msg
} // namespace lidar_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_LIDAR_POINT_FIELD_H_
