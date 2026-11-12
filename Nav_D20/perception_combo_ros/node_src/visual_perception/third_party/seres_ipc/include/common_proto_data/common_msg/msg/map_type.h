#ifndef D10_APPLICATION_COMMON_MAP_TYPE_H_
#define D10_APPLICATION_COMMON_MAP_TYPE_H_

#include <cstdint>

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct MapType {
    // Enumeration value
    uint8_t map_type;

    // Define enumeration constants
    static constexpr uint8_t GLOBAL_MAP = 1;
    static constexpr uint8_t MAP_UPDATE = 2;
    static constexpr uint8_t LOCAL_MAP = 3;
    static constexpr uint8_t EXPLORE_MAP = 4;
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_MAP_TYPE_H_
