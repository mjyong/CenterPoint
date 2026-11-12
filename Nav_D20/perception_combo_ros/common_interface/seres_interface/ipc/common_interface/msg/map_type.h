#ifndef _APPLICATION_COMMON_MAP_TYPE_H_
#define _APPLICATION_COMMON_MAP_TYPE_H_

#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
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
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_MAP_TYPE_H_
