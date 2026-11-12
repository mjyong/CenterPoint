#ifndef D10_APPLICATION_MAP_SERVICE_MAP_OPERATION_H_
#define D10_APPLICATION_MAP_SERVICE_MAP_OPERATION_H_

#include <cstdint>
#include <string>

// Assume Header and MapType are defined in other header files
#include "common_msg/msg/header.h"
#include "common_msg/msg/map_type.h"

#pragma pack(1)

namespace seres_ipc {
namespace map_service_msg {
namespace msg {
struct MapOperation {
    // Message header
    common_msg::msg::Header header;
    // Global stitched map file address
    char filepath[2048];
    // Map type
    common_msg::msg::MapType type;
    // Enumeration value
    uint8_t opr;

    // Define enumeration constants
    static constexpr uint8_t OPT_UPLOAD = 0; // Upload
    static constexpr uint8_t OPT_LOAD = 1;   // Load
    static constexpr uint8_t OPT_DEL = 2;    // Delete
};
} // namespace msg
} // namespace map_service_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_MAP_SERVICE_MAP_OPERATION_H_
