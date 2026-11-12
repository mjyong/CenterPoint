#ifndef D10_APPLICATION_MAP_SERVICE_MAP_UPLOAD_H_
#define D10_APPLICATION_MAP_SERVICE_MAP_UPLOAD_H_

#include "common_msg/msg/header.h"
#include "common_msg/msg/map_type.h"

#pragma pack(1)

namespace seres_ipc {
namespace map_service_msg {
namespace msg {
struct MapUpload {
    // Message header
    common_msg::msg::Header header;
    // Global stitched map file address
    char filepath[2048];
    // Map type
    common_msg::msg::MapType type;
};
} // namespace msg
} // namespace map_service_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_MAP_SERVICE_MAP_UPLOAD_H_
