#ifndef D10_APPLICATION_INTERACTION_MANAGER_BROAD_CAST_INFO_H_
#define D10_APPLICATION_INTERACTION_MANAGER_BROAD_CAST_INFO_H_

#include <string>
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace interaction_manager_msg {
namespace msg {
struct BroadCastInfo {
    // Message header
    common_msg::msg::Header header;
    // Broadcast information
    char cast_info[256];
};
} // namespace msg
} // namespace interaction_manager_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_INTERACTION_MANAGER_BROAD_CAST_INFO_H_
