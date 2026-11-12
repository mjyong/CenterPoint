#ifndef D10_APPLICATION_COMMON_HEADER_H_
#define D10_APPLICATION_COMMON_HEADER_H_

#include <cstdint>
#include "time.h"

#pragma pack(1)

namespace seres_ipc {
namespace common_msg {
namespace msg {
struct Header {
    // Current module name,根据物理架构模块名确定，命名不要超过32个字节
    char module_name[32]; 
    // Timestamp in milliseconds
    double timestamp_ms;
    // Forward module names
    char pre_module_names[128][32];
    // Forward module timestamps
    Time pre_module_recvtime[128];
    // Frame ID
    char frame_id[10];
};
} // namespace msg
} // namespace common_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_COMMON_HEADER_H_

