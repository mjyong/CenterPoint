#ifndef _APPLICATION_COMMON_MSG_TIME_H_
#define _APPLICATION_COMMON_MSG_TIME_H_

#include <cstdint>

#pragma pack(4)

namespace seres_ipc {
namespace common_interface {
namespace msg {
    struct Time {
        // Number of seconds since a predefined epoch.
        uint32_t sec;
        // Number of nanoseconds within the current second. Range: 0 - 999,999,999
        uint32_t nanosec;
    };

} // namespace msg
} // namespace common_interface
} // namespace seres_ipc

#pragma pack()
#endif // _APPLICATION_COMMON_MSG_TIME_H_
