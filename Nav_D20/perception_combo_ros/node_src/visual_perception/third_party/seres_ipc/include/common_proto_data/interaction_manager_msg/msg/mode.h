#ifndef D10_APPLICATION_INTERACTION_MANAGER_MODE_H_
#define D10_APPLICATION_INTERACTION_MANAGER_MODE_H_

#include <cstdint>
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace interaction_manager_msg {
namespace msg {
struct Mode {
    // Message header
    common_msg::msg::Header header;
    // Mode data, required
    uint8_t mode;
    // Task data, optional
    uint8_t task;

    // Define enumeration constants
    static constexpr uint8_t MODE_FACTORY = 0;
    static constexpr uint8_t MODE_CALIBRATION = 1;
    static constexpr uint8_t MODE_UPDATE = 2;
    static constexpr uint8_t MODE_SLEEP = 3;
    static constexpr uint8_t MODE_ERROR = 4;
    static constexpr uint8_t MODE_WORK = 5;
};
} // namespace msg
} // namespace interaction_manager_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_INTERACTION_MANAGER_MODE_H_
