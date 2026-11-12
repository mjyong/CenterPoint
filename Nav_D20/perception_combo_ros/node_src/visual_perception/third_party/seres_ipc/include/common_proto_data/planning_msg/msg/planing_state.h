#ifndef D10_APPLICATION_PLANNING_PLANING_STATE_H_
#define D10_APPLICATION_PLANNING_PLANING_STATE_H_

#include <cstdint>
#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace planning_msg {
namespace msg {

struct PlaningState {
    // Message header
    common_msg::msg::Header header;
    // Navigation task execution state
    uint8_t nav_task_exe_state;
    // Action task execution state
    uint8_t act_task_exe_state;

    // Define enumeration constants
    static constexpr uint8_t ON_GOING = 0; // Executing
    static constexpr uint8_t COMPLETE = 1; // Completed
    static constexpr uint8_t PAUSE = 2; // Paused
    static constexpr uint8_t FAIL = 3; // Failed
    static constexpr uint8_t DWELL_TIME = 4; // Dwell time
};
} // namespace msg
} // namespace planning_msg
} // namespace seres_ipc

#pragma pack()
#endif // D10_APPLICATION_PLANNING_PLANING_STATE_H_
