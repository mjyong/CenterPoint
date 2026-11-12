#ifndef D10_APPLICATION_PLANING_PROC_TASK_PLANING_PROC_TASK_H_
#define D10_APPLICATION_PLANING_PROC_TASK_PLANING_PROC_TASK_H_

#include "common_msg/msg/header.h"
#include "nav_schedule.h"
#include "action_schedule.h"

#pragma pack(1)
namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct PlaningProcTask {
    common_msg::msg::Header header;
    uint32_t valid_size;
    NavSchedule nav_sche[255];
    uint32_t nav_sche_size;
    ActionSchedule act_sche[255];
    uint32_t act_sche_size;
    char filepath[256];  // Assuming maximum length for string
    uint8_t loadpoitype;

    static constexpr uint8_t LOAD_POI_DATA = 0;
    static constexpr uint8_t LOAD_POI_FILE = 1;
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc
#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_PLANING_PROC_TASK_H_