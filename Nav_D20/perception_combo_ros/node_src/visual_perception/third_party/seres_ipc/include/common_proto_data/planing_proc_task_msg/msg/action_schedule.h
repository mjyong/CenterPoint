#ifndef D10_APPLICATION_PLANING_PROC_TASK_ACTION_SCHEDULE_H_
#define D10_APPLICATION_PLANING_PROC_TASK_ACTION_SCHEDULE_H_

#pragma pack(1)

#include "take_photo.h"
#include "linear_scan.h"
#include "recharging.h"

namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct ActionSchedule {
    // 枚举常量
    static constexpr uint8_t NONE = 0;
    static constexpr uint8_t TAKE_PHOTO = 1;
    static constexpr uint8_t LINEAR_SCAN = 2;
    static constexpr uint8_t RECHARGING = 3;

    uint8_t action_type; // 枚举
    TakePhoto take_photo;
    LinearScan linear_scan;
    Recharging recharging;
    bool action_on_patrol; // false: 到达POI点后在执行动作; true: 移动过程中执行
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_ACTION_SCHEDULE_H_