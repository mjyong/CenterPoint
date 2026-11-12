#ifndef D10_APPLICATION_PLANING_PROC_TASK_RECHARGING_H_
#define D10_APPLICATION_PLANING_PROC_TASK_RECHARGING_H_

#include "common_msg/msg/point.h"

#pragma pack(1)

namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct Recharging {
    common_msg::msg::Point target_point;  // Target point for recharging
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_RECHARGING_H_