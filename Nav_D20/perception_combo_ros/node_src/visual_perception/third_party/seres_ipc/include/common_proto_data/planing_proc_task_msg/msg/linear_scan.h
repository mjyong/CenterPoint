#ifndef D10_APPLICATION_PLANING_PROC_TASK_LINEAR_SCAN_H_
#define D10_APPLICATION_PLANING_PROC_TASK_LINEAR_SCAN_H_

#include "common_msg/msg/point.h"

#pragma pack(1)

namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct LinearScan {
    // LinearScan
    common_msg::msg::Point target_start_point;
    common_msg::msg::Point target_stop_point;
    double duration;  // 扫描持续时间 秒
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_LINEAR_SCAN_H_