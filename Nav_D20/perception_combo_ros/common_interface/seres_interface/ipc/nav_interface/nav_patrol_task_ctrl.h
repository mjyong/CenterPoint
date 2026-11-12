#ifndef D10_1230_NAV_PATRO_TASK_CTRL_H_
#define D10_1230_NAV_PATRO_TASK_CTRL_H_

#include "task.h"
#include "utils/seres_vector.h"
#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct NavPatroTaskCtrl {
        common_interface::msg::Header header;
        //任务开始时间
        uint64_t start_time;
        //任务结束时间
        uint64_t end_time;
        //巡逻周期 单位：小时
        uint32_t patrol_cycle;
        Seres::vector<Task, 10> tasks;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()
#endif