#ifndef D10_APPLICATION_PLANING_PROC_TASK_NAV_SCHEDULE_H_
#define D10_APPLICATION_PLANING_PROC_TASK_NAV_SCHEDULE_H_

#pragma pack(1)

#include "common_msg/msg/point.h"
#include "common_msg/msg/quaternion.h"

namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct NavSchedule {
    // 枚举常量
    static constexpr uint8_t AUTO_ROUTING = 1;   // 输入目标点，自动规划路线
    static constexpr uint8_t FIXED_ROUTING = 2;  // 输入目标路径
    uint8_t nav_type; // 枚举

    int32_t valid_size;
    common_msg::msg::Point target_points[2000];  // nav_type为auto_routing时,只使用第一个点
    bool attitude_adjustment_at_end;
    common_msg::msg::Quaternion post_at_end;
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_NAV_SCHEDULE_H_