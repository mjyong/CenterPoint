#ifndef D10_1230_PLANNING_SCHEDULE_H_
#define D10_1230_PLANNING_SCHEDULE_H_

#include "utils/seres_vector.h"
#include "common_interface/msg/point.h"
#include "common_interface/msg/quaternion.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum NavType{
        AUTO_ROUTING = 1, //输入目标点，自动规划路线
        FIXED_ROUTING = 2 //输入目标路径
    };

    struct PlanningSchedule {
        NavType type;
        int32_t valid_size;
        Seres::vector<common_interface::msg::Point, 100000> target_points;
        bool attitude_adjustment_at_end;
        common_interface::msg::Quaternion pose_at_end;       
    };
}
}
}

#pragma pack()

#endif
