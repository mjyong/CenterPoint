#ifndef D10_1230_PLANING_PROC_TASK_H_
#define D10_1230_PLANING_PROC_TASK_H_

#include "common_interface/msg/header.h"
#include "planning_schedule.h"
#include "utils/seres_vector.h"
#include "utils/seres_string.h"
#include "action_schedule.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum LoadPOIType{
        LOAD_POI_DATA = 1, // 加载任务点集
        LOAD_POI_FILE = 2 // 加载指定任务文件路径
    };

    struct PlaningProcTask {
        common_interface::msg::Header header;
        int valid_size;
        Seres::vector<PlanningSchedule, 100> nav_sche;
        Seres::vector<ActionSchedule, 100> act_sche;
        Seres::string<128> filepath;
        LoadPOIType load_poi_type;

    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif 
