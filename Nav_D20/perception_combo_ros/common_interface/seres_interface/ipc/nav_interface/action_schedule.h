#ifndef D10_1230_ACTION_SCHEDULE_H_
#define D10_1230_ACTION_SCHEDULE_H_

#include "common_interface/msg/point.h"
#include "take_photo.h"
#include "linear_scan.h"
#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum ActionType{
        ActionTypeTakePhoto = 1,
        ActionTypeLinearScan = 2,
        ActionTypeRecharging = 3,
        ActionTypeNone = 0
    };

    struct ActionSchedule {
        ActionType type;
        TakePhoto photo;
        LinearScan scan;
        common_interface::msg::Point recharging_target_point;
        bool action_on_patrol;
    };

}
}
}

#pragma pack()

#endif