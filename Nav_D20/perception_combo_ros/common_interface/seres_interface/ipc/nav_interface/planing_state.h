#ifndef D10_1230_PLANING_STATE_H_
#define D10_1230_PLANING_STATE_H_

#include "common_interface/msg/header.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum NavTaskExeState {
        NavTaskExeStateOnGoing = 0,    //执行中
        NavTaskExeStateComplete = 1,   //完成
        NavTaskExeStatePause = 2,      //暂停
        NavTaskExeStateFail = 3        //失败
    };
    

    enum ActTaskExeState {
        ActTaskExeStateOnGoing = 0,     //执行中
        ActTaskExeStateComplete = 1,    //完成
        ActTaskExeStatePause = 2,       //暂停
        ActTaskExeStateFail = 3,        //失败
        ActTaskExeStateDwellTime = 4    //停留计时
    };
    struct PlaningState {
        common_interface::msg::Header header;
        uint8_t poi_id;
        NavTaskExeState nav_task_exe_state;
        ActTaskExeState act_task_exe_state;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif