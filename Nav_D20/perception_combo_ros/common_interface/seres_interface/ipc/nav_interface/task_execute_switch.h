#ifndef D10_1230_TASK_EXECUTE_SWITCH_H_
#define D10_1230_TASK_EXECUTE_SWITCH_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum TaskExeState{
        TaskExeStateNone = 0,
        TaskExeStateStart = 1, //开始任务，顺序执行POI任务集
        TaskExeStatePause = 2, //暂停
        TaskExeStateResume = 3, //继续
        TaskExeStateStop = 4, //结束，结束剩余所有POI任务
        TaskExeStateNext = 6, //切换至下一个POI任务
        TaskExeStatePrev = 7 //切换至上一个POI任务
    };

    struct TaskExecuteSwitch {
        common_interface::msg::Header header;
        TaskExeState task_exe_state;
        uint32_t task_id;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif
