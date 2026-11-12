#ifndef D10_1230_TASK_H
#define D10_1230_TASK_H

#include "utils/seres_string.h"
#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum ClientType {
        REMOTE_CONTROLLER = 0,          // 遥控器
        WEB_TERMINAL,              // Web 终端
        APP                       // 平板APP
    };

    enum ETask {
        PATROL = 0,     // 巡逻
        TREAMING,  // 推流
        REMOTE_CTRL,     // 远控
        PTZ_CTRL     // 云台控制
    };

    enum TaskState{
        TaskStateOff = 0,       //关闭
        TaskStateOn,        //开启
        TaskStatePause,     // 暂停
        TaskStateRecovery,   //恢复
        TaskStateUnknown
    };

    struct Task {
        Seres::string<64> uuid;
        ClientType client_type;
        ETask task;
        TaskState state;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc
#pragma pack()
#endif
