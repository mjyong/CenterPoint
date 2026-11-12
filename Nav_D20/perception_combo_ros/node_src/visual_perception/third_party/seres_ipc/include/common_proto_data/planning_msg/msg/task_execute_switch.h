#ifndef D10_APPLICATION_PLANNING_TASK_EXECUTE_SWITCH_H_
#define D10_APPLICATION_PLANNING_TASK_EXECUTE_SWITCH_H_

#include "common_msg/msg/header.h"

#pragma pack(1)
namespace seres_ipc {
namespace planning_msg {
namespace msg {

struct TaskExecuteSwitch {
    common_msg::msg::Header header;
    uint32_t unique_id;
    uint8_t task_exe_state; // 枚举值

    // 枚举值常量定义
    static constexpr uint8_t NONE = 0;
    static constexpr uint8_t START = 1; // 开始任务，顺序执行POI任务集
    static constexpr uint8_t PAUSE = 2; // 暂停
    static constexpr uint8_t RESUME = 3; // 继续
    static constexpr uint8_t STOP = 4; // 结束，结束剩余所有POI任务
    static constexpr uint8_t NEXT = 6; // 切换至下一个POI任务
    static constexpr uint8_t PREV = 7; // 切换至上一个POI任务
};

}  // namespace msg
}  // namespace planning_msg
}  // namespace seres_ipc
#pragma pack()

#endif  // D10_APPLICATION_PLANNING_TASK_EXECUTE_SWITCH_H_