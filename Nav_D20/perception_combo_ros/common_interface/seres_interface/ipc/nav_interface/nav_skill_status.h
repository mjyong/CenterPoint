#ifndef D10_1230_NAV_SKILL_STATUS_H_
#define D10_1230_NAV_SKILL_STATUS_H_

#include "common_interface/msg/header.h"
#include "utils/seres_string.h"
#include "task.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    // enum ClientType {
    //     REMOTE_CONTROLLER = 0,         // 遥控器
    //     WEB_TERMINAL,              // Web 终端
    //     APP,                        // 平板APP
    //     DEFAULT                    // 默认值
    // };
    enum Status{
        SUCCESS     = 0, //执行成功
        FAILURE     = 1, //执行失败
        UNKNOW          //未知
    };


    struct NavSkillStatus {
        common_interface::msg::Header header;
        Seres::string<64> uuid;
        ClientType client_type;
        Seres::string<64> skill;
        Status status;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif 