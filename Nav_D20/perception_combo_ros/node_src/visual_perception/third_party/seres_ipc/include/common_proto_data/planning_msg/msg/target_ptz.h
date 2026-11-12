#ifndef D10_APPLICATION_PLANNING_TARGET_PTZ_H_
#define D10_APPLICATION_PLANNING_TARGET_PTZ_H_

#include "common_msg/msg/header.h"

#pragma pack(1)

namespace seres_ipc {
namespace planning_msg {
namespace msg {

struct TargetPTZ {
    common_msg::msg::Header header;
    double p;
    double t;
    double z;
    bool take_photo;
};

}  // namespace msg
}  // namespace planning_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANNING_TARGET_PTZ_H_