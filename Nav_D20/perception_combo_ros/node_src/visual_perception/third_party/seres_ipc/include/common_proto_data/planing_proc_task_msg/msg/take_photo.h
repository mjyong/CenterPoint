#ifndef D10_APPLICATION_PLANING_PROC_TASK_TAKE_PHOTO_H_
#define D10_APPLICATION_PLANING_PROC_TASK_TAKE_PHOTO_H_

#include "common_msg/msg/point.h"

#pragma pack(1)

namespace seres_ipc {
namespace planing_proc_task_msg {
namespace msg {

struct TakePhoto {
    common_msg::msg::Point target_point;  // 目标点坐标
    uint32_t num_of_photo;  // 拍照数量
    double time_interval;  // 拍照时间间隔 秒
};

}  // namespace msg
}  // namespace planing_proc_task_msg
}  // namespace seres_ipc

#pragma pack()

#endif  // D10_APPLICATION_PLANING_PROC_TASK_TAKE_PHOTO_H_