#ifndef D10_1230_LIDAR_PRE_PROCESS_STATE_H_
#define D10_1230_LIDAR_PRE_PROCESS_STATE_H_

#include "common_interface/msg/header.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum LidarPreProcessStateStates{
        SENSOR_DATA_ANOMALY_TIMESTAMP = 0,       // 传感器时间戳异常
        SENSOR_DATA_ANOMALY_FREQUENCY,       // 数据频率异常
        SENSOR_DATA_ANOMALY_CONTENT         // 数据内容异常
    };
    struct LidarPreProcessState {
        common_interface::msg::Header header;
        LidarPreProcessStateStates state;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif