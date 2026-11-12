#ifndef D10_1230_NAV_ALGORITHM_DIAGNOSTIC_STATUS_H
#define D10_1230_NAV_ALGORITHM_DIAGNOSTIC_STATUS_H

#include "common_interface/msg/header.h"
#include "utils/seres_vector.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
    struct NavAlgorithmDiagnosticStatus {
        enum AlgorithmDiagnosticID{
            // 传感器数据异常
            SENSOR_DATA_ANOMALY_TIMESTAMP = 0,       // 传感器时间戳异常
            SENSOR_DATA_ANOMALY_FREQUENCY,       // 数据频率异常
            SENSOR_DATA_ANOMALY_CONTENT,         // 数据内容异常

            // 定位异常 - 里程计异常
            LOCALIZATION_ODOMETRY_RESIDUAL,      // 残差不收敛
            LOCALIZATION_ODOMETRY_MATCH_POINTS,  // 匹配点数太少
            LOCALIZATION_ODOMETRY_MATCH_SCORE,   // 匹配得分低

            // 定位异常 - 地图匹配异常
            LOCALIZATION_MAP_MATCHING_SCORE,     // 地图匹配得分太低
            LOCALIZATION_MAP_MISMATCH,           // 地图误匹配

            // 建图异常 - 里程计异常
            MAPPING_ODOMETRY_OPTIMIZATION,       // 优化不收敛
            MAPPING_ODOMETRY_MATCH_POINTS,       // 匹配点数太少
            MAPPING_ODOMETRY_MATCH_SCORE,        // 匹配得分低

            // 建图异常 - 回环异常
            MAPPING_LOOP_CLOSURE,                // 回环异常

            // 建图异常 - 后端优化异常
            MAPPING_BACKEND_OPTIMIZATION,        // 优化不收敛

            // 其他状态
            HEALTHY,                             // 正常状态
            UNKNOWN                              // 未知状态
        };
        common_interface::msg::Header header;
        Seres::vector<AlgorithmDiagnosticID, 1024> alg_diag_id;
    };
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()

#endif
