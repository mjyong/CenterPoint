#ifndef D10_1230_ODOMETRY_STATE_H
#define D10_1230_ODOMETRY_STATE_H

#include "common_interface/msg/header.h"

#pragma pack(4)
namespace seres_ipc {
namespace nav_interface {
namespace msg {
    enum OdometryStateStates{
        LOCALIZATION_ODOMETRY_RESIDUAL = 0,      // 残差不收敛 
        LOCALIZATION_ODOMETRY_MATCH_POINTS,  // 匹配点数太少
        LOCALIZATION_ODOMETRY_MATCH_SCORE,   // 匹配得分低
        LOCALIZATION_ODOMETRY_OK   // 匹配得分低
    };

    struct OdometryState {
        common_interface::msg::Header header;
        OdometryStateStates state;
    };
}
}
}

#pragma pack()

#endif 
