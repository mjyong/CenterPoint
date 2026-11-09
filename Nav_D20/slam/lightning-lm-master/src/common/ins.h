
//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_INS_H
#define LIGHTNING_INS_H

#include "common/eigen_types.h"

namespace lightning {

/// INS状态枚举
enum InsState {
    INS_INVALID = 0,  /// 无效数据
    INS_VALID = 1,    /// 有效数据（已对齐）
    INS_ALIGNING = 2  /// 对齐中（可选）
};

/// 内部定义的里程计观测，如果需要ROS，则从sensor_msg/Imu 的消息转换过来

struct Ins
{
  double stamp;
  InsState state = INS_INVALID;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::Vector3d velocity;
  Eigen::Vector3d velocity_covariance;
  Eigen::Vector3d position_covariance;
  Eigen::Vector3d orientation_covariance;
};

using INSPtr = std::shared_ptr<Ins>;

}  // namespace lightning

#endif  // LIGHTNING_INS_H


