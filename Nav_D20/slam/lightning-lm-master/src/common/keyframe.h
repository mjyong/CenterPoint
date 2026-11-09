//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_KEYFRAME_H
#define LIGHTNING_KEYFRAME_H

#include "common/eigen_types.h"
#include "common/nav_state.h"
#include "common/point_def.h"
#include "common/std_types.h"

namespace lightning {

/// 关键帧描述
/// NOTE: 在添加后端后，需要加锁
class Keyframe {
   public:
    using Ptr = std::shared_ptr<Keyframe>;

    Keyframe() {}
    Keyframe(unsigned long id, CloudPtr cloud, NavState state)
        : id_(id), cloud_(cloud), state_(state), pose_lio_(state.GetPose()) {
        timestamp_ = state_.timestamp_;
        pose_opt_ = pose_lio_;
    }

    unsigned long GetID() const { return id_; }
    CloudPtr GetCloud() const { return cloud_; }

    SE3 GetLIOPose() {
        UL lock(data_mutex_);
        return pose_lio_;
    }

    void SetLIOPose(const SE3& pose) {
        UL lock(data_mutex_);
        pose_lio_ = pose;

        // also set opt
        pose_opt_ = pose_lio_;
    }

    SE3 GetOptPose() {
        UL lock(data_mutex_);
        return pose_opt_;
    }

    void SetOptPose(const SE3& pose) {
        UL lock(data_mutex_);
        pose_opt_ = pose;
    }

    void SetState(NavState s) {
        UL lock(data_mutex_);
        state_ = s;
    }

    NavState GetState() {
        UL lock(data_mutex_);
        return state_;
    }

   protected:
    unsigned long id_ = 0;

    double timestamp_ = 0;
    CloudPtr cloud_ = nullptr;  /// 降采样之后的点云

    std::mutex data_mutex_;
    SE3 pose_lio_;  // 前端的pose
    SE3 pose_opt_;  // 后端优化后的pose
    SE3 pose_ins_;  // INS的pose
    Eigen::Vector3d ins_position_covariance_ = Eigen::Vector3d::Zero();  // INS位置协方差
    Eigen::Vector3d ins_orientation_covariance_ = Eigen::Vector3d::Zero();  // INS姿态协方差
    NavState state_;  // 卡尔曼滤波器状态
    bool has_ins_pose_ = false; // 是否有有效的INS位姿

   public:
    void SetInsPose(const SE3& pose) {
        UL lock(data_mutex_);
        pose_ins_ = pose;
        has_ins_pose_ = true;
    }

    SE3 GetInsPose() {
        UL lock(data_mutex_);
        return pose_ins_;
    }

    bool HasInsPose() {
        UL lock(data_mutex_);
        return has_ins_pose_;
    }

    void SetInsCovariance(const Eigen::Vector3d& pos_cov, const Eigen::Vector3d& ori_cov) {
        UL lock(data_mutex_);
        ins_position_covariance_ = pos_cov;
        ins_orientation_covariance_ = ori_cov;
    }

    Eigen::Vector3d GetInsPositionCovariance() {
        UL lock(data_mutex_);
        return ins_position_covariance_;
    }

    Eigen::Vector3d GetInsOrientationCovariance() {
        UL lock(data_mutex_);
        return ins_orientation_covariance_;
    }
};

}  // namespace lightning

#endif  // LIGHTNING_KEYFRAME_H
