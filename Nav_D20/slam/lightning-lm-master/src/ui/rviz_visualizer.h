#pragma once

#include <memory>
#include <map>

#include "common/eigen_types.h"
#include "common/keyframe.h"
#include "common/nav_state.h"
#include "common/point_def.h"

namespace lightning::ui {

class RvizVisualizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RvizVisualizer();
    ~RvizVisualizer();

    /// @brief 初始化ROS发布器，需要传入ROS节点
    bool Init(rclcpp::Node::SharedPtr node);

    void Reset(const std::vector<Keyframe::Ptr>& keyframes);

    /// 更新激光地图点云，在激光定位中的地图发生改变时，由fusion调用
    void UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud);

    /// 更新动态图层点云
    void UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud);

    /// 更新kalman滤波器状态
    void UpdateNavState(const NavState& state);

    /// 更新最新的pose
    void UpdateRecentPose(const SE3& pose);

    /// 更新一次scan和它对应的Pose
    void UpdateScan(CloudPtr cloud, const SE3& pose);

    void UpdateKF(std::shared_ptr<Keyframe> kf);

    /// 等待显示线程结束，并释放资源
    void Quit();

    /// 用户是否已经退出UI
    bool ShouldQuit();

    /// 设置IMU到雷达的外参
    void SetTImuLidar(const SE3& T_imu_lidar);

    /// 设置需要保留多少个扫描数据
    void SetCurrentScanSize(int current_scan_size);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace lightning::ui