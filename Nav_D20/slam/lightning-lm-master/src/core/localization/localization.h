#pragma once

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/int32.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "common/imu.h"
#include "core/lio/laser_mapping.h"
#include "core/super_lio_frontend/laser_mapping_superlio.h"
#include "core/system/async_message_process.h"
#include "core/localization/pose_graph/smoother.h"
#include "core/maps/tiled_map.h"
#include "common/ins.h"
/// 预声明
namespace lightning {
namespace ui {
class RvizVisualizer;
}

namespace loc {

/**
 * 实时定位接口实现
 */
class Localization {
   public:
    struct Options {
        Options() {}

        bool online_mode_ = false;  // 在线模式还是离线模式
        bool with_ui_ = false;      // 是否带ui

        /// 参数
        SE3 T_body_imu_;

        bool enable_lidar_odom_skip_ = false;  // 是否允许激光里程计跳帧
        int lidar_odom_skip_num_ = 1;          // 如果允许跳帧，跳多少帧
        bool use_super_lio_ = false;           // 是否使用SuperLIO前端
    };
    std::deque<INSPtr> ins_buffer_;
    Localization(Options options = Options());
    ~Localization() = default;

    /**
     * 初始化，读配置参数
     * @param yaml_path
     * @param global_map_path
     * @param init_reloc_pose
     */
    bool Init(const std::string& yaml_path, const std::string& global_map_path);

    /// 处理lidar消息
    void ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg);
    void ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr laser_msg);

    /// 处理IMU消息
    void ProcessIMUMsg(IMUPtr imu);

    // void ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) override;

    /// 由外部设置pose，适用于手动重定位
    /// @param skip_ndt: true 时跳过 NDT 配准，直接信任外部给的位姿
    void SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t,
                         bool skip_ndt = false);

    /// 切换地图并重新初始化位姿
    bool SwitchMapAndReinit(const std::string& map_path, const SE3& init_pose);

    /// 结束，保存临时地图
    void Finish();

    /// 获取地图点云（静态+动态）
    CloudPtr GetMapCloud();

    /// 异步处理函数
    void LidarOdomProcCloud(CloudPtr);

    using TFCallback = std::function<void(const geometry_msgs::msg::TransformStamped& odom)>;
    using OdometryCallback = std::function<void(const nav_msgs::msg::Odometry& odom)>;
    using PointcloudBodyCallback = std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>;
    using PointcloudWorldCallback = std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>;
    using LidarOdomCallback = std::function<void(const NavState& state)>;

    void SetTFCallback(TFCallback&& callback);
    void SetOdometryCallback(OdometryCallback&& callback);
    void SetPointcloudWorldCallback(PointcloudWorldCallback&& callback);
    void SetPointcloudBodyCallback(PointcloudBodyCallback&& callback);
    void SetLidarOdomCallback(LidarOdomCallback&& callback);
    double GetConfidence() const;
    bool IsNDTInited() const;
    bool IsLocFailed() const;
    bool WasReinitTriggered() const;
    bool IsInitPoseSet() const;

   private:
    /// 模块  ========================================================================================================
    mutable std::mutex global_mutex_;  // 防止处理过程中被重复init
    Options options_;

    /// 预处理
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess

    /// 前端
    std::shared_ptr<LaserMapping> lio_ = nullptr;
    std::shared_ptr<LaserMappingSuperLIO> super_lio_ = nullptr;
    Keyframe::Ptr lio_kf_ = nullptr;

    // ui
    std::shared_ptr<ui::RvizVisualizer> ui_ = nullptr;

    /// 全局分块地图（按位姿动态加载局部地图到 IVox）
    std::shared_ptr<TiledMap> tiled_map_ = nullptr;

    /// 输出平滑器
    std::shared_ptr<PoseSmoother> smoother_ = nullptr;

    /// async 处理
    sys::AsyncMessageProcess<CloudPtr> lidar_odom_proc_cloud_;  // lidar odom 处理点云

    /// 框架相关
    TFCallback tf_callback_;
    OdometryCallback odometry_callback_;
    PointcloudBodyCallback pointcloud_body_callback_;
    PointcloudWorldCallback pointcloud_world_callback_;
    LidarOdomCallback lidar_odom_callback_;

    /// 输入检查
    double last_imu_time_ = 0;
    double last_odom_time_ = 0;
    double last_cloud_time_ = 0;

    /// INS 数据处理
    INSPtr FindNearestINS(double stamp);
    void RemoveOldINS(double stamp);

};
}  // namespace loc

}  // namespace lightning