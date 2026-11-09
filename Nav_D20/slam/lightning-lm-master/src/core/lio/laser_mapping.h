#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/ins.h"
#include "common/keyframe.h"
#include "common/options.h"
#include "core/ivox3d/ivox3d.h"
#include "core/lio/eskf.hpp"
#include "core/lio/imu_processing.hpp"
#include "core/maps/tiled_map.h"
#include "pointcloud_preprocess.h"

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning {

namespace ui {
class RvizVisualizer;
}

/**
 * laser mapping
 * 目前有个问题：点云在缓存之后，实际处理的并不是最新的那个点云（通常是buffer里的前一个），这是因为bag里的点云用的开始时间戳，导致
 * 点云的结束时间要比IMU多0.1s左右。为了同步最近的IMU，就只能处理缓冲队列里的那个点云，而不是最新的点云
 */
class LaserMapping {
   public:
    struct Options {
        Options() {}

        bool is_in_slam_mode_ = true;  // 是否在slam模式下

        /// 关键帧阈值
        double kf_dis_th_ = 2;
        double kf_angle_th_ = 10 * M_PI / 180.0;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

    LaserMapping(Options options = Options());
    ~LaserMapping() {
        if (ivox_async_thread_.joinable()) {
            ivox_async_thread_.join();
        }
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init without ros
    bool Init(const std::string &config_yaml);

    bool Run();

    // callbacks of lidar and imu
    /// 处理ROS2的点云
    void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    /// 处理livox的点云
    void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg);

    /// 如果已经做了预处理，也可以直接处理点云
    void ProcessPointCloud2(CloudPtr cloud);

    void ProcessIMU(const lightning::IMUPtr &msg_in);

    /// 保存前端的地图
    void SaveMap();
    // void SetUI(std::shared_ptr<ui::PangolinWindow> ui) { ui_ = ui; }
    void SetUI(std::shared_ptr<ui::RvizVisualizer> ui) { ui_ = ui; }

    /// 获取关键帧
    Keyframe::Ptr GetKeyframe() const { return last_kf_; }

    /// 获取激光的状态
    NavState GetState() const { return state_point_; }

    /// 获取IMU状态
    NavState GetIMUState() const {
        if (p_imu_->IsIMUInited()) {
            auto nav_state = kf_imu_.GetX();
            return nav_state;
        } else {
            NavState s;
            s.pose_is_ok_ = false;
            return s;
        }
    }

    CloudPtr GetScanUndist() const { return scan_undistort_; }
    CloudPtr GetScanImu() const { return scan_cloud_imu_; }
    /// 获取最新的点云
    CloudPtr GetRecentCloud();

    std::vector<Keyframe::Ptr> GetAllKeyframes() { return all_keyframes_; }

    /**
     * 计算全局地图
     * @param use_lio_pose
     * @return
     */
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);

    /// 设置全局分块地图，用于 IVox 动态加载
    void SetTiledMap(std::shared_ptr<TiledMap> map) { tiled_map_ = map; }

    /// 设置初始位姿（用于首帧地图加载和 ESKF 初始化）
    /// @param skip_ndt: true 时跳过 NDT 配准，直接信任外部给的位姿（如 INS RTK_FIXED）
    void SetInitPose(const SE3& pose, bool skip_ndt = false);

    /// 重置定位状态，用于外部触发重新初始化（地图切换/重定位）
    void Reinit();

    /// 处理 INS 观测，作为 ESKF 的 GPS 观测融合
    void ProcessINS(const INSPtr& ins);

    /// 重置 IVox，下次 Run 时重新从 TiledMap 加载
    void ResetIVox() {
        map_loaded_ = false;
        ndt_inited_ = false;
        flg_first_scan_ = true;
    }
    /// NDT位置初始化是否完成
    bool IsNDTInited() const { return ndt_inited_; }

    /// 获取最近一次 NDT 匹配置信度
    double GetConfidence() const { return ndt_confidence_; }

    /// 定位是否故障（effect num过少或残差过大）
    bool IsLocFailed() const { return loc_failed_; }

    /// 是否触发了自动重初始化
    bool WasReinitTriggered() const { return reinit_triggered_; }

    /// 获取重初始化时的上一个有效位姿
    SE3 GetLastGoodPose() const { return last_good_pose_; }

    /// 外部初始位姿是否已设置
    bool IsInitPoseSet() const { return init_pose_set_by_external_; }

   private:
    // sync lidar with imu
    bool SyncPackages();
    void UpdateImuStatistics(std::pair<Vec3d, Vec3d>& imu_gyro, std::pair<Vec3d, Vec3d>& imu_acc);
    std::pair<Vec3d, Vec3d> CalculateMeanAndVariance(const std::vector<Vec3d>& data_window);

    void ObsModel(NavState &s, ESKF::CustomObservationModel &obs);

    inline void PointBodyToWorld(const PointType &pi, PointType &po) {
        Vec3d p_global(state_point_.rot_ * (state_point_.offset_R_lidar_ * pi.getVector3fMap().cast<double>() +
                                            state_point_.offset_t_lidar_) +
                       state_point_.pos_);

        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.intensity = pi.intensity;
    }

    void MapIncremental();

    /// 根据 TiledMap 当前加载的局部地图重建 IVox（同步版本，实际重建逻辑）
    void SyncIVoxFromTiledMap();

    /// 触发异步地图加载与 IVox 重建（非阻塞）
    void AsyncSyncIVoxFromTiledMap();

    /// 检查异步加载是否完成，完成则替换 IVox
    bool CheckAsyncIVoxReady();

    /// 多级NDT+yaw搜索初始化位置（定位模式首帧使用）
    bool YawSearch(SE3& pose, double& confidence, CloudPtr input, CloudPtr output);

    /// GPS/INS 观测函数（注册到 ESKF）
    void GPSObsModel(NavState& s, ESKF::CustomObservationModel& obs);

    bool LoadParamsFromYAML(const std::string &yaml);

    /// 创建关键帧
    void MakeKF();

   private:
    Options options_;

    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// local map related
    double filter_size_map_min_ = 0;

    /// params
    std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
    std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
    std::string map_file_path_;

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    /// point clouds data
    CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion
    CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body
    CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world
    CloudPtr scan_cloud_imu_{new PointCloudType()};
    std::vector<PointVector> nearest_points_;         // nearest points of current scan
    std::vector<Vec4f> corr_pts_;                     // inlier pts
    std::vector<Vec4f> corr_norm_;                    // inlier plane norms
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan

    std::vector<float> residuals_;           // point-to-plane residuals
    std::vector<bool> point_selected_surf_;  // selected points
    std::vector<Vec4f> plane_coef_;          // plane coeffs

    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_;

    std::deque<PointCloudType::Ptr> lidar_buffer_;
    std::deque<double> lidar_max_distance_buffer_;
    std::deque<lightning::IMUPtr> imu_buffer_;

    /// options
    bool keep_first_imu_estimation_ = false;    // 在没有建立地图前，是否要使用前几帧的IMU状态
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;

    bool enable_skip_lidar_ = true;  // 雷达是否需要跳帧
    int skip_lidar_num_ = 5;         // 每隔多少帧跳一个雷达
    int skip_lidar_cnt_ = 0;

    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    int effect_feat_num_ = 0, frame_num_ = 0;

    double last_lidar_time_ = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    MeasureGroup measures_;  // sync IMU and lidar scan

    ESKF kf_;      // 点云时刻的IMU状态
    ESKF kf_imu_;  // imu 最新时刻的eskf状态

    NavState state_point_;  // ekf current state
    NavState last_ekf_state_point_;

    Vec3d pos_lidar_;  // lidar position after eskf update
    SO3 euler_cur_;    // rotation in euler angles
    bool extrinsic_est_en_ = true;
    bool use_aa_ = false;  // use anderson acceleration?

    std::shared_ptr<ui::RvizVisualizer> ui_ = nullptr;
    std::vector<Vec3d> gyro_data_window_;
    std::vector<Vec3d> acc_data_window_;
    size_t imu_window_size_ = 50;  // 可根据需求调整滑窗大小

    /// TiledMap 动态加载相关
    std::shared_ptr<TiledMap> tiled_map_ = nullptr;  // 全局分块地图
    SE3 last_map_sync_pose_;                          // 上次地图同步位姿
    bool map_loaded_ = false;                         // 地图是否已加载
    double map_update_dist_ = 100.0;                  // 位姿移动超过该距离触发 IVox 重建(m)，与 chunk_size 对齐
    float map_downsample_res_ = 0.5;                  // 地图点云下采样分辨率(m)

    /// INS/GPS 观测数据（供 GPSObsModel 使用）
    SE3 ins_pose_;                                    // INS 位姿
    Vec3d ins_pos_cov_ = Vec3d(1.0, 1.0, 1.0);       // 位置协方差
    Vec3d ins_rot_cov_ = Vec3d(0.1, 0.1, 0.1);       // 姿态协方差
    bool ins_obs_ready_ = false;                      // 是否有待处理的 INS 观测

    /// 外部初始位姿（IMU 初始化后恢复用）
    SE3 external_init_pose_;
    bool init_pose_set_by_external_ = false;           // 是否通过外部 SetInitPose 设置
    bool init_pose_restored_ = false;                  // IMU 初始化后是否已恢复位姿
    bool skip_ndt_init_ = false;                       // 跳过 NDT 配准，直接信任外部位姿

    /// NDT初始化相关参数
    double min_init_confidence_ = 2.0;                 // NDT初始化最小置信度
    bool ndt_inited_ = false;                          // NDT位置初始化是否完成
    double ndt_confidence_ = 0.0;                      // NDT最近一次匹配置信度

    /// 定位故障检测参数
    int loc_fail_count_ = 0;                           // 连续故障帧计数
    bool loc_failed_ = false;                          // 定位是否故障
    bool need_reinit_ = false;                         // 需要重新初始化（下一帧触发）
    bool reinit_triggered_ = false;                    // 重初始化已触发，下一帧 init block 需用 last_good_pose_
    SE3 last_good_pose_;                               // 故障前的最后一个有效位姿
    int min_effect_feat_ = 100;                     // 最小有效特征点数
    double max_avg_residual_ = 0.15;                    // 最大平均残差(m)
    int max_fail_count_ = 100;                          // 连续故障帧阈值

    /// 异步 IVox 重建相关
    std::thread ivox_async_thread_;                    // 异步加载线程
    std::mutex ivox_async_mutex_;                     // 保护异步结果
    std::atomic_bool ivox_async_running_ = {false};    // 异步加载是否正在进行
    std::atomic_bool ivox_async_ready_ = {false};      // 异步结果是否就绪
    std::shared_ptr<IVoxType> ivox_async_result_;      // 异步构建好的 IVox
};

}  // namespace lightning

#endif  // FASTER_LIO_LASER_MAPPING_H