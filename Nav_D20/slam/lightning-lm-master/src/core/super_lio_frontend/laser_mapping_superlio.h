#ifndef LIGHTNING_LASER_MAPPING_SUPERLIO_H
#define LIGHTNING_LASER_MAPPING_SUPERLIO_H

#ifndef ROOT
#define ROOT "./"
#endif

#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>
#include <deque>
#include <vector>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/measure_group.h"
#include "common/point_def.h"
#include "core/lio/pointcloud_preprocess.h"

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "lio/ESKF.h"
#include "OctVoxMap/OctVoxMap.hpp"

namespace lightning {

namespace ui {
class RvizVisualizer;
}

class LaserMappingSuperLIO {
public:
    struct Options {
        Options() {}
        bool is_in_slam_mode_ = true;
        double kf_dis_th_ = 2;
        double kf_angle_th_ = 20 * M_PI / 180.0;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LaserMappingSuperLIO(Options options = Options());
    ~LaserMappingSuperLIO();

    bool Init(const std::string& config_yaml);
    bool Run();

    void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg);
    void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg);
    void ProcessPointCloud2(CloudPtr cloud);
    void ProcessIMU(const lightning::IMUPtr& msg_in);

    void SaveMap();
    void SetUI(std::shared_ptr<ui::RvizVisualizer> ui) { ui_ = ui; }

    Keyframe::Ptr GetKeyframe() const { return last_kf_; }
    NavState GetState() const;
    NavState GetIMUState() const;
    CloudPtr GetScanUndist() const { return scan_undistort_; }
    CloudPtr GetRecentCloud();
    std::vector<Keyframe::Ptr> GetAllKeyframes() { return all_keyframes_; }
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);

private:
    bool SyncPackages();
    bool LoadParamsFromYAML(const std::string& yaml);
    void MakeKF();

    // ---- SuperLIO core algorithms ----
    bool kf_init();
    bool map_init();
    void Propagation_Undistort();
    void DownSample();
    void Observe();
    void UpdateMap();
    void DetectZUPT();

    // ---- helpers ----
    void LightningStateToSuperLIO(const NavState& ls, LI2Sup::SysState& ss) const;
    void SuperLIOStateToLightning(const LI2Sup::SysState& ss, NavState& ls) const;

private:
    Options options_;

    // ---- SuperLIO types ----
    using OctVoxMapType = LI2Sup::OctVoxMap<BASIC::V3, BASIC::scalar>;
    using KNNHeapType = LI2Sup::KNNHeap<5, BASIC::V3>;
    using V3AlignedVec = std::vector<BASIC::V3, Eigen::aligned_allocator<BASIC::V3>>;

    // ---- SuperLIO internals: ESKF + OctVoxMap ----
    LI2Sup::ESKF::Ptr kf_sl_;
    LI2Sup::ESKF::Ptr kf_sl_imu_;
    OctVoxMapType::Ptr ivox_sl_;
    pcl::VoxelGrid<PointType> voxel_scan_;

    // ---- params ----
    std::vector<double> extrinT_{3, 0.0};
    std::vector<double> extrinR_{9, 0.0};
    double filter_size_map_min_ = 0;
    double filter_size_scan_ = 0.3;
    
    // ---- IVox params ----
    double ivox_grid_resolution_ = 0.5;
    int ivox_nearby_type_ = 18;
    int ivox_hash_capacity_ = 1000000;

    // ---- point cloud preprocess ----
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;

    // ---- keyframes ----
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    // ---- point clouds ----
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr scan_down_body_{new PointCloudType()};
    CloudPtr scan_down_world_{new PointCloudType()};
    int effect_feat_num_ = 0;

    // ---- SuperLIO observe data ----
    V3AlignedVec points_body_v3_;
    V3AlignedVec points_world_v3_;
    std::vector<std::array<double, 4>> abcd_vec_;

    // ---- propagate states for undistortion ----
    struct PropagateState {
        double time = 0;
        BASIC::M3 R = BASIC::M3::Identity();
        BASIC::V3 p = BASIC::V3::Zero();
        BASIC::V3 v = BASIC::V3::Zero();
        BASIC::V3 a = BASIC::V3::Zero();
        BASIC::V3 w = BASIC::V3::Zero();
    };
    std::vector<PropagateState> propagate_states_;

    // ---- buffers & sync ----
    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_;
    std::deque<PointCloudType::Ptr> lidar_buffer_;
    std::deque<lightning::IMUPtr> imu_buffer_;

    // ---- timing ----
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    int scan_count_ = 0;

    // ---- flags ----
    bool flg_first_scan_ = true;
    double last_lidar_end_time_ = 0.0;

    // ---- SuperLIO params ----
    double gravity_norm_ = 9.7946;
    int kf_max_iterations_ = 4;
    int pub_step_ = 10;
    double kf_quit_eps_ = 1e-3;
    double imu_ng_ = 1e-5;
    double imu_na_ = 1e-2;
    double imu_nbg_ = 1e-6;
    double imu_nba_ = 1e-4;
    BASIC::M3 lidar_robo_yaw_ = BASIC::M3::Identity();
    int frame_num_ = 0;
    int imu_init_count_ = 0;
    BASIC::V3 mean_gyro_ = BASIC::V3::Zero();
    BASIC::V3 mean_acce_ = BASIC::V3::Zero();
    BASIC::SE3 sys_init_pose_;
    BASIC::SE3 lidar_to_imu_;

    // ---- state ----
    MeasureGroup measures_;
    NavState state_point_;
    NavState imu_state_;
    SO3 euler_cur_;
    Vec3d pos_lidar_;
    bool extrinsic_est_en_ = true;
    bool imu_initialized_ = false;

    // ---- ZUPT ----
    bool is_zupt_active_ = false;
    int zupt_stationary_frames_ = 0;

    std::shared_ptr<ui::RvizVisualizer> ui_ = nullptr;
};

}  // namespace lightning

#endif
