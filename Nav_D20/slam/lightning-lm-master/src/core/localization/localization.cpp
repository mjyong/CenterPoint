
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "ui/rviz_visualizer.h"

namespace lightning::loc {

// ！ 构造函数
Localization::Localization(Options options) { options_ = options; }

// ！初始化函数
bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    UL lock(global_mutex_);
    if (lio_ != nullptr || super_lio_ != nullptr) {
        // 若已经启动，则变为初始化
        Finish();
    }

    YAML_IO yaml(yaml_path);
    options_.with_ui_ = yaml.GetValue<bool>("system", "with_ui");
    options_.use_super_lio_ = yaml.GetValue<bool>("system", "use_super_lio");
    // 加载IMU达到车体的外参
    auto extrinsic_T = yaml.GetValue<std::vector<double>>("system", "extrinsic_T");
    auto extrinsic_R = yaml.GetValue<std::vector<double>>("system", "extrinsic_R");
    options_.T_body_imu_ = SE3(math::MatFromArray<double>(extrinsic_R),
                                 math::VecFromArray<double>(extrinsic_T));

    /// 创建全局分块地图（TiledMap）
    TiledMap::Options map_opt;
    map_opt.map_path_ = global_map_path;
    map_opt.enable_dynamic_polygon_ = false;
    tiled_map_ = std::make_shared<TiledMap>(map_opt);
    if (!tiled_map_->LoadMapIndex()) {
        LOG(ERROR) << "Failed to load map index: " << global_map_path;
        return false;
    }
    LOG(INFO) << "TiledMap loaded, map_path=" << global_map_path;

    /// LIO 前端
    if (options_.use_super_lio_) {
        LOG(INFO) << "Using SuperLIO frontend";
        LaserMappingSuperLIO::Options opt_super_lio;
        opt_super_lio.is_in_slam_mode_ = false;

        super_lio_ = std::make_shared<LaserMappingSuperLIO>(opt_super_lio);
        if (!super_lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init super_lio module";
            return false;
        }
    } else {
        /// lidar odom前端
        LaserMapping::Options opt_lio;
        opt_lio.is_in_slam_mode_ = false;

        lio_ = std::make_shared<LaserMapping>(opt_lio);
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio";
            return false;
        }
        // 绑定 TiledMap，LIO 内部按位姿动态加载局部地图到 IVox
        lio_->SetTiledMap(tiled_map_);
    }

    /// 输出平滑器
    smoother_ = std::make_shared<PoseSmoother>(0.1);

    /// 异步处理队列（仅保留 LIO 队列）
    options_.enable_lidar_odom_skip_ = yaml.GetValue<bool>("system", "enable_lidar_odom_skip");
    options_.lidar_odom_skip_num_ = yaml.GetValue<int>("system", "lidar_odom_skip_num");

    lidar_odom_proc_cloud_.SetMaxSize(1);
    lidar_odom_proc_cloud_.SetName("激光里程计");
    lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);
    lidar_odom_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarOdomProcCloud(cloud); });

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.Start();
    }

    /// 预处理器
    preprocess_.reset(new PointCloudPreprocess());
    std::string lio_type = "superlio";
    if (options_.use_super_lio_) {
        lio_type = "superlio";
    }else{
        lio_type = "fasterlio";
    }
    preprocess_->Blind() = yaml.GetValue<double>(lio_type, "blind");
    preprocess_->TimeScale() = yaml.GetValue<double>(lio_type, "time_scale");
    int lidar_type = yaml.GetValue<int>(lio_type, "lidar_type");
    preprocess_->NumScans() = yaml.GetValue<int>(lio_type, "scan_line");
    preprocess_->PointFilterNum() = yaml.GetValue<int>(lio_type, "point_filter_num");

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    }else if (lidar_type == 4) {
        preprocess_->SetLidarType(LidarType::XT32);
        LOG(INFO) << "Using XT32 Lidar";
    } else if (lidar_type == 5) {
        preprocess_->SetLidarType(LidarType::MID360);
        LOG(INFO) << "Using MID360 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
    }

    return true;
}

void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
    if ((lio_ == nullptr && super_lio_ == nullptr) || tiled_map_ == nullptr) {
        return;
    }
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
    if ((lio_ == nullptr && super_lio_ == nullptr) || tiled_map_ == nullptr) {
        return;
    }
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::LidarOdomProcCloud(CloudPtr cloud) {
    if (lio_ == nullptr && super_lio_ == nullptr) {
        return;
    }

    NavState lo_state;
    if (options_.use_super_lio_) {
        super_lio_->ProcessPointCloud2(cloud);
        if (!super_lio_->Run()) {
            return;
        }
        lo_state = super_lio_->GetState();
    } else {
        lio_->ProcessPointCloud2(cloud);
        if (!lio_->Run()) {
            // 检测自动重初始化：如果 INS buffer 中有 RTK_FIXED 数据，用最新 INS 位姿覆盖（跳过 NDT）
            if (lio_->WasReinitTriggered()) {
                INSPtr latest_ins = nullptr;
                if (!ins_buffer_.empty()) {
                    // 从 INS buffer 取最新的一组有效 INS 数据
                    for (auto it = ins_buffer_.rbegin(); it != ins_buffer_.rend(); ++it) {
                        if ((*it)->state == INS_VALID) {
                            latest_ins = *it;
                            break;
                        }
                    }
                }
                if (latest_ins) {
                    LOG(INFO) << "Reinit: using latest INS (RTK_FIXED) pose, skip NDT. pos="
                              << latest_ins->pose.block<3, 1>(0, 3).transpose();
                    lio_->SetInitPose(SE3(latest_ins->pose.block<3, 3>(0, 0),
                                          latest_ins->pose.block<3, 1>(0, 3)),
                                      true);
                }
                // 如果 INS 无效，不做任何事，Run() 的 init block 会用 last_good_pose_ 回退
            }
            return;
        }
        lo_state = lio_->GetState();
    }

    // INS 融合：查找时间最近的 INS 数据，作为 ESKF GPS 观测
    INSPtr nearest_ins = FindNearestINS(lo_state.timestamp_);
    if (nearest_ins && nearest_ins->state == INS_VALID) {
        if (options_.use_super_lio_) {
            // super_lio 暂不支持 ProcessINS，跳过
        } else {
            lio_->ProcessINS(nearest_ins);
            lo_state = lio_->GetState();  // 获取融合后状态
        }
        //lo_state.ins_pose_ = SE3(nearest_ins->pose.block<3,3>(0,0), nearest_ins->pose.block<3,1>(0,3));
        lo_state.ins_valid_ = true;
    } else {
        lo_state.ins_valid_ = false;
    }

    // Smoother 更新（配准时刻的位姿）
    // if (smoother_) {
    //     smoother_->PushPose(lo_state.GetPose());
    // }

    // 发布 LIO 里程计状态
    // if (lidar_odom_callback_) {
    //     lidar_odom_callback_(lo_state);
    // }

    // 发布 odom
    // if (odometry_callback_) {
    //     nav_msgs::msg::Odometry odom;
    //     odom.header.frame_id = "map";
    //     odom.header.stamp = math::FromSec(lo_state.timestamp_);
    //     odom.child_frame_id = "base_link";
    //     SE3 pose_body = lo_state.GetPose() * options_.T_body_imu_.inverse();
    //     odom.pose.pose.position.x = pose_body.translation().x();
    //     odom.pose.pose.position.y = pose_body.translation().y();
    //     odom.pose.pose.position.z = pose_body.translation().z();
    //     Eigen::Quaterniond q = pose_body.so3().unit_quaternion();
    //     odom.pose.pose.orientation.x = q.x();
    //     odom.pose.pose.orientation.y = q.y();
    //     odom.pose.pose.orientation.z = q.z();
    //     odom.pose.pose.orientation.w = q.w();
    //     odometry_callback_(odom);
    // }

    // 发布点云（世界坐标系）
    if (pointcloud_world_callback_) {
        CloudPtr scan_undistort;
        if (options_.use_super_lio_) {
            scan_undistort = super_lio_->GetScanUndist();
        } else {
            scan_undistort = lio_->GetScanUndist();
        }
        if (scan_undistort && !scan_undistort->empty()) {
            CloudPtr cloud_world(new PointCloudType);
            SE3 T_global_lidar(lo_state.offset_R_lidar_, lo_state.offset_t_lidar_);
            SE3 pose_lidar = lo_state.GetPose() * T_global_lidar;
            pcl::transformPointCloud(*scan_undistort, *cloud_world, pose_lidar.matrix().cast<float>());
            sensor_msgs::msg::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud_world, cloud_msg);
            cloud_msg.header.stamp = math::FromSec(lo_state.timestamp_);
            cloud_msg.header.frame_id = "map";
            pointcloud_world_callback_(cloud_msg);
        }
    }
}

void Localization::ProcessIMUMsg(IMUPtr imu) {
    if ((lio_ == nullptr && super_lio_ == nullptr) || tiled_map_ == nullptr) {
        return;
    }
    double this_imu_time = imu->timestamp;
    if (last_imu_time_ > 0 && this_imu_time < last_imu_time_) {
        LOG(WARNING) << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;
    }
    last_imu_time_ = this_imu_time;

    if (options_.use_super_lio_) {
        super_lio_->ProcessIMU(imu);
    } else {
        lio_->ProcessIMU(imu);
    }

    /// 获取 IMU 预测状态（高频）
    NavState dr_state;
    if (options_.use_super_lio_) {
        dr_state = super_lio_->GetIMUState();
    } else {
        dr_state = lio_->GetIMUState();
    }
    dr_state.omega_b_ = imu->angular_velocity;

    if (!dr_state.pose_is_ok_) {
        return;
    }

    // 用 IMU 自身时间戳覆盖 LIO 状态时间戳
    dr_state.timestamp_ = imu->timestamp;

    // Smoother DR 推进 + 高频平滑输出
    if (smoother_ && odometry_callback_) {
        // smoother_->PushDRPose(dr_state.GetPose());
        // SE3 smoothed_pose = smoother_->GetPose();
        NavState output = dr_state;
        // output.pos_ = smoothed_pose.translation();
        // output.rot_ = smoothed_pose.so3();

        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = "map";
        odom.header.stamp = math::FromSec(output.timestamp_);
        odom.child_frame_id = "base_link";
        SE3 transform = options_.T_body_imu_.inverse();
        SE3 pose_body = output.GetPose() * transform;
        odom.pose.pose.position.x = pose_body.translation().x();
        odom.pose.pose.position.y = pose_body.translation().y();
        odom.pose.pose.position.z = pose_body.translation().z();
        Eigen::Quaterniond q = pose_body.so3().unit_quaternion();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        // 速度: world -> IMU体坐标系 -> body系
        // v_imu = R_imu_world * v_world = R_world_imu^T * v_world
        // v_body = R_body_imu * (v_imu - omega_imu × t_body_imu)
        Vec3d vel_imu = output.rot_.inverse() * output.vel_;
        Vec3d vel_b = transform.inverse().so3() * (vel_imu - imu->angular_velocity.cross(transform.translation()));
        odom.twist.twist.linear.x = vel_b[0];
        odom.twist.twist.linear.y = vel_b[1];
        odom.twist.twist.linear.z = vel_b[2];
        // 角速度: IMU体坐标系 -> body系
        Vec3d omega_b = transform.inverse().so3() * imu->angular_velocity;
        odom.twist.twist.angular.x = omega_b[0];
        odom.twist.twist.angular.y = omega_b[1];
        odom.twist.twist.angular.z = omega_b[2];

        odometry_callback_(odom);
    }
}

void Localization::Finish() {
    if (ui_) {
        ui_->Quit();
    }
    lidar_odom_proc_cloud_.Quit();
}

void Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t,
                                   bool skip_ndt) {
    UL lock(global_mutex_);
    /// 设置 LIO 初始位姿
    if (lio_) {
        lio_->SetInitPose(SE3(q, t), skip_ndt);
    }
}
void Localization::SetOdometryCallback(OdometryCallback&& callback) { odometry_callback_ = callback; }
void Localization::SetTFCallback(Localization::TFCallback&& callback) { tf_callback_ = callback; }

void Localization::SetPointcloudWorldCallback(PointcloudWorldCallback&& callback) { pointcloud_world_callback_ = callback; }

void Localization::SetPointcloudBodyCallback(PointcloudBodyCallback&& callback) { pointcloud_body_callback_ = callback; }

void Localization::SetLidarOdomCallback(LidarOdomCallback&& callback) { lidar_odom_callback_ = callback; }

INSPtr Localization::FindNearestINS(double stamp) {
    if (ins_buffer_.empty()) {
        return nullptr;
    }
    // 使用二分查找找到第一个时间戳 >= stamp 的元素
    auto it = std::lower_bound(ins_buffer_.begin(), ins_buffer_.end(), stamp,
                               [](const INSPtr& ins, double t) { return ins->stamp < t; });
    if (it == ins_buffer_.begin()) {
        return *it;
    }
    if (it == ins_buffer_.end()) {
        return ins_buffer_.back();
    }
    double diff1 = std::abs((*it)->stamp - stamp);
    double diff2 = std::abs((*(it - 1))->stamp - stamp);
    return diff1 < diff2 ? *it : *(it - 1);
}

void Localization::RemoveOldINS(double stamp) {
    const double keep_window = 0.5;
    double threshold = stamp - keep_window;
    while (!ins_buffer_.empty() && ins_buffer_.front()->stamp < threshold) {
        ins_buffer_.pop_front();
    }
}

double Localization::GetConfidence() const {
    UL lock(global_mutex_);
    if (lio_) {
        return lio_->GetConfidence();
    }
    return 0.0;
}

bool Localization::IsNDTInited() const {
    UL lock(global_mutex_);
    if (lio_) {
        return lio_->IsNDTInited();
    }
    return false;
}

bool Localization::IsLocFailed() const {
    UL lock(global_mutex_);
    if (lio_) {
        return lio_->IsLocFailed();
    }
    return false;
}

bool Localization::WasReinitTriggered() const {
    UL lock(global_mutex_);
    if (lio_) {
        return lio_->WasReinitTriggered();
    }
    return false;
}

bool Localization::IsInitPoseSet() const {
    UL lock(global_mutex_);
    if (lio_) {
        return lio_->IsInitPoseSet();
    }
    return false;
}



bool Localization::SwitchMapAndReinit(const std::string& map_path, const SE3& init_pose) {
    UL lock(global_mutex_);
    LOG(INFO) << "SwitchMapAndReinit: map_path=" << map_path;

    // 重新构造 TiledMap（options_ 是 private，通过构造函数传入新路径）
    TiledMap::Options map_opt;
    map_opt.map_path_ = map_path;
    map_opt.enable_dynamic_polygon_ = false;
    tiled_map_ = std::make_shared<TiledMap>(map_opt);
    if (!tiled_map_->LoadMapIndex()) {
        LOG(ERROR) << "Failed to load map index: " << map_path;
        return false;
    }

    // 重新绑定 TiledMap 到 LIO，并重置所有定位状态
    if (lio_) {
        lio_->SetTiledMap(tiled_map_);
        lio_->Reinit();       // 重置 NDT/IMU/ESKF/关键帧/缓冲
        lio_->SetInitPose(init_pose);  // 设置新的先验位姿
    }

    // 重置 Smoother
    if (smoother_) {
        smoother_->Reset();
    }

    LOG(INFO) << "Map switch and reinitialization completed.";
    return true;
}

CloudPtr Localization::GetMapCloud() {
    UL lock(global_mutex_);
    if (tiled_map_) {
        auto map_cloud = tiled_map_->GetAllMap();
        if (map_cloud && !map_cloud->empty()) {
            CloudPtr filtered(new PointCloudType);
            pcl::VoxelGrid<PointType> voxel;
            auto sz = 0.5;
            voxel.setLeafSize(sz, sz, sz);
            voxel.setInputCloud(map_cloud);
            voxel.filter(*filtered);
            return filtered;
        }
    }
    return nullptr;
}

}  // namespace lightning::loc
