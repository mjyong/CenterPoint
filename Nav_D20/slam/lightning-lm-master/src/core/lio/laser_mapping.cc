#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>
#include <yaml-cpp/yaml.h>
#include <fstream>

#include "common/options.h"
#include "core/lightning_math.hpp"
#include "laser_mapping.h"
#include "ui/rviz_visualizer.h"
#include "wrapper/ros_utils.h"

namespace lightning {

bool LaserMapping::Init(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    ESKF::Options eskf_options;
    eskf_options.max_iterations_ = fasterlio::NUM_MAX_ITERATIONS;
    eskf_options.epsi_ = 1e-3 * Eigen::Matrix<double, 23, 1>::Ones();
    eskf_options.lidar_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs) { ObsModel(s, obs); };
    eskf_options.gps_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs) { GPSObsModel(s, obs); };
    eskf_options.use_aa_ = use_aa_;
    kf_.Init(eskf_options);

    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_scan;
    Vec3d lidar_T_wrt_IMU;
    Mat3d lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        fasterlio::NUM_MAX_ITERATIONS = yaml["fasterlio"]["max_iteration"].as<int>();
        fasterlio::ESTI_PLANE_THRESHOLD = yaml["fasterlio"]["esti_plane_threshold"].as<float>();

        filter_size_scan = yaml["fasterlio"]["filter_size_scan"].as<float>();
        filter_size_map_min_ = yaml["fasterlio"]["filter_size_map"].as<float>();
        keep_first_imu_estimation_ = yaml["fasterlio"]["keep_first_imu_estimation"].as<bool>();
        gyr_cov = yaml["fasterlio"]["gyr_cov"].as<float>();
        acc_cov = yaml["fasterlio"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["fasterlio"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["fasterlio"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["fasterlio"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["fasterlio"]["time_scale"].as<double>();
        lidar_type = yaml["fasterlio"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["fasterlio"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["fasterlio"]["point_filter_num"].as<int>();
        extrinsic_est_en_ = yaml["fasterlio"]["extrinsic_est_en"].as<bool>();
        extrinT_ = yaml["fasterlio"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["fasterlio"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["fasterlio"]["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["fasterlio"]["ivox_nearby_type"].as<int>();
        use_aa_ = yaml["fasterlio"]["use_aa"].as<bool>();

        skip_lidar_num_ = yaml["fasterlio"]["skip_lidar_num"].as<int>();
        enable_skip_lidar_ = skip_lidar_num_ > 0;

        // NDT初始化参数（定位模式使用）
        if (yaml["lidar_loc"]) {
            min_init_confidence_ = yaml["lidar_loc"]["min_init_confidence"].as<double>();
            lidar_loc::grid_search_angle_step = yaml["lidar_loc"]["grid_search_angle_step"].as<double>();
            lidar_loc::grid_search_angle_range = yaml["lidar_loc"]["grid_search_angle_range"].as<double>();
            LOG(INFO) << "NDT init params: min_conf=" << min_init_confidence_
                      << ", step=" << lidar_loc::grid_search_angle_step
                      << ", range=" << lidar_loc::grid_search_angle_range;
        }

    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

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
    } else if (lidar_type == 4) {
        preprocess_->SetLidarType(LidarType::XT32);
        LOG(INFO) << "Using XT32 Lidar";
    } else if (lidar_type == 5) {
        preprocess_->SetLidarType(LidarType::MID360);
        LOG(INFO) << "Using MID360 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_scan, filter_size_scan, filter_size_scan);

    lidar_T_wrt_IMU = math::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = math::MatFromArray<double>(extrinR_);
    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(Vec3d(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(Vec3d(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(Vec3d(b_acc_cov, b_acc_cov, b_acc_cov));

    return true;
}

LaserMapping::LaserMapping(Options options) : options_(options) {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

void LaserMapping::ProcessIMU(const lightning::IMUPtr &imu) {
    publish_count_++;

    double timestamp = imu->timestamp;

    UL lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    if (p_imu_->IsIMUInited()) {
        /// 更新最新imu状态
        // 2026-11-09: 加速度和 ImuProcess::UndistortPcl 一样按初始化时的重力归一化。kygi665 的 IMU 发的是 g（静止时约 1.0），
        // 原来直接用原始读数，重力状态是 9.81，两帧 LIO 之间每秒多出约 8.8 m/s^2 的下坠；
        // /lightning/odom 用的就是这个高频预测（GetIMUState），回放 0921 时 z 掉到 -46 m、速度 30 m/s。
        const Vec3d acc = imu->linear_acceleration * (G_m_s2 / p_imu_->GetMeanAccNorm());
        kf_imu_.Predict(timestamp - last_timestamp_imu_, p_imu_->Q_, imu->angular_velocity, acc);

        // LOG(INFO) << "newest wrt lidar: " << timestamp - kf_.GetX().timestamp_;

        /// 更新ui
        if (ui_) {
           ui_->UpdateNavState(kf_imu_.GetX());
           //LOG(INFO) << "--------------------kf_imu_ 速度: " << kf_imu_.GetX().vel_.transpose();
        }
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(imu);
}

bool LaserMapping::Run() {
    // 定位模式：故障检测触发的自动重新初始化
    if (!options_.is_in_slam_mode_ && need_reinit_) {
        LOG(WARNING) << "LIO: auto reinit triggered by loc fault detection.";
        Reinit();
        need_reinit_ = false;
        return false;  // 本帧不处理，下一帧重新初始化
    }

    if (!SyncPackages()) {
        return false;
    }

    // 定位模式：先做NDT位置初始化，再做IMU初


    
    // 序列: 先验位置 -> 加载先验地图 -> 多级NDT初始化 -> IMU初始化
    if (!options_.is_in_slam_mode_ && !ndt_inited_) {
        // 重初始化回退：如果上层没来得及用 INS 覆盖，使用故障前最后一个有效位姿
        if (reinit_triggered_ && !init_pose_set_by_external_) {
            LOG(WARNING) << "LIO: reinit fallback to last good pose at "
                         << last_good_pose_.translation().transpose();
            external_init_pose_ = last_good_pose_;
            init_pose_set_by_external_ = true;
            skip_ndt_init_ = false;  // 非 INS 位姿，需要 NDT 配准
            reinit_triggered_ = false;
        }

        // 1. 使用外部给的先验位姿
        // 注意: 原点(0,0,0)是合法初始位置, 只用标志位判断是否已设置
        if (!init_pose_set_by_external_) {
            LOG(ERROR) << "LIO: external init pose not set, skip. Call SetInitPose first.";
            first_lidar_time_ = measures_.lidar_end_time_;
            return false;
        }
        state_point_.SetPose(external_init_pose_);
        kf_.ChangeX(state_point_);

        // 2. 使用先验位置加载先验地图，初始化IVox
        if (tiled_map_) {
            tiled_map_->LoadOnPose(external_init_pose_);

            // 如果 LoadOnPose 没有加载到任何区块，说明 init pose 距离地图太远
            // 尝试使用地图的起始功能点作为 fallback
            if (tiled_map_->NumActiveChunks() == 0) {
                auto fps = tiled_map_->GetAllFP();
                if (!fps.empty()) {
                    LOG(WARNING) << "No map chunks loaded near init pose ("
                                 << external_init_pose_.translation().transpose()
                                 << "). Using map's start functional point as fallback: "
                                 << fps.front().pose_.translation().transpose();
                    external_init_pose_ = fps.front().pose_;
                    state_point_.SetPose(external_init_pose_);
                    kf_.ChangeX(state_point_);
                    tiled_map_->LoadOnPose(external_init_pose_);
                } else {
                    LOG(ERROR) << "No map chunks loaded and no functional points available. "
                               << "Check init_t in config or map_path.";
                }
            }

            SyncIVoxFromTiledMap();
            last_map_sync_pose_ = external_init_pose_;
            map_loaded_ = true;
        }

        // 3. 使用多级NDT初始化位置（使用原始点云，IMU未初始化无法去畸变）
        // 如果外部已指定跳过 NDT（如 INS RTK_FIXED 提供的精确位姿），则直接信任
        if (skip_ndt_init_) {
            LOG(INFO) << "NDT init skipped (trusting external pose): "
                      << external_init_pose_.translation().transpose();
            ndt_inited_ = true;
            ndt_confidence_ = 0.0;
            init_pose_restored_ = false;
            skip_ndt_init_ = false;  // 重置标志位

            first_lidar_time_ = measures_.lidar_end_time_;
            flg_first_scan_ = false;
            return true;  // NDT 跳过，下一帧开始 IMU 初始化
        }

        CloudPtr input_cloud = measures_.scan_;
        if (!input_cloud || input_cloud->empty()) {
            LOG(WARNING) << "NDT init: input cloud is empty, skip.";
            return false;
        }

        // 点云从LiDAR坐标系转换到IMU坐标系
        // p_imu = R_lidar_imu * p_lidar + t_lidar_imu
        Mat3d R_lidar_imu = math::MatFromArray<double>(extrinR_);
        Vec3d t_lidar_imu = math::VecFromArray<double>(extrinT_);
        CloudPtr input_cloud_imu(new PointCloudType);
        input_cloud_imu->reserve(input_cloud->size());
        for (const auto& pt : input_cloud->points) {
            Vec3d p_lidar(pt.x, pt.y, pt.z);
            Vec3d p_imu = R_lidar_imu * p_lidar + t_lidar_imu;
            PointType pt_new = pt;
            pt_new.x = p_imu(0);
            pt_new.y = p_imu(1);
            pt_new.z = p_imu(2);
            input_cloud_imu->push_back(pt_new);
        }
        input_cloud_imu->header = input_cloud->header;

        double fitness_score;
        SE3 ndt_pose = state_point_.GetPose();
        CloudPtr output_cloud(new PointCloudType);
        if (!YawSearch(ndt_pose, fitness_score, input_cloud_imu, output_cloud)) {
            LOG(ERROR) << "Multi-level NDT initialization failed, will retry next scan.";
            return false;  // NDT失败，下一帧重试
        }

        // 4. NDT初始化成功，更新ESKF位姿
        state_point_.SetPose(ndt_pose);
        state_point_.confidence_ = fitness_score;
        kf_.ChangeX(state_point_);
        ndt_inited_ = true;
        ndt_confidence_ = fitness_score;
        external_init_pose_ = ndt_pose;  // 保存NDT精化后的位姿，供IMU初始化后恢复
        init_pose_restored_ = false;

        LOG(INFO) << "NDT init success, pose: " << ndt_pose.translation().transpose()
                  << ", score: " << fitness_score;

        first_lidar_time_ = measures_.lidar_end_time_;
        flg_first_scan_ = false;
        return true;  // NDT初始化完成，下一帧开始IMU初始化
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);

    // IMU 初始化阶段可能覆盖NDT初始化的位姿，需要恢复
    if (!options_.is_in_slam_mode_ && !init_pose_restored_) {
        NavState s = kf_.GetX();
        if (s.pos_.norm() < 1e-3) {
            // 位姿被 IMU 初始化覆盖，恢复NDT初始化的位姿
            s.pos_ = external_init_pose_.translation();
            s.rot_ = external_init_pose_.so3();
            kf_.ChangeX(s);
            LOG(INFO) << "LIO: restored NDT init pose after IMU init: " << s.pos_.transpose();
        }
        // IMU 初始化完成后标记已恢复（仅恢复一次）
        if (p_imu_->IsIMUInited()) {
            init_pose_restored_ = true;
        }
    }

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return false;
    }
    // state_point_ = kf_.GetX();
    // LOG(INFO) << "--------------------kf_imu_ 速度: " << state_point_.vel_.transpose();
    // std::cout << "--------------------kf_imu_ 位置: " << state_point_.pos_.transpose() << std::endl;
    // std::cout << "--------------------kf_imu_ 旋转: " << state_point_.rot_.matrix().transpose() << std::endl;
    /// the first scan
    if (flg_first_scan_) {
        state_point_ = kf_.GetX();

        if (options_.is_in_slam_mode_) {
            // 建图模式：用首帧点云初始化 IVox
            scan_down_world_->resize(scan_undistort_->size());
            for (int i = 0; i < scan_undistort_->size(); i++) {
                PointBodyToWorld(scan_undistort_->points[i], scan_down_world_->points[i]);
            }
            ivox_->AddPoints(scan_down_world_->points);
        }
        // 定位模式：IVox 已在 NDT 初始化时从 TiledMap 加载，无需再初始化

        first_lidar_time_ = measures_.lidar_end_time_;
        state_point_.timestamp_ = lidar_end_time_;
        flg_first_scan_ = false;
        if (options_.is_in_slam_mode_) {
            return true;  // 建图模式首帧跳过配准
        }
        // 定位模式继续配准
    }

    if (enable_skip_lidar_) {
        skip_lidar_cnt_++;
        skip_lidar_cnt_ = skip_lidar_cnt_ % skip_lidar_num_;

        if (skip_lidar_cnt_ != 0) {
            /// 更新UI中的内容
            if (ui_) {
                ui_->UpdateNavState(kf_.GetX());
                ui_->UpdateScan(scan_undistort_, kf_.GetX().GetPose());
            }

            return false;
        }
    }
    {
        static double last_lidar_end_time_ = 0.0;
        double dt_lidar = measures_.lidar_end_time_ - last_lidar_end_time_;
        if (last_lidar_end_time_ > 0 && dt_lidar > 0) {
            double expected_imu_hz = 100.0;
            int expected_imu_count = static_cast<int>(dt_lidar * expected_imu_hz);
            int actual_imu_count = static_cast<int>(measures_.imu_.size());
            int imu_drop = expected_imu_count - actual_imu_count;
            if (imu_drop > 0) {
                LOG(WARNING) << "[ SuperLIO ] IMU drop: " << imu_drop
                             << " (expected: " << expected_imu_count
                             << ", actual: " << actual_imu_count
                             << ", dt_lidar: " << std::fixed << std::setprecision(3) << dt_lidar << "s)";
            }
        }
        last_lidar_end_time_ = measures_.lidar_end_time_;
    }




    // LOG(INFO) << "LIO get cloud at beg: " << std::setprecision(14) << measures_.lidar_begin_time_
    //           << ", end: " << measures_.lidar_end_time_;

    if (last_lidar_time_ > 0 && (measures_.lidar_begin_time_ - last_lidar_time_) > 0.5) {
        LOG(ERROR) << "检测到雷达断流，时长：" << (measures_.lidar_begin_time_ - last_lidar_time_);
    }

    last_lidar_time_ = measures_.lidar_begin_time_;

    flg_EKF_inited_ = (measures_.lidar_begin_time_ - first_lidar_time_) >= fasterlio::INIT_TIME;

    /// downsample
    voxel_scan_.setInputCloud(scan_undistort_);
    voxel_scan_.filter(*scan_down_body_);

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return false;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);

    // 定位模式：根据当前位姿滚动加载 TiledMap 局部地图，异步重建 IVox
    if (!options_.is_in_slam_mode_ && tiled_map_) {
        // 先检查异步加载是否完成，完成则替换 IVox
        if (CheckAsyncIVoxReady()) {
            last_map_sync_pose_ = kf_.GetX().GetPose();
            LOG(INFO) << "Async IVox replaced successfully.";
        }

        // 判断是否需要触发新的异步加载（异步未运行时才触发）
        if (!ivox_async_running_) {
            SE3 current_pose = kf_.GetX().GetPose();
            if (last_map_sync_pose_.translation().norm() < 1e-3 ||
                (current_pose.translation() - last_map_sync_pose_.translation()).norm() > map_update_dist_) {
                tiled_map_->LoadOnPose(current_pose);
                AsyncSyncIVoxFromTiledMap();
                // 临时更新 last_map_sync_pose_ 避免重复触发，异步完成后会再次更新
                last_map_sync_pose_ = current_pose;
            }
        }
    }

    Timer::Evaluate(
        [&, this]() {
            // 成员变量预分配
            residuals_.resize(cur_pts, 0);
            point_selected_surf_.resize(cur_pts, true);
            plane_coef_.resize(cur_pts, Vec4f::Zero());

            auto old_state = kf_.GetX();

            kf_.Update(ESKF::ObsType::LIDAR, 1e-3);
            state_point_ = kf_.GetX();

            if (keep_first_imu_estimation_ && all_keyframes_.size() < 5 &&
                (old_state.rot_.inverse() * state_point_.rot_).log().norm() > 0.3 * M_PI / 180) {
                kf_.ChangeX(old_state);
                state_point_ = old_state;

                LOG(INFO) << "set state as prediction";
            }

            // LOG(INFO) << "old yaw: " << old_state.rot_.angleZ() << ", new: " << state_point_.rot_.angleZ();

            state_point_.timestamp_ = measures_.lidar_end_time_;
            euler_cur_ = state_point_.rot_;
            pos_lidar_ = state_point_.pos_ + state_point_.rot_ * state_point_.offset_t_lidar_;
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");
    state_point_ = kf_.GetX();
    if(measures_.imu_.size()>0)
        state_point_.omega_b_ = measures_.imu_.back()->angular_velocity;
    // LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " down " << cur_pts
    //           << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    // 定位模式：基于 effect num 和平均残差判断定位状态
    if (!options_.is_in_slam_mode_) {
        double avg_residual = 0.0;
        if (effect_feat_num_ > 0) {
            double sum = 0.0;
            for (int i = 0; i < effect_feat_num_; i++) {
                sum += std::abs(corr_pts_[i][3]);
            }
            avg_residual = sum / effect_feat_num_;
        }

        // 定位状态判断
        bool fail_this_frame = (effect_feat_num_ < min_effect_feat_) || (avg_residual > max_avg_residual_);
        if (fail_this_frame) {
            loc_fail_count_++;
            if (loc_fail_count_ >= max_fail_count_) {
                loc_failed_ = true;
                need_reinit_ = true;  // 标记需要重新初始化，下一帧 Run() 开头执行
                LOG(ERROR) << "定位故障! 触发自动重初始化. effect_num=" << effect_feat_num_
                           << " avg_residual=" << avg_residual
                           << " fail_count=" << loc_fail_count_;
            } else {
                LOG(WARNING) << "定位质量下降: effect_num=" << effect_feat_num_
                             << " avg_residual=" << avg_residual
                             << " fail_count=" << loc_fail_count_ << "/" << max_fail_count_;
            }
        } else {
            if (loc_failed_ || loc_fail_count_ > 0) {
                LOG(INFO) << "定位质量恢复: effect_num=" << effect_feat_num_
                          << " avg_residual=" << avg_residual;
            }
            last_good_pose_ = state_point_.GetPose();
            loc_fail_count_ = 0;
            loc_failed_ = false;
        }
    }
    LOG(INFO) << "time: " << std::setprecision(14)<< state_point_.timestamp_<< ", state: " << state_point_.pos_.transpose() ;
    /// keyframes
    if (last_kf_ == nullptr) {
        MakeKF();
    } else {
        SE3 last_pose = last_kf_->GetLIOPose();
        SE3 cur_pose = state_point_.GetPose();
        // if ((last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_ ||
        //     (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_) {
        //     MakeKF();
        // } else if (!options_.is_in_slam_mode_ && (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 2.0) {
        //     MakeKF();
        // }
        bool dis_exceed = (last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_;
        bool angle_exceed = (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_;
        // 非建图模式下, 距上次关键帧超过 3 秒, 无论是否移动超过阈值都强制生成
        bool time_exceed = !options_.is_in_slam_mode_ &&
                           (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 3.0;
        if (dis_exceed || angle_exceed || time_exceed) {
            MakeKF();
        }
    }

    /// 更新kf_for_imu
    kf_imu_ = kf_;
    // if (!measures_.imu_.empty()) {
    //     double t = measures_.imu_.back()->timestamp;
    //     for (auto &imu : imu_buffer_) {
    //         double dt = imu->timestamp - t;
    //         kf_imu_.Predict(dt, p_imu_->Q_, imu->angular_velocity, imu->linear_acceleration);
    //         t = imu->timestamp;
    //     }
    // }

    /// 更新ui
    // if (ui_) {
    //     ui_->UpdateNavState(kf_imu_.GetX());
    //     //LOG(INFO) << "--------------------kf_imu_ 速度: " << kf_imu_.GetX().vel_.transpose();
    // }
    if (ui_) {
        // ui_->UpdateNavState(kf_.GetX());
        SE3 pose_imu = state_point_.GetPose()*SE3(state_point_.offset_R_lidar_,state_point_.offset_t_lidar_);
        ui_->UpdateScan(scan_undistort_, pose_imu);
    }

    return true;
}

void LaserMapping::MakeKF() {
    CloudPtr scan_cloud_imu_{new PointCloudType()};  // downsampled scan in imu
    for (int i = 0; i < scan_undistort_->size(); i++) {
        Vec3d p_global((state_point_.offset_R_lidar_ * scan_undistort_->points[i].getVector3fMap().cast<double>() +
                                            state_point_.offset_t_lidar_));
        PointType po;
        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.intensity = scan_undistort_->points[i].intensity;
        scan_cloud_imu_->push_back(po);

    }
    scan_cloud_imu_->header = scan_undistort_->header;
    //Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_undistort_, state_point_);
    Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_cloud_imu_, state_point_);
    if (last_kf_) {
        // LOG(INFO) << "last kf lio: " << last_kf_->GetLIOPose().translation().transpose()
        //           << ", opt: " << last_kf_->GetOptPose().translation().transpose();

        /// opt pose 用之前的递推
        SE3 delta = last_kf_->GetLIOPose().inverse() * kf->GetLIOPose();
        kf->SetOptPose(last_kf_->GetOptPose() * delta);
    } else {
        kf->SetOptPose(kf->GetLIOPose());
    }

    kf->SetState(state_point_);

    // LOG(INFO) << "LIO: create kf " << kf->GetID() << ", state: " << state_point_.pos_.transpose()
    //           << ", kf opt pose: " << kf->GetOptPose().translation().transpose()
    //           << ", lio pose: " << kf->GetLIOPose().translation().transpose() << ", time: " << std::setprecision(14)
    //           << state_point_.timestamp_;

    if (options_.is_in_slam_mode_) {
        all_keyframes_.emplace_back(kf);
    }

    last_kf_ = kf;
}

void LaserMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
                      << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            // LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
            //           << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(CloudPtr cloud) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;

            double timestamp = math::ToSec(cloud->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.scan_ = lidar_buffer_.front();
        measures_.lidar_begin_time_ = time_buffer_.front();

        if (measures_.scan_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else if (measures_.scan_->points.back().time / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_begin_time_ + measures_.scan_->points.back().time / double(1000);
            lidar_mean_scantime_ +=
                (measures_.scan_->points.back().time / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        lo::lidar_time_interval = lidar_mean_scantime_;

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->timestamp;
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) {
            break;
        }

        measures_.imu_.push_back(imu_buffer_.front());

        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    // LOG(INFO) << "sync: " << std::setprecision(14) << measures_.lidar_begin_time_ << ", " <<
    // measures_.lidar_end_time_;

    return true;
}

void LaserMapping::MapIncremental() {
    // 建图模式：用观测点云增量构建地图
    // 定位模式：不增量，IVox 由 TiledMap 动态提供
    if (!options_.is_in_slam_mode_) {
        return;
    }

    PointVector points_to_add;
    PointVector point_no_need_downsample;

    size_t cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(index.begin(), index.end(), [&](const size_t &i) {
        PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);

        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = math::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= fasterlio::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < fasterlio::NUM_MATCH_POINTS; readd_i++) {
                    if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }

            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

void LaserMapping::SyncIVoxFromTiledMap() {
    auto t_start = std::chrono::high_resolution_clock::now();
    // 方案A：重建 IVox（不改 IVox 源码，利用构造函数清空）
    ivox_ = std::make_shared<IVoxType>(ivox_options_);
    // 灌入 TiledMap 当前已加载的局部地图
    if (tiled_map_) {
        auto map_cloud = tiled_map_->GetAllMap();
        if (map_cloud && !map_cloud->empty()) {
            // 下采样降低地图密度，减少配准计算量
            CloudPtr filtered(new PointCloudType);
            pcl::VoxelGrid<PointType> voxel;
            voxel.setLeafSize(map_downsample_res_, map_downsample_res_, map_downsample_res_);
            voxel.setInputCloud(map_cloud);
            voxel.filter(*filtered);
            ivox_->AddPoints(filtered->points);
            auto t_end = std::chrono::high_resolution_clock::now();
            double cost_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            LOG(ERROR) << "IVox rebuilt from TiledMap, raw: " << map_cloud->size()
                      << " downsampled: " << filtered->size()
                      << " (res=" << map_downsample_res_ << ")"
                      << " cost: " << cost_ms << " ms";
        } else {
            LOG(WARNING) << "TiledMap returns empty cloud during IVox sync";
        }
    }
}

void LaserMapping::AsyncSyncIVoxFromTiledMap() {
    if (ivox_async_running_) {
        LOG(INFO) << "Async IVox sync already running, skip.";
        return;
    }

    // 确保之前的线程已结束
    if (ivox_async_thread_.joinable()) {
        ivox_async_thread_.join();
    }

    ivox_async_running_ = true;
    ivox_async_ready_ = false;

    ivox_async_thread_ = std::thread([this]() {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 在后台线程中重建 IVox
        auto new_ivox = std::make_shared<IVoxType>(ivox_options_);

        if (tiled_map_) {
            auto map_cloud = tiled_map_->GetAllMap();
            if (map_cloud && !map_cloud->empty()) {
                CloudPtr filtered(new PointCloudType);
                pcl::VoxelGrid<PointType> voxel;
                voxel.setLeafSize(map_downsample_res_, map_downsample_res_, map_downsample_res_);
                voxel.setInputCloud(map_cloud);
                voxel.filter(*filtered);
                new_ivox->AddPoints(filtered->points);
                auto t_end = std::chrono::high_resolution_clock::now();
                double cost_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                LOG(ERROR) << "Async IVox rebuilt, raw: " << map_cloud->size()
                          << " downsampled: " << filtered->size()
                          << " (res=" << map_downsample_res_ << ")"
                          << " cost: " << cost_ms << " ms";
                if (cost_ms > 500.0) {
                    LOG(ERROR) << "IVox rebuild too slow: " << cost_ms << " ms, may block localization!";
                }
            } else {
                LOG(WARNING) << "TiledMap returns empty cloud during async IVox sync";
            }
        }

        {
            std::lock_guard<std::mutex> lock(ivox_async_mutex_);
            ivox_async_result_ = new_ivox;
        }
        ivox_async_ready_ = true;
        ivox_async_running_ = false;
    });
}

bool LaserMapping::CheckAsyncIVoxReady() {
    if (!ivox_async_ready_) {
        return false;
    }

    // 确保线程已结束
    if (ivox_async_thread_.joinable()) {
        ivox_async_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(ivox_async_mutex_);
        if (ivox_async_result_) {
            ivox_ = ivox_async_result_;
            ivox_async_result_.reset();
        }
    }
    ivox_async_ready_ = false;
    return true;
}

void LaserMapping::SetInitPose(const SE3& pose, bool skip_ndt) {
    NavState s = kf_.GetX();
    s.pos_ = pose.translation();
    s.rot_ = pose.so3();
    kf_.ChangeX(s);
    external_init_pose_ = pose;
    init_pose_set_by_external_ = true;
    init_pose_restored_ = false;
    skip_ndt_init_ = skip_ndt;
    ndt_inited_ = false;  // 重置NDT初始化状态，等待重新初始化
    reinit_triggered_ = false;  // 上层已提供新位姿，清除重初始化标志
    LOG(INFO) << "LIO init pose set" << (skip_ndt ? " (skip NDT)" : "")
              << ": " << pose.translation().transpose()
              << ", rot: " << pose.so3().unit_quaternion().coeffs().transpose();
}

void LaserMapping::Reinit() {
    LOG(INFO) << "LIO Reinit: resetting all localization states.";

    // 保存故障前最后一个有效位姿，作为重初始化的回退位姿
    last_good_pose_ = state_point_.GetPose();
    reinit_triggered_ = true;
    LOG(INFO) << "LIO Reinit: saved last good pose at "
              << last_good_pose_.translation().transpose();

    // 等待异步加载线程结束
    if (ivox_async_thread_.joinable()) {
        ivox_async_thread_.join();
    }
    ivox_async_running_ = false;
    ivox_async_ready_ = false;

    // 重置 NDT/IMU 初始化标志
    ndt_inited_ = false;
    init_pose_restored_ = false;
    ndt_confidence_ = 0.0;
    // 重置首帧标志，让 Run() 重新走初始化流程
    flg_first_scan_ = true;
    map_loaded_ = false;
    // 清掉外部位姿标志，让上层有机会用 INS 最新数据覆盖
    init_pose_set_by_external_ = false;
    skip_ndt_init_ = false;

    // 重置故障检测状态
    loc_fail_count_ = 0;
    loc_failed_ = false;
    need_reinit_ = false;

    // 重置 IMU 处理器，使其重新初始化（重力、零偏）
    p_imu_->Reset();

    // 清空关键帧
    all_keyframes_.clear();
    last_kf_ = nullptr;
    kf_id_ = 0;

    // 重置 ESKF 状态（使用故障前有效位姿，由后续 Run() init block 或上层 SetInitPose 覆盖）
    NavState s;
    s.SetPose(last_good_pose_);
    kf_.ChangeX(s);
    state_point_ = s;

    // 清空缓冲
    UL lock(mtx_buffer_);
    lidar_buffer_.clear();
    time_buffer_.clear();
    imu_buffer_.clear();
    lidar_pushed_ = false;
}

bool LaserMapping::YawSearch(SE3& pose, double& confidence, CloudPtr input, CloudPtr output) {
    SE3 init_pose = pose;
    auto RPYXYZ = math::SE3ToRollPitchYaw(init_pose);
    double init_yaw = RPYXYZ.yaw;

    confidence = 0;
    bool yaw_search_success = false;

    // 获取地图点云
    if (!tiled_map_) {
        LOG(ERROR) << "TiledMap is null, cannot perform YawSearch";
        return false;
    }
    CloudPtr map_cloud = tiled_map_->GetAllMap();
    if (map_cloud->empty()) {
        LOG(WARNING) << "Map cloud is empty, cannot perform YawSearch";
        return false;
    }

    int step = lidar_loc::grid_search_angle_step;
    double radius = lidar_loc::grid_search_angle_range * constant::kDEG2RAD;
    double angle_search_step = 2 * radius / step;

    std::vector<double> searched_yaw;
    std::vector<double> scores(step);
    std::vector<SE3> pose_opti(step);

    for (int i = 0; i < step; ++i) {
        double search_yaw = init_yaw + i * angle_search_step - radius;
        searched_yaw.emplace_back(search_yaw);
    }

    LOG(INFO) << "YawSearch: init yaw: " << init_yaw << ", p: " << RPYXYZ.pitch
              << ", ro: " << RPYXYZ.roll << ", search from "
              << searched_yaw.front() << " to " << searched_yaw.back()
              << ", step=" << step;

    /// 使用多分辨率NDT进行角度搜索 - 并行计算
    #pragma omp parallel for
    for (int i = 0; i < step; ++i) {
        double fitness_score = 0;
        PoseRPYD local_RPYXYZ = RPYXYZ;
        local_RPYXYZ.yaw = searched_yaw[i];
        SE3 pose_esti = math::XYZRPYToSE3(local_RPYXYZ);

        // 多分辨率NDT参数
        std::vector<double> res{10.0, 5.0, 4.0, 3.0};
        Eigen::Matrix4f Tw2 = pose_esti.matrix().cast<float>();

        for (auto& r : res) {
            pcl::NormalDistributionsTransform<PointType, PointType> ndt;
            ndt.setTransformationEpsilon(0.05);
            ndt.setStepSize(0.7);
            ndt.setMaximumIterations(40);
            ndt.setResolution(r);

            // 对地图和输入点云进行体素滤波
            CloudPtr rough_map = math::VoxelGrid(map_cloud, r * 0.1);
            CloudPtr rough_input = math::VoxelGrid(input, r * 0.1);

            ndt.setInputTarget(rough_map);
            ndt.setInputSource(rough_input);

            // 每个线程创建自己的 output cloud 避免竞争
            PointCloudType local_output;
            ndt.align(local_output, Tw2);
            Tw2 = ndt.getFinalTransformation();

            fitness_score = ndt.getTransformationProbability();
        }

        // 转换回SE3
        Eigen::Matrix3d rot = Tw2.block<3, 3>(0, 0).cast<double>();
        Quatd q_3d = Quatd(rot);
        Vec3d t_3d = Tw2.block<3, 1>(0, 3).cast<double>();
        q_3d.normalize();

        pose_esti = SE3(q_3d, t_3d);

        scores[i] = fitness_score;
        pose_opti[i] = pose_esti;
    }

    // find best match
    auto best_score_idx = std::max_element(scores.begin(), scores.end()) - scores.begin();
    confidence = scores.at(best_score_idx);
    pose = pose_opti.at(best_score_idx);
    auto RPYXYZ_ = math::SE3ToRollPitchYaw(pose);

    LOG(INFO) << "YawSearch best confidence: " << confidence
              << ", yaw:" << RPYXYZ_.yaw * 180.0 / M_PI;

    if (confidence > min_init_confidence_) {
        LOG(INFO) << "YawSearch success, score: " << confidence
                  << ", th=" << min_init_confidence_
                  << ", pose: " << pose.translation().transpose();
        yaw_search_success = true;
    } else {
        LOG(WARNING) << "YawSearch failed, score: " << confidence
                     << ", th=" << min_init_confidence_;
    }

    return yaw_search_success;
}

void LaserMapping::ProcessINS(const INSPtr& ins) {
    //return;
    if (!ins || ins->state != INS_VALID) return;
    // 存储 INS 位姿和协方差，供 GPSObsModel 使用
    ins_pose_ = SE3(ins->pose.block<3, 3>(0, 0), ins->pose.block<3, 1>(0, 3));
    ins_pos_cov_ = ins->position_covariance;
    ins_rot_cov_ = ins->orientation_covariance;
    ins_obs_ready_ = true;
    // ESKF GPS 观测更新
    kf_.Update(ESKF::ObsType::GPS, 1.0);
    state_point_ = kf_.GetX();
    ins_obs_ready_ = false;
}

void LaserMapping::GPSObsModel(NavState& s, ESKF::CustomObservationModel& obs) {
    if (!ins_obs_ready_) {
        obs.valid_ = false;
        return;
    }
    // 残差 = INS观测位姿 - ESKF预测位姿
    obs.residual_.resize(6);
    obs.residual_.head<3>() = ins_pose_.translation() - s.pos_;
    obs.residual_.tail<3>() = (s.rot_.inverse() * ins_pose_.so3()).log();

    // 雅可比矩阵：位置对位置为单位阵，旋转为右扰动（近似单位阵）
    constexpr int state_dim = ESKF::state_dim_;
    obs.h_x_.resize(6, state_dim);
    obs.h_x_.setZero();
    obs.h_x_.block<3, 3>(0, 0) = Mat3d::Identity();  // dr_pos / d_pos
    obs.h_x_.block<3, 3>(3, 3) = Mat3d::Identity();  // dr_rot / d_rot

    // 观测协方差
    obs.R_.resize(6, 6);
    obs.R_.setZero();
    obs.R_.block<3, 3>(0, 0) = ins_pos_cov_.asDiagonal();
    obs.R_.block<3, 3>(3, 3) = ins_rot_cov_.asDiagonal();
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(NavState &s, ESKF::CustomObservationModel &obs) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot_ * s.offset_R_lidar_).cast<float>();
            auto t_wl = (s.rot_ * s.offset_t_lidar_ + s.pos_).cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                Vec3f p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                points_near.clear();

                /** Find the closest surfaces in the map **/
                // if (obs.converge_) {
                ivox_->GetClosestPoint(point_world, points_near, fasterlio::NUM_MATCH_POINTS);
                point_selected_surf_[i] = points_near.size() >= fasterlio::MIN_NUM_MATCH_POINTS;
                if (point_selected_surf_[i]) {
                    point_selected_surf_[i] =
                        math::esti_plane(plane_coef_[i], points_near, fasterlio::ESTI_PLANE_THRESHOLD);
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        obs.valid_ = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            obs.h_x_ = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            obs.residual_.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const Mat3f off_R = s.offset_R_lidar_.matrix().cast<float>();
            const Vec3f off_t = s.offset_t_lidar_.cast<float>();
            const Mat3f Rt = s.rot_.matrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Vec3f point_this_be = corr_pts_[i].head<3>();
                Mat3f point_be_crossmat = math::SKEW_SYM_MATRIX(point_this_be);
                Vec3f point_this = off_R * point_this_be + off_t;
                Mat3f point_crossmat = math::SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                Vec3f norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                Vec3f C(Rt * norm_vec);
                Vec3f A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    Vec3f B(point_be_crossmat * off_R.transpose() * C);
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0], B[1],
                        B[2], C[0], C[1], C[2];
                } else {
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0;
                }

                /// 增加了cauchy's robust kernel
                float res = -corr_pts_[i][3];
                float rho, drho;

                const float delta = 2.0;
                const float dsqr = delta * delta;
                const float dsqr_inv = 1.0 / dsqr;

                if (res >= 0) {
                    rho = dsqr * std::log(1 + res * dsqr_inv);
                    drho = 1.0 / (1 + res * dsqr_inv);
                } else {
                    rho = -dsqr * std::log(1 - res * dsqr_inv);
                    drho = 1.0 / (1 - res * dsqr_inv);
                }

                obs.residual_(i) = rho;
                obs.h_x_.block<1, 12>(i, 0) = obs.h_x_.block<1, 12>(i, 0).eval() * drho;

                // obs.residual_(i) = res;
            });
        },
        "    ObsModel (IEKF Build Jacobian)");

    /// 填入中位数平方误差
    std::vector<double> res_sq2;
    for (size_t i = 0; i < cnt_pts; ++i) {
        if (point_selected_surf_[i]) {
            double r = residuals_[i];
            res_sq2.emplace_back(r * r);
        }
    }

    std::sort(res_sq2.begin(), res_sq2.end());
    obs.lidar_residual_mean_ = res_sq2[res_sq2.size() / 2];
    obs.lidar_residual_max_ = res_sq2[res_sq2.size() - 1];
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////

CloudPtr LaserMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto &kf : all_keyframes_) {
        CloudPtr cloud = kf->GetCloud();

        CloudPtr cloud_filter(new PointCloudType);

        if (use_voxel) {
            voxel.setInputCloud(cloud);
            voxel.filter(*cloud_filter);

        } else {
            cloud_filter = cloud;
        }

        CloudPtr cloud_trans(new PointCloudType);

        if (use_lio_pose) {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        } else {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        }

        *global_map += *cloud_trans;

        LOG(INFO) << "kf " << kf->GetID() << ", pose: " << kf->GetOptPose().translation().transpose();
    }

    CloudPtr global_map_filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*global_map_filtered);
    } else {
        global_map_filtered = global_map;
    }

    global_map_filtered->is_dense = false;
    global_map_filtered->height = 1;
    global_map_filtered->width = global_map_filtered->size();

    LOG(INFO) << "global map: " << global_map_filtered->size();

    return global_map_filtered;
}

void LaserMapping::SaveMap() {
    /// 保存地图
    auto global_map = GetGlobalMap(true);

    pcl::io::savePCDFileBinaryCompressed("./data/lio.pcd", *global_map);

    LOG(INFO) << "lio map is saved to ./data/lio.pcd";
}

CloudPtr LaserMapping::GetRecentCloud() {
    if (lidar_buffer_.empty()) {
        return nullptr;
    }

    return lidar_buffer_.front();
}

}  // namespace lightning