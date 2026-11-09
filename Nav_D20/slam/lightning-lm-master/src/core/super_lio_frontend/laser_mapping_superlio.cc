#ifndef VELODYNE_ROS_POINT_DEFINED
#define VELODYNE_ROS_POINT_DEFINED
#endif
#ifndef HESAI_ROS_POINT_DEFINED
#define HESAI_ROS_POINT_DEFINED
#endif
#ifndef OUSTER_ROS_POINT_DEFINED
#define OUSTER_ROS_POINT_DEFINED
#endif
#ifndef LIVOX_POINT_XYZRTLT_DEFINED
#define LIVOX_POINT_XYZRTLT_DEFINED
#endif
#ifndef RSLIDAR_ROS_POINT_DEFINED
#define RSLIDAR_ROS_POINT_DEFINED
#endif

#ifndef ROOT
#define ROOT "./"
#endif

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

#include "common/options.h"
#include "core/lightning_math.hpp"
#include "core/super_lio_frontend/laser_mapping_superlio.h"
#include "ui/rviz_visualizer.h"
#include "wrapper/ros_utils.h"

#include "basic/alias.h"

namespace lightning {

LaserMappingSuperLIO::LaserMappingSuperLIO(Options options) : options_(options) {
    preprocess_.reset(new PointCloudPreprocess());
}

LaserMappingSuperLIO::~LaserMappingSuperLIO() {
    scan_down_body_ = nullptr;
    scan_undistort_ = nullptr;
    scan_down_world_ = nullptr;
    LOG(INFO) << "LaserMappingSuperLIO deconstruct";
}

bool LaserMappingSuperLIO::Init(const std::string& config_yaml) {
    LOG(INFO) << "init LaserMappingSuperLIO from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    kf_sl_.reset(new LI2Sup::ESKF());
    kf_sl_imu_.reset(new LI2Sup::ESKF());
    ivox_sl_.reset(new OctVoxMapType(OctVoxMapType::Options{static_cast<float>(ivox_grid_resolution_), ivox_hash_capacity_}));
    voxel_scan_.setLeafSize(filter_size_scan_, filter_size_scan_, filter_size_scan_);

    scan_undistort_.reset(new PointCloudType());
    scan_down_body_.reset(new PointCloudType());
    scan_down_world_.reset(new PointCloudType());

    LOG(INFO) << "LaserMappingSuperLIO init done";
    return true;
}

bool LaserMappingSuperLIO::LoadParamsFromYAML(const std::string& yaml_file) {
    int lidar_type;
    double filter_size_scan;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        filter_size_scan = yaml["superlio"]["filter_size_scan"].as<float>();
        filter_size_map_min_ = yaml["superlio"]["filter_size_map"].as<float>();
        extrinT_ = yaml["superlio"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["superlio"]["extrinsic_R"].as<std::vector<double>>();
        lidar_type = yaml["superlio"]["lidar_type"].as<int>();
        extrinsic_est_en_ = yaml["superlio"]["extrinsic_est_en"].as<bool>();
        kf_max_iterations_ = yaml["superlio"]["max_iteration"].as<int>();

        imu_ng_ = yaml["superlio"]["gyr_cov"].as<double>();
        imu_na_ = yaml["superlio"]["acc_cov"].as<double>();
        imu_nbg_ = yaml["superlio"]["b_gyr_cov"].as<double>();
        imu_nba_ = yaml["superlio"]["b_acc_cov"].as<double>();

        preprocess_->Blind() = yaml["superlio"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["superlio"]["time_scale"].as<double>();
        preprocess_->NumScans() = yaml["superlio"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["superlio"]["point_filter_num"].as<int>();

        ivox_grid_resolution_ = yaml["superlio"]["ivox_grid_resolution"].as<double>();
        ivox_nearby_type_ = yaml["superlio"]["ivox_nearby_type"].as<int>();
        ivox_hash_capacity_ = yaml["superlio"]["hash_capacity"].as<int>();

        pub_step_ = yaml["superlio"]["pub_step"].as<int>();

        if (yaml["superlio"]["gravity_norm"]) {
            gravity_norm_ = yaml["superlio"]["gravity_norm"].as<double>();
        }
    } catch (const YAML::Exception& e) {
        LOG(ERROR) << "YAML conversion error: " << e.what();
        return false;
    }

    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
    } else if (lidar_type == 4) {
        preprocess_->SetLidarType(LidarType::XT32);
    } else if (lidar_type == 5) {
        preprocess_->SetLidarType(LidarType::MID360);
    } else if (lidar_type == 6) {
        preprocess_->SetLidarType(LidarType::RSLIDAR);
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    lidar_to_imu_.R_ = math::MatFromArray<double>(extrinR_).cast<BASIC::scalar>();
    lidar_to_imu_.t_ = math::VecFromArray<double>(extrinT_).cast<BASIC::scalar>();

    filter_size_scan_ = filter_size_scan;
    voxel_scan_.setLeafSize(filter_size_scan_, filter_size_scan_, filter_size_scan_);

    return true;
}

void LaserMappingSuperLIO::ProcessIMU(const lightning::IMUPtr& imu) {
    double timestamp = imu->timestamp;
    UL lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    if (imu_initialized_) {
        double dt = timestamp - last_timestamp_imu_;
        if (dt > 0) {
            LI2Sup::IMUData imu_sl;
            imu_sl.secs = timestamp;
            imu_sl.gyr = imu->angular_velocity.cast<BASIC::scalar>();
            imu_sl.acc = imu->linear_acceleration.cast<BASIC::scalar>();

            kf_sl_imu_->Predict(imu_sl,dt);

            auto sys_state = kf_sl_imu_->GetSysState();

            SuperLIOStateToLightning(sys_state, imu_state_);

            // if (ui_) {
            //     ui_->UpdateNavState(imu_state_);
            // }
        }
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.emplace_back(imu);
}

void LaserMappingSuperLIO::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    UL lock(mtx_buffer_);
    scan_count_++;
    double timestamp = ToSec(msg->header.stamp);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
        lidar_buffer_.clear();
    }
    CloudPtr cloud(new PointCloudType());
    preprocess_->Process(msg, cloud);
    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp);
    last_timestamp_lidar_ = timestamp;
}

void LaserMappingSuperLIO::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg) {
    UL lock(mtx_buffer_);
    scan_count_++;
    double timestamp = ToSec(msg->header.stamp);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
        lidar_buffer_.clear();
    }
    CloudPtr cloud(new PointCloudType());
    preprocess_->Process(msg, cloud);
    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp);
    last_timestamp_lidar_ = timestamp;
}

void LaserMappingSuperLIO::ProcessPointCloud2(CloudPtr cloud) {
    UL lock(mtx_buffer_);
    scan_count_++;
    double timestamp = math::ToSec(cloud->header.stamp);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
        lidar_buffer_.clear();
    }
    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp);
    last_timestamp_lidar_ = timestamp;
}

bool LaserMappingSuperLIO::SyncPackages() {
    UL lock(mtx_buffer_);
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }
    if (!lidar_pushed_) {
        measures_.scan_ = lidar_buffer_.front();
        measures_.lidar_begin_time_ = time_buffer_.front();

        if (measures_.scan_->points.size() <= 1) {
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else if (measures_.scan_->points.back().time / 1000.0 < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_begin_time_ + measures_.scan_->points.back().time / 1000.0;
            lidar_mean_scantime_ +=
                (measures_.scan_->points.back().time / 1000.0 - lidar_mean_scantime_) / scan_num_;
        }
        lo::lidar_time_interval = lidar_mean_scantime_;
        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }
    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }
    double imu_time = imu_buffer_.front()->timestamp;
    measures_.imu_.clear();
    while (!imu_buffer_.empty() && imu_time < lidar_end_time_) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) break;
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

// ===== SuperLIO State conversion =====

void LaserMappingSuperLIO::LightningStateToSuperLIO(const NavState& ls, LI2Sup::SysState& ss) const {
    ss.timestamp = ls.timestamp_;
    Eigen::Matrix<BASIC::scalar, 3, 3> rot_mat = ls.rot_.matrix().cast<BASIC::scalar>();
    ss.R = BASIC::SO3(rot_mat);
    ss.p = ls.pos_.cast<BASIC::scalar>();
    ss.v = ls.vel_.cast<BASIC::scalar>();
    ss.bg = ls.bg_.cast<BASIC::scalar>();
    ss.ba = ls.ba_.cast<BASIC::scalar>();
}

void LaserMappingSuperLIO::SuperLIOStateToLightning(const LI2Sup::SysState& ss, NavState& ls) const {
    ls.timestamp_ = ss.timestamp;
    ls.pos_ = ss.p.cast<double>();
    Eigen::Matrix3d rot_mat = ss.R.matrix().cast<double>();
    ls.rot_ = SO3(Eigen::Quaterniond(rot_mat).normalized());
    ls.vel_ = ss.v.cast<double>();
    ls.bg_ = ss.bg.cast<double>();
    ls.ba_ = ss.ba.cast<double>();
    Eigen::Matrix3d extrinR_mat = math::MatFromArray<double>(extrinR_);
    ls.offset_R_lidar_ = SO3(Eigen::Quaterniond(extrinR_mat).normalized());
    ls.offset_t_lidar_ = math::VecFromArray<double>(extrinT_);
    ls.pose_is_ok_ = true;
}

// ===== kf_init: 50 IMU frames + gravity alignment =====

bool LaserMappingSuperLIO::kf_init() {
    for (auto& imu : measures_.imu_) {
        imu_init_count_++;
        BASIC::V3 gyr = imu->angular_velocity.cast<BASIC::scalar>();
        BASIC::V3 acc = imu->linear_acceleration.cast<BASIC::scalar>();
        mean_gyro_ += (gyr - mean_gyro_) / imu_init_count_;
        mean_acce_ += (acc - mean_acce_) / imu_init_count_;
    }
    if (imu_init_count_ < 50) return false;

    BASIC::V3 gravity = -mean_acce_ * gravity_norm_ / mean_acce_.norm();
    BASIC::V3 ref_gravity(0, 0, -gravity_norm_);
    BASIC::M3 init_rot = BASIC::Quat::FromTwoVectors(gravity, ref_gravity).toRotationMatrix();

    BASIC::V3 n = init_rot.col(0);
    double yaw = atan2(n(1), n(0));
    BASIC::M3 R_yaw_inv = Eigen::AngleAxis<BASIC::scalar>(-yaw, BASIC::V3::UnitZ()).toRotationMatrix();
    lidar_robo_yaw_ =
      Eigen::AngleAxisd(0.0 * M_PI / 180.0,
                        Eigen::Vector3d::UnitZ()).toRotationMatrix().cast<BASIC::scalar>();
    BASIC::M3 rot = lidar_robo_yaw_ * R_yaw_inv * init_rot;

    LI2Sup::ESKF::Options ekf_opt;
    ekf_opt.gyro_var_ = imu_ng_;
    ekf_opt.acce_var_ = imu_na_;
    ekf_opt.bias_gyro_var_ = imu_nbg_;
    ekf_opt.bias_acce_var_ = imu_nba_;
    ekf_opt.num_iterations_ = kf_max_iterations_;
    ekf_opt.quit_eps_ = kf_quit_eps_;

    float imu_scale = gravity_norm_ / mean_acce_.norm();
    kf_sl_->SetInitialConditions(ekf_opt, mean_gyro_, BASIC::V3::Zero(), imu_scale, ref_gravity);
    kf_sl_imu_->SetInitialConditions(ekf_opt, mean_gyro_, BASIC::V3::Zero(), imu_scale, ref_gravity);

    auto state = kf_sl_->GetSysState();
    state.R = BASIC::SO3(rot);
    state.p = BASIC::V3::Zero();
    state.timestamp = measures_.imu_.back()->timestamp;
    kf_sl_->SetX(state);
    kf_sl_imu_->SetX(state);

    sys_init_pose_ = kf_sl_->GetSE3();

    imu_initialized_ = true;

    LOG(INFO) << "SuperLIO KF init done, imu_count: " << imu_init_count_
              << ", imu_scale: " << imu_scale;
    return true;
}

// ===== map_init: first 3 lidar frames =====

bool LaserMappingSuperLIO::map_init() {
    frame_num_++;

    size_t pt_size = measures_.scan_->size();
    V3AlignedVec points_world;
    points_world.resize(pt_size);

    BASIC::SE3 transform = sys_init_pose_ * lidar_to_imu_;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, pt_size),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pt = measures_.scan_->points[i];
                BASIC::V3 pt_body(pt.x, pt.y, pt.z);
                points_world[i] = transform * pt_body;
            }
        });

    ivox_sl_->insert(points_world);
    kf_sl_->SetLastObsTime(measures_.lidar_end_time_);

    if (frame_num_ > 3) {
        kf_sl_->init_ = true;
        LOG(INFO) << "SuperLIO map init done, frames: " << frame_num_;
        return true;
    }
    return false;
}

// ===== Propagation_Undistort: ESKF predict + point cloud distortion =====

void LaserMappingSuperLIO::Propagation_Undistort() {
    propagate_states_.clear();

    {
        auto dynamic = kf_sl_->GetDynamicState();
        PropagateState ps;
        ps.time = dynamic.time;
        ps.R = dynamic.R;
        ps.p = dynamic.p;
        ps.v = dynamic.v;
        ps.w = dynamic.w;
        ps.a = dynamic.a;
        propagate_states_.push_back(ps);
    }

    kf_sl_->SetObsTime(measures_.lidar_end_time_);

    for (auto& imu : measures_.imu_) {
        LI2Sup::IMUData imu_sl;
        imu_sl.secs = imu->timestamp;
        imu_sl.gyr = imu->angular_velocity.cast<BASIC::scalar>();
        imu_sl.acc = imu->linear_acceleration.cast<BASIC::scalar>();
        kf_sl_->Predict(imu_sl);
        auto dynamic = kf_sl_->GetDynamicState();

        PropagateState ps;
        ps.time = dynamic.time;
        ps.R = dynamic.R;
        ps.p = dynamic.p;
        ps.v = dynamic.v;
        ps.w = dynamic.w;
        ps.a = dynamic.a;
        propagate_states_.push_back(ps);
    }

    static const BASIC::M3 TLI_R = lidar_to_imu_.R_;
    static const BASIC::V3 TLI_t = lidar_to_imu_.t_;
    const BASIC::SE3 T_end = kf_sl_->GetSE3();
    const BASIC::M3 R_inv = T_end.R_.transpose();
    const BASIC::V3 T_end_t = T_end.t_;
    const double start_time = measures_.lidar_begin_time_;
    auto& scan = measures_.scan_;
    const std::size_t pt_size = scan->size();
    scan_undistort_->resize(pt_size);
    scan_undistort_->header.stamp = measures_.lidar_end_time_*1e9;
    tbb::parallel_for(
    tbb::blocked_range<size_t>(0, pt_size),
    [&](const tbb::blocked_range<size_t>& r) {
        BASIC::M3 R_h, R_t; BASIC::V3 p_h, v_h, acc_t, w_t;
        for (size_t idx = r.begin(); idx < r.end(); ++idx) {
            auto& pt_out = scan_undistort_->points[idx];
            const auto& pt = scan->points[idx];
            pt_out.intensity = pt.intensity;
            double query_time = start_time + pt.time / 1000.0;

            if (query_time > propagate_states_.back().time) {
                BASIC::V3 raw(pt.x, pt.y, pt.z);
                BASIC::V3 eigen_point = TLI_R * raw + TLI_t;
                pt_out.x = eigen_point[0];
                pt_out.y = eigen_point[1];
                pt_out.z = eigen_point[2];
                continue;
            }

            size_t match_idx = 0;
            for (size_t j = 0; j < propagate_states_.size() - 1; ++j) {
                if (propagate_states_[j].time < query_time && propagate_states_[j + 1].time >= query_time) {
                    match_idx = j;
                    break;
                }
            }

            const auto& ms = propagate_states_[match_idx];
            const auto& ms_n = propagate_states_[match_idx + 1];
            double dt = ms_n.time - ms.time;
            double s = (query_time - ms.time) / dt;
            R_h = ms.R; R_t = ms_n.R;
            p_h = ms.p; v_h = ms.v;
            acc_t = ms_n.a; w_t = ms_n.w;

            BASIC::M3 R_i = BASIC::Quat(R_h).slerp(s, BASIC::Quat(R_t)).toRotationMatrix();
            BASIC::V3 t_ei(p_h + v_h * dt + 0.5 * acc_t * dt * dt - T_end_t);
            BASIC::V3 raw(pt.x, pt.y, pt.z);
            BASIC::V3 eigen_point = R_inv * (R_i * (TLI_R * raw + TLI_t) + t_ei);
            pt_out.x = eigen_point[0];
            pt_out.y = eigen_point[1];
            pt_out.z = eigen_point[2];
        }
    });
}

// ===== DownSample =====

void LaserMappingSuperLIO::DownSample() {
    voxel_scan_.setInputCloud(scan_undistort_);
    voxel_scan_.filter(*scan_down_body_);
}

// ===== Observe: OctVoxMap nearest search + plane fit + ESKF update =====

struct ThreadACC{
    BASIC::M6d HTVH = BASIC::M6d::Zero();
    BASIC::V6d HTVr = BASIC::V6d::Zero();
};

void LaserMappingSuperLIO::Observe() {
    size_t pt_size = scan_down_body_->size();
    if (pt_size == 0) return;

    points_body_v3_.resize(pt_size);
    points_world_v3_.resize(pt_size);
    abcd_vec_.resize(pt_size);

    static std::vector<float> lengths;
    lengths.resize(pt_size);
    for (size_t i = 0; i < pt_size; ++i) {
        const auto& pt = scan_down_body_->points[i];
        points_body_v3_[i] = BASIC::V3(pt.x, pt.y, pt.z);
        lengths[i] = points_body_v3_[i].norm();
    }

    std::vector<int> effect_idxs(pt_size);
    std::iota(effect_idxs.begin(), effect_idxs.end(), 0);
    size_t effect_num = pt_size;

    std::vector<bool> effect_mask(pt_size, false);

    ivox_sl_->reset_max_group();

    kf_sl_->UpdateObserve([&](const LI2Sup::ESKF::KFState& kf_state,
                              BASIC::M6& HTVH, BASIC::V6& HTVr) {
        BASIC::SE3 pose = kf_state.pose;
        BASIC::M3d R_transpose = (pose.R_.transpose()).cast<double>();
        bool need_converge = kf_state.need_converge;

        tbb::enumerable_thread_specific<ThreadACC> tls_acc;

        tbb::parallel_for(
        tbb::blocked_range<size_t>(0, effect_num),
        [&](const tbb::blocked_range<size_t>& r) {
            KNNHeapType top_K;
            auto& local_acc = tls_acc.local();
            for (size_t r_idx = r.begin(); r_idx < r.end(); ++r_idx) {
                int idx = effect_idxs[r_idx];
                BASIC::V3 point_world = pose * points_body_v3_[idx];
                points_world_v3_[idx] = point_world;

                if (!need_converge) {
                    top_K.reset();
                    ivox_sl_->getTopK(point_world, top_K);

                    if (top_K.count < 4) {
                        effect_mask[idx] = false;
                        continue;
                    }

                    std::array<double, 4> abcd;
                    Eigen::Matrix<double, 5, 3> A;
                    Eigen::Matrix<double, 5, 1> b = Eigen::Matrix<double, 5, 1>::Constant(-1.0);
                    int N = std::min<int>(top_K.count, 5);
                    for (int j = 0; j < N; ++j) {
                        A.row(j) = top_K.points_[j].template cast<double>();
                    }
                    Eigen::Vector3d nv = A.topRows(N).colPivHouseholderQr().solve(b.topRows(N));
                    double nn = nv.norm();
                    if (nn < 1e-6) { effect_mask[idx] = false; continue; }
                    nv /= nn;
                    abcd[0] = nv[0]; abcd[1] = nv[1]; abcd[2] = nv[2];
                    abcd[3] = 1.0 / nn;

                    bool plane_ok = true;
                    for (int j = 0; j < N; ++j) {
                        auto& p = top_K.points_[j];
                        double dist = abcd[0]*p[0] + abcd[1]*p[1] + abcd[2]*p[2] + abcd[3];
                        if (std::abs(dist) > 0.1) { plane_ok = false; break; }
                    }
                    if (!plane_ok) { effect_mask[idx] = false; continue; }

                    double error = abcd[0]*point_world[0] + abcd[1]*point_world[1] + abcd[2]*point_world[2] + abcd[3];
                    if (lengths[idx] <= 81 * error * error) { effect_mask[idx] = false; continue; }

                    effect_mask[idx] = true;
                    abcd_vec_[idx] = abcd;

                    BASIC::V3d normvec(abcd[0], abcd[1], abcd[2]);
                    BASIC::V3d nb = R_transpose * normvec;
                    BASIC::V3d point_body_d = points_body_v3_[idx].cast<double>();
                    BASIC::V6d J;
                    J.head<3>() = point_body_d.cross(nb);
                    J.tail<3>() = normvec;

                    local_acc.HTVH += J * 1000 * J.transpose();
                    local_acc.HTVr -= J * 1000 * error;
                } else {
                    if (!effect_mask[idx]) continue;

                    auto& abcd = abcd_vec_[idx];
                    double error = abcd[0]*point_world[0] + abcd[1]*point_world[1] + abcd[2]*point_world[2] + abcd[3];
                    if (lengths[idx] <= 81 * error * error) { effect_mask[idx] = false; continue; }

                    BASIC::V3d normvec(abcd[0], abcd[1], abcd[2]);
                    BASIC::V3d nb = R_transpose * normvec;
                    BASIC::V3d point_body_d = points_body_v3_[idx].cast<double>();
                    BASIC::V6d J;
                    J.head<3>() = point_body_d.cross(nb);
                    J.tail<3>() = normvec;

                    local_acc.HTVH += J * 1000 * J.transpose();
                    local_acc.HTVr -= J * 1000 * error;
                }
            }
        });

        BASIC::M6d sum_HTVH = BASIC::M6d::Zero();
        BASIC::V6d sum_HTVr = BASIC::V6d::Zero();
        for (const auto& local_acc : tls_acc) {
            sum_HTVH += local_acc.HTVH;
            sum_HTVr += local_acc.HTVr;
        }
        HTVH = sum_HTVH.cast<BASIC::scalar>();
        HTVr = sum_HTVr.cast<BASIC::scalar>();

        if (need_converge) return;

        effect_num = 0;
        for (size_t i = 0; i < pt_size; ++i) {
            if (effect_mask[i]) {
                effect_idxs[effect_num++] = i;
            }
        }
    });

    effect_feat_num_ = static_cast<int>(effect_num);
}

// ===== UpdateMap: insert into OctVoxMap =====

void LaserMappingSuperLIO::UpdateMap() {
    size_t pt_size = scan_down_body_->size();
    if (pt_size == 0) return;

    BASIC::SE3 last_pose = kf_sl_->GetSE3();
    points_world_v3_.resize(pt_size);

    for (size_t i = 0; i < pt_size; ++i) {
        points_world_v3_[i] = last_pose * points_body_v3_[i];
    }
    ivox_sl_->insert(points_world_v3_);
}

// ===== DetectZUPT: zero velocity update =====

void LaserMappingSuperLIO::DetectZUPT() {
    if (measures_.imu_.empty()) return;

    BASIC::V3 gyro_sum = BASIC::V3::Zero();
    BASIC::V3 gyro_sq_sum = BASIC::V3::Zero();

    for (const auto& imu : measures_.imu_) {
        BASIC::V3 gyr = imu->angular_velocity.cast<BASIC::scalar>();
        gyro_sum += gyr;
        gyro_sq_sum += gyr.array().square().matrix();
    }

    size_t imu_count = measures_.imu_.size();
    BASIC::V3 gyro_mean = gyro_sum / imu_count;
    double gyro_mean_z = gyro_mean.z();

    BASIC::V3 gyro_var = (gyro_sq_sum / imu_count) - (gyro_mean.array().square().matrix());
    double gyro_std_z = std::sqrt(std::max(0.0, (double)gyro_var.z()));

    const double GYRO_MEAN_THRESHOLD = 0.01;
    const double GYRO_STD_THRESHOLD = 0.01;
    bool is_stationary = (gyro_mean_z < GYRO_MEAN_THRESHOLD && gyro_std_z < GYRO_STD_THRESHOLD);

    if (is_stationary) {
        zupt_stationary_frames_++;
    } else {
        zupt_stationary_frames_ = 0;
        if (is_zupt_active_) {
            is_zupt_active_ = false;
            LOG(INFO) << "Exiting ZUPT mode";
        }
        return;
    }

    if (zupt_stationary_frames_ >= 1 && !is_zupt_active_) {
        is_zupt_active_ = true;
        LOG(INFO) << "Entering ZUPT mode";
    }

    if (is_zupt_active_) {
        auto sys_state = kf_sl_->GetSysState();
        sys_state.v.setZero();
        kf_sl_->SetX(sys_state);

        LI2Sup::ESKF::ObsFunc zupt_obs = [this](const LI2Sup::ESKF::KFState& kf_state,
                                                BASIC::M6& HT_Vinv_H, BASIC::V6& HT_Vinv_r) {
            HT_Vinv_H.setZero();
            HT_Vinv_r.setZero();

            const BASIC::SE3& pose = kf_state.pose;
            const BASIC::SO3& current_R = pose.so3();
            BASIC::V3 gravity = kf_sl_->GetGravity();
            BASIC::V3 expected_z_axis = -gravity.normalized();
            BASIC::V3 current_z_axis = current_R.matrix().col(2);
            BASIC::V3 att_residual = current_z_axis - expected_z_axis;

            BASIC::M3 W_att = BASIC::M3::Identity() * 10000;
            Eigen::Matrix3f H_att = -BASIC::SO3::hat(current_z_axis);

            HT_Vinv_H.block<3, 3>(0, 0) += H_att.transpose() * W_att * H_att;
            HT_Vinv_r.head<3>() += H_att.transpose() * W_att * att_residual;
        };
        kf_sl_->UpdateObserve(zupt_obs);
    }
}

// ===== Run =====

bool LaserMappingSuperLIO::Run() {
    if (!SyncPackages()) {
        return false;
    }

    if (flg_first_scan_) {
        if (!kf_init()) return false;
        if (!map_init()) return false;

        flg_first_scan_ = false;
        first_lidar_time_ = measures_.lidar_end_time_;
        auto sys_state = kf_sl_->GetSysState();
        SuperLIOStateToLightning(sys_state, state_point_);
        state_point_.timestamp_ = lidar_end_time_;

        LOG(INFO) << "SuperLIO frontend init done";
        return true;
    }

    frame_num_++;

    {
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

    Propagation_Undistort();

    if (scan_undistort_->size() < 5) {
        LOG(WARNING) << "Too few points after undistort";
        return false;
    }

    DownSample();
    size_t cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points after downsample";
        return false;
    }

    Observe();

    if (effect_feat_num_ < 5) {
        LOG(WARNING) << "[ SuperLIO ]: in: " << scan_undistort_->size()<<", Too few effective points: " << effect_feat_num_ << ", skip update";
        return false;
    }

    auto sys_state = kf_sl_->GetSysState();
    SuperLIOStateToLightning(sys_state, state_point_);
    state_point_.timestamp_ = measures_.lidar_end_time_;
    euler_cur_ = state_point_.rot_;
    pos_lidar_ = state_point_.pos_ + state_point_.rot_ * state_point_.offset_t_lidar_;

    UpdateMap();

    LOG(INFO) << "[ SuperLIO ]: in: " << scan_undistort_->size()
              << " down: " << cur_pts
              << " effect: " << effect_feat_num_
              << " ivox_grids: " << ivox_sl_->size();

    if (last_kf_ == nullptr) {
        MakeKF();
    } else {
        SE3 last_pose = last_kf_->GetLIOPose();
        SE3 cur_pose = state_point_.GetPose();
        bool dis_exceed = (last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_;
        bool angle_exceed = (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_;
        // 非建图模式下, 距上次关键帧超过 3 秒, 无论是否移动超过阈值都强制生成
        bool time_exceed = !options_.is_in_slam_mode_ &&
                           (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 3.0;
        if (dis_exceed || angle_exceed || time_exceed) {
            MakeKF();
        }
    }

    if (ui_) {
        ui_->UpdateNavState(state_point_);
        if (frame_num_ % pub_step_ == 0) {
            SE3 pose_imu = state_point_.GetPose();
            ui_->UpdateScan(scan_undistort_, pose_imu);
        }
    }
    kf_sl_imu_->CopyStateFrom(*kf_sl_);


    return true;
}

void LaserMappingSuperLIO::MakeKF() {
    // CloudPtr scan_cloud_imu(new PointCloudType());
    // for (int i = 0; i < scan_undistort_->size(); i++) {
    //     Vec3d p_global(scan_undistort_->points[i].getVector3fMap().cast<double>());
    //     PointType po;
    //     po.x = p_global(0); po.y = p_global(1); po.z = p_global(2);
    //     po.intensity = scan_undistort_->points[i].intensity;
    //     scan_cloud_imu->push_back(po);
    // }
    auto scan_cloud_kf = std::make_shared<PointCloudType>(*scan_undistort_);
    Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_cloud_kf, state_point_);
    //Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_undistort_, state_point_);
    if (last_kf_) {
        SE3 delta = last_kf_->GetLIOPose().inverse() * kf->GetLIOPose();
        kf->SetOptPose(last_kf_->GetOptPose() * delta);
    } else {
        kf->SetOptPose(kf->GetLIOPose());
    }
    kf->SetState(state_point_);

    if (options_.is_in_slam_mode_) {
        all_keyframes_.emplace_back(kf);
    }
    last_kf_ = kf;
}

NavState LaserMappingSuperLIO::GetState() const {
    return state_point_;
}

NavState LaserMappingSuperLIO::GetIMUState() const {
    if (imu_initialized_ && kf_sl_imu_) {
        auto ss = kf_sl_imu_->GetSysState();

        NavState ns;
        SuperLIOStateToLightning(ss, ns);
        return ns;
    }
    NavState s;
    s.pose_is_ok_ = false;
    return s;
}

CloudPtr LaserMappingSuperLIO::GetRecentCloud() {
    if (lidar_buffer_.empty()) return nullptr;
    return lidar_buffer_.front();
}

CloudPtr LaserMappingSuperLIO::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto& kf : all_keyframes_) {
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
    }

    CloudPtr filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*filtered);
    } else {
        filtered = global_map;
    }
    filtered->is_dense = false;
    filtered->height = 1;
    filtered->width = filtered->size();

    LOG(INFO) << "SuperLIO global map: " << filtered->size();
    return filtered;
}

void LaserMappingSuperLIO::SaveMap() {
    auto gm = GetGlobalMap(true);
    pcl::io::savePCDFileBinaryCompressed("./data/superlio.pcd", *gm);
    LOG(INFO) << "SuperLIO map saved to ./data/superlio.pcd";
}

}  // namespace lightning
