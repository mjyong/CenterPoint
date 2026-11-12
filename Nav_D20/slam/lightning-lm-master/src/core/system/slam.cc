//
// Created by xiang on 25-5-6.
//

#include "core/system/slam.h"
#include "core/g2p5/g2p5.h"
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

#ifndef ROOT
#define ROOT "./"
#endif

#include "core/lio/laser_mapping.h"
#include "core/loop_closing/loop_closing.h"
#include "core/maps/tiled_map.h"
#include "core/super_lio_frontend/laser_mapping_superlio.h"
#include "ui/rviz_visualizer.h"
#include "wrapper/ros_utils.h"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>

namespace lightning {
static Eigen::Vector3d lla_to_enu(const Eigen::Vector3d& lla, const GeographicLib::LocalCartesian& geo_converter_) {
    Eigen::Vector3d enu;
    double enu_x, enu_y, enu_z;
    // lat、lon、alt为纬度、经度、高度
    geo_converter_.Forward(lla.x(), lla.y(), lla.z(), enu_x, enu_y, enu_z);

    enu.x() = enu_x;
    enu.y() = enu_y;
    enu.z() = enu_z;
    return enu;
}

// 计算从 poses_a 到 poses_b 的 SE3 变换，使用 SVD 方法（Umeyama 算法）
static SE3 ComputeTransformSVD(const std::vector<SE3>& poses_a, const std::vector<SE3>& poses_b) {
    assert(poses_a.size() == poses_b.size() && poses_a.size() > 0);
    const size_t N = poses_a.size();
    // 提取平移部分
    std::vector<Vec3d> points_a, points_b;
    points_a.reserve(N);
    points_b.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        points_a.push_back(poses_a[i].translation());
        points_b.push_back(poses_b[i].translation());
    }
    // 计算质心
    Vec3d centroid_a = Vec3d::Zero();
    Vec3d centroid_b = Vec3d::Zero();
    for (size_t i = 0; i < N; ++i) {
        centroid_a += points_a[i];
        centroid_b += points_b[i];
    }
    centroid_a /= N;
    centroid_b /= N;
    // 去中心化点
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < N; ++i) {
        Vec3d a = points_a[i] - centroid_a;
        Vec3d b = points_b[i] - centroid_b;
        H += a * b.transpose();
    }
    // SVD 分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    // 计算旋转矩阵 R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();
    // 确保行列式为 +1（防止反射）
    if (R.determinant() < 0) {
        // 如果行列式为负，将 V 的最后一列取反
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    // 计算平移 t = centroid_b - R * centroid_a
    Vec3d t = centroid_b - R * centroid_a;
    // 构造 SE3
    SO3 rot(R);
    return SE3(rot, t);
}
SlamSystem::SlamSystem(lightning::SlamSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

bool SlamSystem::Init(const std::string& yaml_path) {
    // lio_ = std::make_shared<LaserMapping>();
    // if (!lio_->Init(yaml_path)) {
    //     LOG(ERROR) << "failed to init lio module";
    //     return false;
    // }

    auto yaml = YAML::LoadFile(yaml_path);
    options_.use_super_lio_ = yaml["system"]["use_super_lio"].as<bool>();
    if (options_.use_super_lio_) {
        LOG(INFO) << "Using SuperLIO frontend";
        LaserMappingSuperLIO::Options super_lio_opt;
        super_lio_opt.is_in_slam_mode_ = true;
        super_lio_ = std::make_shared<LaserMappingSuperLIO>(super_lio_opt);
        if (!super_lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init super_lio module";
            return false;
        }
    } else {
        lio_ = std::make_shared<LaserMapping>();
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio module";
            return false;
        }
    }

    options_.with_loop_closing_ = yaml["system"]["with_loop_closing"].as<bool>();
    options_.with_visualization_ = yaml["system"]["with_ui"].as<bool>();
    options_.with_2dvisualization_ = yaml["system"]["with_2dui"].as<bool>();
    options_.with_gridmap_ = yaml["system"]["with_g2p5"].as<bool>();
    options_.step_on_kf_ = yaml["system"]["step_on_kf"].as<bool>();
    options_.with_ins_ = yaml["system"]["with_ins"].as<bool>();
    double ins_lat = yaml["system"]["ins_lat"].as<double>();
    double ins_lon = yaml["system"]["ins_lon"].as<double>();
    double ins_height = yaml["system"]["ins_height"].as<double>();
    ins_topic_ = yaml["common"]["ins_topic"].as<std::string>();
    // 读取组合导航类型 (novatel 或 kygi665)
    if (yaml["common"]["ins_type"]) {
        ins_type_ = yaml["common"]["ins_type"].as<std::string>();
    }
    // kygi665 模式下使用专属话题 (仅 INS 数据源)
    if (yaml["common"]["kygi665_ins_topic"]) {
        kygi665_ins_topic_ = yaml["common"]["kygi665_ins_topic"].as<std::string>();
    }
    geo_converter_.Reset(ins_lat, ins_lon, ins_height);
    time_offset = yaml["fasterlio"]["time_offset"].as<double>();

    if (options_.with_loop_closing_) {
        LOG(INFO) << "slam with loop closing";
        LoopClosing::Options options;
        options.online_mode_ = options_.online_mode_;
        // 在线建图时禁用INS先验约束，以降低优化时间
        options.with_ins_prior_ = false;
        lc_ = std::make_shared<LoopClosing>(options);
        lc_->Init(yaml_path);
    }

    // if (options_.with_visualization_) {
    //     LOG(INFO) << "slam with 3D UI (Rviz)";
    //     ui_ = std::make_shared<ui::RvizVisualizer>();
    //     // 初始化将在节点创建后执行（如果在线模式）
    //     lio_->SetUI(ui_);
    // }
    if (options_.with_visualization_) {
        LOG(INFO) << "slam with 3D UI (Rviz)";
        ui_ = std::make_shared<ui::RvizVisualizer>();
        if (options_.use_super_lio_) {
            super_lio_->SetUI(ui_);
        } else {
            lio_->SetUI(ui_);
        }
    }

    if (options_.with_gridmap_) {
        g2p5::G2P5::Options opt;
        opt.online_mode_ = options_.online_mode_;

        g2p5_ = std::make_shared<g2p5::G2P5>(opt);
        g2p5_->Init(yaml_path);

        if (options_.with_loop_closing_) {
            /// 当发生回环时，触发一次重绘
            lc_->SetLoopClosedCB([this]() { g2p5_->RedrawGlobalMap(); });
        }

        if (options_.with_2dvisualization_) {
            g2p5_->SetMapUpdateCallback([this](g2p5::G2P5MapPtr map) {
                cv::Mat image = map->ToCV();
                cv::imshow("map", image);

                if (options_.step_on_kf_) {
                    cv::waitKey(0);

                } else {
                    cv::waitKey(10);
                }
            });
        }
    }

    if (options_.online_mode_) {
        LOG(INFO) << "online mode, creating ros2 node ... ";

        /// subscribers
        node_ = std::make_shared<rclcpp::Node>("lightning_slam");

        imu_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        lidar_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        ins_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // 创建多线程执行器（3线程），使 IMU、LiDAR、INS 回调组可并行处理
        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(rclcpp::ExecutorOptions(), 3);
        executor_->add_node(node_);

        imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
        cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
        livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

        rclcpp::QoS qos(10);
        // qos.best_effort();

        rclcpp::SubscriptionOptions imu_sub_opts;
        imu_sub_opts.callback_group = imu_cb_group_;

        rclcpp::SubscriptionOptions lidar_sub_opts;
        lidar_sub_opts.callback_group = lidar_cb_group_;

        rclcpp::SubscriptionOptions ins_sub_opts;
        ins_sub_opts.callback_group = ins_cb_group_;

        // IMU 订阅: 统一使用 sensor_msgs/Imu
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos,
            [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                IMUPtr imu = std::make_shared<IMU>();
                imu->timestamp = ToSec(msg->header.stamp) + time_offset;
                imu->linear_acceleration =
                    Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                // 角速度单位为 deg/s, 转换为 rad/s
                imu->angular_velocity =
                    Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z) * M_PI / 180.0;
                double current_time = ToSec(node_->now());
                double imu_diff = current_time - imu->timestamp;
                if (imu_diff > 0.04) {
                    LOG(ERROR) << "IMU 传感器时间戳与系统时间差值过大: " << imu_diff << " 秒";
                }
                // MonitorIMUFrameRate(imu->timestamp, current_time);
                ProcessIMU(imu);
            },
            imu_sub_opts);
        // huace ins imu 数据
        // imu_sub_ = node_->create_subscription<msg_interfaces::msg::Hcrawimub>(
        // imu_topic_, qos, [this](msg_interfaces::msg::Hcrawimub::SharedPtr msg) {
        //     IMUPtr imu = std::make_shared<IMU>();
        //     imu->timestamp = ToSec(msg->header.stamp);

        //     imu -> linear_acceleration[0] = msg->angular_acceleration.x;
        //     imu -> linear_acceleration[1] = -msg->angular_acceleration.y;
        //     imu -> linear_acceleration[2] = msg->angular_acceleration.z;

        //     imu -> angular_velocity[0] = msg -> angular_velocity.x * M_PI / 180.0;
        //     imu -> angular_velocity[1] = -msg -> angular_velocity.y * M_PI / 180.0;
        //     imu -> angular_velocity[2] = msg -> angular_velocity.z * M_PI / 180.0;

        //     ProcessIMU(imu);
        // });
        if (options_.with_ins_) {
            if (ins_type_ == "kygi665") {
                // kygi665 INSPVAX 接口
                sub_kygi665_ins_ = node_->create_subscription<kygi665_msgs::msg::INSPVAX>(
                    kygi665_ins_topic_, qos,
                    [this](kygi665_msgs::msg::INSPVAX::SharedPtr msg) {
                        Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                        Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                        INSPtr ins = std::make_shared<Ins>();
                        ins->stamp = rclcpp::Time(msg->header.stamp).seconds();
                        ins->pose.block<3, 1>(0, 3) = enu;
                        double azimuth_rad = (180.0 - msg->azimuth) * M_PI / 180.0;
                        double roll = 0;
                        double pitch = 0;
                        ins->pose.block<3, 3>(0, 0) = (Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                                                       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                                       Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                                          .toRotationMatrix();
                        // kygi665 pos_type.data 与 novatel 常量值一致: INS_RTKFIXED=56, INS_RTKFLOAT=55
                        // 要求 ins_status 为 INS_SOLUTION_GOOD(3) 才可用
                        bool ins_solution_good = (msg->ins_status.data == 3);
                        uint32_t pos_type = msg->pos_type.data;
                        if (ins_solution_good && (pos_type == 56 || pos_type == 55)) {
                            ins->state = INS_VALID;
                        } else {
                            ins->state = INS_INVALID;
                        }
                        ins->position_covariance =
                            Eigen::Vector3d(msg->latitude_stdev, msg->longitude_stdev, msg->height_stdev);
                        ins->orientation_covariance =
                            Eigen::Vector3d(msg->roll_stdev, msg->pitch_stdev, msg->azimuth_stdev);
                        {
                            std::lock_guard<std::mutex> lock(ins_buffer_mutex_);
                            ins_buffer_.push_back(ins);
                        }
                        // 发布 INS 位姿可视化
                        if (ins_pose_pub_ && ins->state == INS_VALID) {
                            geometry_msgs::msg::PoseStamped pose_msg;
                            pose_msg.header.stamp = msg->header.stamp;
                            pose_msg.header.frame_id = "map";
                            pose_msg.pose.position.x = enu.x();
                            pose_msg.pose.position.y = enu.y();
                            pose_msg.pose.position.z = enu.z();
                            Eigen::Matrix3d rot = ins->pose.block<3, 3>(0, 0);
                            Eigen::Quaterniond quat(rot);
                            pose_msg.pose.orientation.x = quat.x();
                            pose_msg.pose.orientation.y = quat.y();
                            pose_msg.pose.orientation.z = quat.z();
                            pose_msg.pose.orientation.w = quat.w();
                            ins_pose_pub_->publish(pose_msg);
                            ins_path_.poses.push_back(pose_msg);
                            ins_path_.header.stamp = pose_msg.header.stamp;
                            ins_path_pub_->publish(ins_path_);
                        }
                        double current_time = ToSec(node_->now());
                        double ins_diff = current_time - ins->stamp;
                        if (ins_diff > 0.1) {
                            LOG(ERROR) << "INS 传感器时间戳与系统时间差值过大: " << ins_diff << " 秒";
                        }
                    },
                    ins_sub_opts);
            } else {
                // novatel INSPVAX 接口
                sub_ins_ = node_->create_subscription<novatel_oem7_msgs::msg::INSPVAX>(
                    ins_topic_, qos,
                    [this](novatel_oem7_msgs::msg::INSPVAX::SharedPtr msg) {
                        Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                        Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                        INSPtr ins = std::make_shared<Ins>();
                        ins->stamp = rclcpp::Time(msg->header.stamp).seconds();
                        ins->pose.block<3, 1>(0, 3) = enu;
                        // INS azimuth 是与正北（Y轴）的夹角，顺时针为正
                        // 转换为 ENU 坐标系下与东（X轴）的夹角，逆时针为正
                        // 转换公式：angle_from_east = 90° - azimuth
                        double azimuth_rad = (90.0 - msg->azimuth) * M_PI / 180.0;
                        double roll = 0;   // msg->roll * M_PI / 180.0;
                        double pitch = 0;  // msg->pitch * M_PI / 180.0;
                        // 调试输出
                        static int debug_count = 0;
                        if (debug_count++ < 10) {
                            LOG(INFO) << "INS azimuth: " << msg->azimuth
                                      << "°, converted to: " << azimuth_rad * 180.0 / M_PI << "° from East";
                            LOG(INFO) << "INS position: " << enu.transpose();

                            // 计算欧拉角
                            Eigen::Matrix3d rot = (Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                                                   Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                                   Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                                      .toRotationMatrix();
                            Eigen::Vector3d euler = rot.eulerAngles(2, 1, 0);  // ZYX顺序
                            LOG(INFO) << "INS rotation (ZYX euler): " << euler.transpose() * 180.0 / M_PI << "°";
                        }

                        ins->pose.block<3, 3>(0, 0) = (Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                                                       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                                       Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                                          .toRotationMatrix();
                        if (msg->pos_type.type == 56) {
                            ins->state = INS_VALID;
                        } else {
                            ins->state = INS_INVALID;
                        }
                        ins->position_covariance =
                            Eigen::Vector3d(msg->latitude_stdev, msg->longitude_stdev, msg->height_stdev);
                        ins->orientation_covariance =
                            Eigen::Vector3d(msg->roll_stdev, msg->pitch_stdev, msg->azimuth_stdev);
                        {
                            std::lock_guard<std::mutex> lock(ins_buffer_mutex_);
                            double sys_now =
                                std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                            // LOG(INFO) << "[INS_RECV] stamp=" << std::fixed << std::setprecision(6) << ins->stamp
                            //           << " sys=" << sys_now << " buf_size=" << ins_buffer_.size()
                            //           << " state=" << (ins->state == INS_VALID ? "VALID" : "INVALID");
                            ins_buffer_.push_back(ins);
                        }
                        // 发布 INS 位姿可视化
                        if (ins_pose_pub_ && ins->state == INS_VALID) {
                            geometry_msgs::msg::PoseStamped pose_msg;
                            pose_msg.header.stamp = msg->header.stamp;
                            pose_msg.header.frame_id = "map";
                            pose_msg.pose.position.x = enu.x();
                            pose_msg.pose.position.y = enu.y();
                            pose_msg.pose.position.z = enu.z();
                            // 从旋转矩阵提取四元数
                            Eigen::Matrix3d rot = ins->pose.block<3, 3>(0, 0);
                            Eigen::Quaterniond quat(rot);
                            pose_msg.pose.orientation.x = quat.x();
                            pose_msg.pose.orientation.y = quat.y();
                            pose_msg.pose.orientation.z = quat.z();
                            pose_msg.pose.orientation.w = quat.w();
                            ins_pose_pub_->publish(pose_msg);

                            // 更新路径
                            ins_path_.poses.push_back(pose_msg);
                            ins_path_.header.stamp = pose_msg.header.stamp;
                            ins_path_pub_->publish(ins_path_);
                        }
                        double current_time = ToSec(node_->now());
                        double ins_diff = current_time - ins->stamp;
                        if (ins_diff > 0.1) {
                            LOG(ERROR) << "INS 传感器时间戳与系统时间差值过大: " << ins_diff << " 秒";
                        }
                        // MonitorINSFrameRate(ins->stamp, current_time);
                        //  ProcessIMU(imu);
                    },
                    ins_sub_opts);
            }
        }
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                double lidar_time = ToSec(cloud->header.stamp) + time_offset;
                double sys_time = ToSec(node_->now());
                double lidar_diff = sys_time - lidar_time;
                if (lidar_diff > 0.2) {
                    LOG(ERROR) << "LiDAR 传感器时间戳与系统时间差值过大: " << lidar_diff << " 秒";
                }
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            },
            lidar_sub_opts);

        // livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        //     livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
        //         Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        //     });

        // 初始化 INS 可视化发布器
        ins_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/ins_pose", 10);
        ins_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/ins_path", 10);
        ins_path_.header.frame_id = "map";

        savemap_service_ = node_->create_service<SaveMapService>(
            "lightning/save_map", [this](const SaveMapService::Request::SharedPtr& req,
                                         SaveMapService::Response::SharedPtr res) { SaveMap(req, res); });

        LOG(INFO) << "online slam node has been created.";
        // 初始化 Rviz 可视化器
        if (ui_) {
            ui_->Init(node_);
        }
    }

    return true;
}

SlamSystem::~SlamSystem() {
    if (ui_) {
        ui_->Quit();
    }
}

void SlamSystem::StartSLAM(std::string map_name) {
    map_name_ = map_name;
    running_ = true;
}
SE3 SlamSystem::GetLastKeyframePose() {
    if (cur_kf_) {
        return cur_kf_->GetOptPose();
    }
    return SE3();
}
void SlamSystem::SaveMap(const SaveMapService::Request::SharedPtr request,
                         SaveMapService::Response::SharedPtr response) {
    map_name_ = request->map_id;
    std::string save_path = "./data/" + map_name_ + "/";

    // SaveMap(save_path);
    SaveMapByTemplate(save_path);
    response->response = 0;
}

void SlamSystem::SaveMap(const std::string& path) {
    std::string save_path = path;
    if (save_path.empty()) {
        save_path = "./data/" + map_name_ + "/";
    }

    LOG(INFO) << "slam map saving to " << save_path;

    // 在保存地图前执行全局优化（包括INS约束）
    if (options_.with_loop_closing_ && options_.with_ins_ && lc_ != nullptr) {
        LOG(INFO) << "Performing global optimization with INS constraints before saving map";
        // 设置INS到LIO的外参变换
        if (T_ins_lio_computed_) {
            lc_->SetInsLioExtrinsic(T_ins_lio_);
        } else {
            LOG(WARNING) << "T_ins_lio not computed, skipping global optimization with INS";
        }
        lc_->GlobalOptimizationWithINS();

        // 优化后重新绘制栅格地图
        if (options_.with_gridmap_ && g2p5_ != nullptr) {
            g2p5_->RedrawGlobalMap();
            g2p5_->WaitForRedrawComplete();  // 等待重绘完成，确保栅格地图使用优化后的位姿
        }
    }

    // 检查是否有有效关键帧，避免空地图保存时访问空容器的 front() 导致 SIGSEGV
    bool has_keyframes = false;
    if (options_.use_super_lio_) {
        has_keyframes = !super_lio_->GetAllKeyframes().empty();
    } else {
        has_keyframes = !lio_->GetAllKeyframes().empty();
    }
    if (!has_keyframes) {
        LOG(WARNING) << "No keyframes in map, skipping save";
        return;
    }

    if (!std::filesystem::exists(save_path)) {
        std::filesystem::create_directories(save_path);
    } else {
        std::filesystem::remove_all(save_path);
        std::filesystem::create_directories(save_path);
    }

    CloudPtr global_map;
    SE3 start_pose;
    if (options_.use_super_lio_) {
        global_map = super_lio_->GetGlobalMap(false);
        start_pose = super_lio_->GetAllKeyframes().front()->GetOptPose();
    } else {
        global_map = lio_->GetGlobalMap(!options_.with_loop_closing_);
        start_pose = lio_->GetAllKeyframes().front()->GetOptPose();
    }

    // auto global_map_no_loop = lio_->GetGlobalMap(true);
    // auto global_map = lio_->GetGlobalMap(!options_.with_loop_closing_);
    // // auto global_map_raw = lio_->GetGlobalMap(!options_.with_loop_closing_, false, 0.1);
    // if(global_map->size()==0) {
    //     LOG(ERROR) << "global map is empty, skip saving";
    //     return;
    // }

    // TiledMap::Options tm_options;
    // tm_options.map_path_ = save_path;

    // TiledMap tm(tm_options);
    // SE3 start_pose = lio_->GetAllKeyframes().front()->GetOptPose();
    // tm.ConvertFromFullPCD(global_map, start_pose, save_path);

    TiledMap::Options tm_options;
    tm_options.map_path_ = save_path;

    TiledMap tm(tm_options);
    tm.ConvertFromFullPCD(global_map, start_pose, save_path);

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *global_map);
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_no_loop.pcd", *global_map_no_loop);
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_raw.pcd", *global_map_raw);

    if (options_.with_gridmap_) {
        /// 存为ROS兼容的模式
        auto map = g2p5_->GetNewestMap()->ToROS();
        const int width = map.info.width;
        const int height = map.info.height;

        cv::Mat nav_image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            const int rowStartIndex = y * width;
            for (int x = 0; x < width; ++x) {
                const int index = rowStartIndex + x;
                int8_t data = map.data[index];
                if (data == 0) {                                   // Free
                    nav_image.at<uchar>(height - 1 - y, x) = 255;  // White
                } else if (data == 100) {                          // Occupied
                    nav_image.at<uchar>(height - 1 - y, x) = 0;    // Black
                } else {                                           // Unknown
                    nav_image.at<uchar>(height - 1 - y, x) = 128;  // Gray
                }
            }
        }

        cv::imwrite(save_path + "/map.png", nav_image);

        /// yaml
        std::ofstream yamlFile(save_path + "/map.yaml");
        if (!yamlFile.is_open()) {
            LOG(ERROR) << "failed to write map.yaml";
            return;  // 文件打开失败
        }

        try {
            YAML::Emitter emitter;
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "image" << YAML::Value << "map.png";
            emitter << YAML::Key << "mode" << YAML::Value << "trinary";
            emitter << YAML::Key << "width" << YAML::Value << map.info.width;
            emitter << YAML::Key << "height" << YAML::Value << map.info.height;
            emitter << YAML::Key << "resolution" << YAML::Value << map.info.resolution;
            std::vector<double> orig{map.info.origin.position.x, map.info.origin.position.y, 0};
            emitter << YAML::Key << "origin" << YAML::Value << orig;
            emitter << YAML::Key << "negate" << YAML::Value << 0;
            emitter << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
            emitter << YAML::Key << "free_thresh" << YAML::Value << 0.25;

            emitter << YAML::EndMap;

            yamlFile << emitter.c_str();
            yamlFile.close();
        } catch (...) {
            yamlFile.close();
            return;
        }
    }

    LOG(INFO) << "map saved";
}

void SlamSystem::ProcessIMU(const lightning::IMUPtr& imu) {
    // if (running_ == false) {
    //     return;
    // }
    // lio_->ProcessIMU(imu);
    if (running_ == false) {
        return;
    }
    if (options_.use_super_lio_) {
        super_lio_->ProcessIMU(imu);
    } else {
        lio_->ProcessIMU(imu);
    }
}

void SlamSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }
    // 获取激光雷达时间戳
    double lidar_time_f = ToSec(cloud->header.stamp);
    // 监测帧率与丢帧（丢帧检测使用传感器时间，帧率统计使用系统时间）
    double current_time = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    MonitorLidarFrameRate(lidar_time_f, current_time);
    // lio_->ProcessPointCloud2(cloud);
    // lio_->Run();

    // auto kf = lio_->GetKeyframe();

    if (options_.use_super_lio_) {
        super_lio_->ProcessPointCloud2(cloud);
        super_lio_->Run();
    } else {
        lio_->ProcessPointCloud2(cloud);
        lio_->Run();
    }

    Keyframe::Ptr kf;
    if (options_.use_super_lio_) {
        kf = super_lio_->GetKeyframe();
    } else {
        kf = lio_->GetKeyframe();
    }

    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }
    // 获取激光雷达时间戳
    double lidar_time = cur_kf_->GetState().timestamp_;
    // 找出最近的INS数据
    if (options_.with_ins_) {
        INSPtr nearest_ins = FindNearestINS(lidar_time);
        if (nearest_ins) {
            // LOG(INFO) << "Nearest INS found at stamp " << nearest_ins->stamp
            //         << ", state " << nearest_ins->state
            //         << ", time diff " << (nearest_ins->stamp - lidar_time);
            SE3 ins_pose(nearest_ins->pose.block<3, 3>(0, 0), nearest_ins->pose.block<3, 1>(0, 3));
            if (nearest_ins->state == INS_VALID) {
                cur_kf_->SetInsPose(ins_pose);
                // 设置 INS 协方差
                cur_kf_->SetInsCovariance(nearest_ins->position_covariance, nearest_ins->orientation_covariance);
                // 如果尚未计算 INS 到 LIO 的变换，使用第一帧有效INS计算外参
                if (!T_ins_lio_computed_) {
                    SE3 lio_pose = cur_kf_->GetLIOPose();
                    T_ins_lio_ = ins_pose * lio_pose.inverse();
                    T_ins_lio_computed_ = true;
                    LOG(INFO) << "Computed T_ins_lio using first valid INS: " << T_ins_lio_.matrix();
                }
            } else {
                LOG(INFO) << "INS invalid, using LIO pose as opt pose.";
            }
        } else {
            LOG(WARNING) << "No INS data found in buffer for lidar time " << std::fixed << std::setprecision(15)
                         << lidar_time;
        }
        // 清理旧数据
        RemoveOldINS(lidar_time);
    }
    if (options_.with_loop_closing_) {
        lc_->AddKF(cur_kf_);
    }
    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    // 检查运动条件并触发地图回调
    if (monitor_enabled_ && map_callback_) {
        SE3 current_pose = cur_kf_->GetOptPose();
        CheckMotionAndTrigger(current_pose);
    }

    // if (ui_) {
    //     ui_->UpdateKF(cur_kf_);
    // }
}

void SlamSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }
    // 获取激光雷达时间戳
    double lidar_time_f = ToSec(cloud->header.stamp);
    // 监测帧率与丢帧（丢帧检测使用传感器时间，帧率统计使用系统时间）
    double current_time = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // MonitorLidarFrameRate(lidar_time_f, current_time);
    //  lio_->ProcessPointCloud2(cloud);
    //  lio_->Run();

    // auto kf = lio_->GetKeyframe();

    if (options_.use_super_lio_) {
        super_lio_->ProcessPointCloud2(cloud);
        super_lio_->Run();
    } else {
        lio_->ProcessPointCloud2(cloud);
        lio_->Run();
    }

    Keyframe::Ptr kf;
    if (options_.use_super_lio_) {
        kf = super_lio_->GetKeyframe();
    } else {
        kf = lio_->GetKeyframe();
    }

    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }
    // 获取激光雷达时间戳
    double lidar_time = cur_kf_->GetState().timestamp_;
    // 找出最近的INS数据
    if (options_.with_ins_) {
        INSPtr nearest_ins = FindNearestINS(lidar_time);
        if (nearest_ins) {
            // LOG(INFO) << "Nearest INS found at stamp " << nearest_ins->stamp
            //         << ", state " << nearest_ins->state
            //         << ", time diff " << (nearest_ins->stamp - lidar_time);
            SE3 ins_pose(nearest_ins->pose.block<3, 3>(0, 0), nearest_ins->pose.block<3, 1>(0, 3));
            if (nearest_ins->state == INS_VALID) {
                cur_kf_->SetInsPose(ins_pose);
                // 设置 INS 协方差
                cur_kf_->SetInsCovariance(nearest_ins->position_covariance, nearest_ins->orientation_covariance);

                // 如果尚未计算 INS 到 LIO 的变换，使用第一帧有效INS计算外参
                if (!T_ins_lio_computed_) {
                    SE3 lio_pose = cur_kf_->GetLIOPose();
                    // 使用第一帧有效INS计算外参：T_ins_lio = ins_pose * lio_pose.inverse()
                    T_ins_lio_ = ins_pose * lio_pose.inverse();
                    T_ins_lio_computed_ = true;
                    LOG(INFO) << "Computed T_ins_lio using first valid INS: " << T_ins_lio_.matrix();
                }
            } else {
                LOG(INFO) << "INS invalid, using LIO pose as opt pose.";
            }
        } else {
            LOG(WARNING) << "No INS data found in buffer for lidar time " << lidar_time;
        }

        // 清理旧数据
        RemoveOldINS(lidar_time);
    }
    if (options_.with_loop_closing_) {
        lc_->AddKF(cur_kf_);
    }

    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    // 检查运动条件并触发地图回调
    if (monitor_enabled_ && map_callback_) {
        SE3 current_pose = cur_kf_->GetOptPose();
        CheckMotionAndTrigger(current_pose);
    }

    // if (ui_) {
    //     ui_->UpdateKF(cur_kf_);
    // }
}
INSPtr SlamSystem::FindNearestINS(double stamp) {
    std::lock_guard<std::mutex> lock(ins_buffer_mutex_);
    if (ins_buffer_.empty()) {
        LOG(WARNING) << "[INS_FIND] lidar_time=" << std::fixed << std::setprecision(6) << stamp << " buffer EMPTY";
        return nullptr;
    }
    LOG(INFO) << "[INS_FIND] lidar_time=" << std::fixed << std::setprecision(6) << stamp
              << " buf_size=" << ins_buffer_.size() << " front=" << ins_buffer_.front()->stamp
              << " back=" << ins_buffer_.back()->stamp;
    auto it = std::lower_bound(ins_buffer_.begin(), ins_buffer_.end(), stamp,
                               [](const INSPtr& ins, double t) { return ins->stamp < t; });
    INSPtr nearest = nullptr;
    if (it == ins_buffer_.begin()) {
        nearest = *it;
    } else if (it == ins_buffer_.end()) {
        // 所有时间戳都小于 stamp，返回最后一个
        nearest = ins_buffer_.back();
    } else {
        // 比较 it 和 it-1 哪个更接近
        double diff1 = std::abs((*it)->stamp - stamp);
        double diff2 = std::abs((*(it - 1))->stamp - stamp);
        nearest = diff1 < diff2 ? *it : *(it - 1);
    }

    // 检查时间差是否在允许范围内
    const double MAX_TIME_DIFF = 0.1;  // 100ms
    if (nearest && std::abs(nearest->stamp - stamp) > MAX_TIME_DIFF) {
        LOG(WARNING) << "INS time diff too large: " << std::abs(nearest->stamp - stamp) << "s, ignoring this INS data.";
        return nullptr;
    }
    return nearest;
}

void SlamSystem::RemoveOldINS(double stamp) {
    std::lock_guard<std::mutex> lock(ins_buffer_mutex_);
    const double keep_window = 5;  // 保留最近0.5秒的数据
    double threshold = stamp - keep_window;
    while (!ins_buffer_.empty() && ins_buffer_.front()->stamp < threshold) {
        ins_buffer_.pop_front();
    }
}

void SlamSystem::MonitorLidarFrameRate(double sensor_time, double sys_time) {
    // 丢帧检测使用传感器时间
    if (first_lidar_time_ == 0.0) {
        first_lidar_time_ = sensor_time;
    }
    if (last_lidar_time_ > 0.0) {
        double interval = sensor_time - last_lidar_time_;
        if (interval > max_interval_) {
            max_interval_ = interval;
        }
        if (interval < min_interval_) {
            min_interval_ = interval;
        }
        // 检测丢帧：间隔超过预期间隔的1.5倍
        if (interval > expected_interval_ * 1.5) {
            int dropped = static_cast<int>(interval / expected_interval_) - 1;
            if (dropped > 0) {
                dropped_frames_ += dropped;
                LOG(WARNING) << "Lidar frame drop detected: interval " << interval << " s, expected "
                             << expected_interval_ << " s, estimated " << dropped << " frames dropped.";
            }
        }
        // 检测零间隔（卡帧）
        if (interval < 1e-6) {
            LOG(WARNING) << "Lidar frame stuck: zero interval (timestamp unchanged).";
        }
        // 检测负间隔（时间戳倒退）
        if (interval < 0.0) {
            LOG(WARNING) << "Lidar frame timestamp reversed: interval " << interval << " s.";
        }
        // 检测粘包：间隔小于预期间隔的0.5倍（异常快速）
        if (interval < expected_interval_ * 0.5) {
            LOG(WARNING) << "Lidar frame sticking detected: interval " << interval << " s, expected "
                         << expected_interval_ << " s, possible packet sticking.";
        }
    }
    last_lidar_time_ = sensor_time;

    // 帧率统计使用系统时间
    if (first_lidar_sys_time_ == 0.0) {
        first_lidar_sys_time_ = sys_time;
    }
    last_lidar_sys_time_ = sys_time;
    frame_count_++;

    // 每100帧计算并输出一次帧率统计（基于系统时间）
    if (frame_count_ % 100 == 0) {
        double total_sys_time = last_lidar_sys_time_ - first_lidar_sys_time_;
        if (total_sys_time > 0.0) {
            frame_rate_ = frame_count_ / total_sys_time;
            LOG(INFO) << "Lidar frame rate (sys): " << frame_rate_ << " Hz, frames: " << frame_count_
                      << ", dropped: " << dropped_frames_ << ", max interval: " << max_interval_
                      << ", min interval: " << min_interval_;
        }
    }
}

void SlamSystem::MonitorIMUFrameRate(double sensor_time, double sys_time) {
    // 丢帧检测使用传感器时间
    if (first_imu_time_ == 0.0) {
        first_imu_time_ = sensor_time;
    }
    if (last_imu_time_ > 0.0) {
        double interval = sensor_time - last_imu_time_;
        if (interval > imu_max_interval_) {
            imu_max_interval_ = interval;
        }
        if (interval < imu_min_interval_) {
            imu_min_interval_ = interval;
        }
        // 检测丢帧：间隔超过预期间隔的1.5倍
        if (interval > imu_expected_interval_ * 1.5) {
            int dropped = static_cast<int>(interval / imu_expected_interval_) - 1;
            if (dropped > 0) {
                imu_dropped_ += dropped;
                LOG(WARNING) << "IMU frame drop detected: interval " << interval << " s, expected "
                             << imu_expected_interval_ << " s, estimated " << dropped << " frames dropped.";
            }
        }
        // 检测零间隔（卡帧）
        if (interval < 1e-6) {
            LOG(WARNING) << "IMU frame stuck: zero interval (timestamp unchanged).";
        }
        // 检测负间隔（时间戳倒退）
        if (interval < 0.0) {
            LOG(WARNING) << "IMU frame timestamp reversed: interval " << interval << " s.";
        }
        // 检测粘包：间隔小于预期间隔的0.5倍（异常快速）
        if (interval < imu_expected_interval_ * 0.5) {
            LOG(WARNING) << "IMU frame sticking detected: interval " << interval << " s, expected "
                         << imu_expected_interval_ << " s, possible packet sticking.";
        }
    }
    last_imu_time_ = sensor_time;

    // 帧率统计使用系统时间
    if (first_imu_sys_time_ == 0.0) {
        first_imu_sys_time_ = sys_time;
    }
    last_imu_sys_time_ = sys_time;
    imu_count_++;

    // 每1000帧计算并输出一次帧率统计（基于系统时间）
    if (imu_count_ % 1000 == 0) {
        double total_sys_time = last_imu_sys_time_ - first_imu_sys_time_;
        if (total_sys_time > 0.0) {
            imu_rate_ = imu_count_ / total_sys_time;
            LOG(INFO) << "IMU frame rate (sys): " << imu_rate_ << " Hz, frames: " << imu_count_
                      << ", dropped: " << imu_dropped_ << ", max interval: " << imu_max_interval_
                      << ", min interval: " << imu_min_interval_;
        }
    }
}

void SlamSystem::MonitorINSFrameRate(double sensor_time, double sys_time) {
    // 丢帧检测使用传感器时间
    if (first_ins_time_ == 0.0) {
        first_ins_time_ = sensor_time;
    }
    if (last_ins_time_ > 0.0) {
        double interval = sensor_time - last_ins_time_;
        if (interval > ins_max_interval_) {
            ins_max_interval_ = interval;
        }
        if (interval < ins_min_interval_) {
            ins_min_interval_ = interval;
        }
        // 检测丢帧：间隔超过预期间隔的1.5倍
        if (interval > ins_expected_interval_ * 1.5) {
            int dropped = static_cast<int>(interval / ins_expected_interval_) - 1;
            if (dropped > 0) {
                ins_dropped_ += dropped;
                LOG(WARNING) << "INS frame drop detected: interval " << interval << " s, expected "
                             << ins_expected_interval_ << " s, estimated " << dropped << " frames dropped.";
            }
        }
        // 检测零间隔（卡帧）
        if (interval < 1e-6) {
            LOG(WARNING) << "INS frame stuck: zero interval (timestamp unchanged).";
        }
        // 检测负间隔（时间戳倒退）
        if (interval < 0.0) {
            LOG(WARNING) << "INS frame timestamp reversed: interval " << interval << " s.";
        }
        // 检测粘包：间隔小于预期间隔的0.5倍（异常快速）
        if (interval < ins_expected_interval_ * 0.5) {
            LOG(WARNING) << "INS frame sticking detected: interval " << interval << " s, expected "
                         << ins_expected_interval_ << " s, possible packet sticking.";
        }
    }
    last_ins_time_ = sensor_time;

    // 帧率统计使用系统时间
    if (first_ins_sys_time_ == 0.0) {
        first_ins_sys_time_ = sys_time;
    }
    last_ins_sys_time_ = sys_time;
    ins_count_++;

    // 每500帧计算并输出一次帧率统计（基于系统时间）
    if (ins_count_ % 500 == 0) {
        double total_sys_time = last_ins_sys_time_ - first_ins_sys_time_;
        if (total_sys_time > 0.0) {
            ins_rate_ = ins_count_ / total_sys_time;
            LOG(INFO) << "INS frame rate (sys): " << ins_rate_ << " Hz, frames: " << ins_count_
                      << ", dropped: " << ins_dropped_ << ", max interval: " << ins_max_interval_
                      << ", min interval: " << ins_min_interval_;
        }
    }
}
void SlamSystem::Spin() {
    if (executor_ != nullptr) {
        executor_->spin();
    }
}

void SlamSystem::SpinSome() {
    if (executor_ != nullptr) {
        executor_->spin_some();
    }
}

void SlamSystem::CancelSpin() {
    if (executor_ != nullptr) {
        executor_->cancel();
    }
}

void SlamSystem::StartMapMonitor(float translation_threshold_m, float rotation_threshold_deg,
                                 MapDataCallback callback) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    monitor_enabled_ = true;
    trans_threshold_ = translation_threshold_m;
    rot_threshold_ = rotation_threshold_deg * M_PI / 180.0;  // 转换为弧度
    map_callback_ = callback;
    trigger_initialized_ = false;
    LOG(INFO) << "Map monitor started with translation threshold " << trans_threshold_ << " m, rotation threshold "
              << rot_threshold_ << " rad.";
}

void SlamSystem::StopMapMonitor() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    monitor_enabled_ = false;
    LOG(INFO) << "Map monitor stopped.";
}

void SlamSystem::CheckMotionAndTrigger(const SE3& current_pose) {
    MapDataCallback callback;
    bool enabled = false;
    float trans_thresh = 0.0;
    float rot_thresh = 0.0;
    {
        std::lock_guard<std::mutex> lock(monitor_mutex_);
        if (!monitor_enabled_ || !map_callback_) {
            return;
        }
        enabled = true;
        trans_thresh = trans_threshold_;
        rot_thresh = rot_threshold_;
        callback = map_callback_;
        if (!trigger_initialized_) {
            last_trigger_pose_ = current_pose;
            trigger_initialized_ = true;
            last_trigger_time_ = std::chrono::steady_clock::now();
            return;
        }
    }
    // 计算相对运动（不需要锁）
    SE3 delta = last_trigger_pose_.inverse() * current_pose;
    double translation = delta.translation().norm();
    double rotation = delta.so3().log().norm();  // 旋转向量模长（弧度）
    auto now = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(now - last_trigger_time_).count();
    if ((translation >= trans_thresh || rotation >= rot_thresh) && elapsed_sec >= 5.0) {
        // 触发回调
        if (options_.with_gridmap_ && g2p5_ != nullptr) {
            auto map = g2p5_->GetNewestMap();
            if (map) {
                // 更新触发位姿和时间（需要锁），避免重复触发
                {
                    std::lock_guard<std::mutex> lock(monitor_mutex_);
                    last_trigger_pose_ = current_pose;
                    last_trigger_time_ = now;
                }
                LOG(INFO) << "********Map monitor triggered: translation " << translation << " m, rotation " << rotation
                          << " rad.";
                // 在单独线程中执行耗时操作（地图转换和回调）
                std::thread([map, callback, current_pose, this]() {
                    auto data = ConvertMapToUint8Array(map);
                    int width = map->ToROS().info.width;
                    int height = map->ToROS().info.height;
                    float resolution = map->GetGridResolution();
                    float min_x, min_y, max_x, max_y;
                    callback(data, width, height, resolution, map->ToROS().info.origin.position.x,
                             map->ToROS().info.origin.position.y);
                }).detach();  // 分离线程，让它在后台运行
            }
        }
    }
}

std::vector<uint8_t> SlamSystem::ConvertMapToUint8Array(const g2p5::G2P5MapPtr& map) {
    auto ros_map = map->ToROS();
    const int width = ros_map.info.width;
    const int height = ros_map.info.height;
    std::vector<uint8_t> result(width * height);
    for (int i = 0; i < width * height; ++i) {
        int8_t val = ros_map.data[i];
        // 映射：-1 -> 128 (未知), 0 -> 255 (空闲), 100 -> 0 (占用)
        if (val == -1) {
            result[i] = 128;
        } else if (val == 0) {
            result[i] = 255;
        } else if (val == 100) {
            result[i] = 0;
        } else {
            // 其他值（理论上不会出现）按比例映射
            result[i] = static_cast<uint8_t>(std::max(0, std::min(255, static_cast<int>(val * 2.55))));
        }
    }
    return result;
}
/// SE3 转为 [x,y,z,qx,qy,qz,qw] 列表，用于 YAML 输出
static std::vector<double> SE3ToVec7(const SE3& pose) {
    const auto& t = pose.translation();
    const auto& q = pose.unit_quaternion();
    return {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
}
void SlamSystem::SaveMapByTemplate(const std::string& path) {
    std::string save_path = path;
    if (save_path.empty()) {
        save_path = "./data/" + map_name_ + "/";
    }
    // 确保路径以 '/' 结尾，避免拼接子目录时缺少分隔符
    if (!save_path.empty() && save_path.back() != '/') {
        save_path += '/';
    }
    LOG(INFO) << "saving map (template) to " << save_path;

    // 在保存地图前执行全局优化（包括INS约束）
    if (options_.with_loop_closing_ && options_.with_ins_ && lc_ != nullptr) {
        LOG(INFO) << "Performing global optimization with INS constraints before saving map (template)";
        // 设置INS到LIO的外参变换
        if (T_ins_lio_computed_) {
            lc_->SetInsLioExtrinsic(T_ins_lio_);
        } else {
            LOG(WARNING) << "T_ins_lio not computed, skipping global optimization with INS";
        }
        lc_->GlobalOptimizationWithINS();

        // 优化后重新绘制栅格地图
        if (options_.with_gridmap_ && g2p5_ != nullptr) {
            g2p5_->RedrawGlobalMap();
            g2p5_->WaitForRedrawComplete();  // 等待重绘完成，确保栅格地图使用优化后的位姿
        }
    }

    // 获取所有关键帧
    std::vector<Keyframe::Ptr> all_kfs;
    if (options_.use_super_lio_) {
        all_kfs = super_lio_->GetAllKeyframes();
    } else {
        all_kfs = lio_->GetAllKeyframes();
    }
    if (all_kfs.empty()) {
        LOG(ERROR) << "no keyframes, skip saving";
        return;
    }

    // 创建目录结构
    std::string time_str = std::to_string((long long)(all_kfs.front()->GetState().timestamp_));
    std::string dir_2d = save_path + "2d/" + time_str + "/";
    std::string dir_3d = save_path + "3d/";
    std::string dir_kf = save_path + "session/keyframes/";
    std::filesystem::create_directories(dir_2d);
    std::filesystem::create_directories(dir_3d);
    std::filesystem::create_directories(dir_kf);

    int last_id = all_kfs.back()->GetID();

    // === 1. 保存关键帧 ===
    for (auto& kf : all_kfs) {
        int id = kf->GetID();
        // 点云
        if (kf->GetCloud() && !kf->GetCloud()->empty()) {
            pcl::io::savePCDFileBinaryCompressed(dir_kf + std::to_string(id) + ".pcd", *kf->GetCloud());
        }
        // yaml
        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "global_id" << YAML::Value << id;
        emitter << YAML::Key << "session_id" << YAML::Value << 0;
        emitter << YAML::Key << "session_local_id" << YAML::Value << id;
        emitter << YAML::Key << "timestamp" << YAML::Value << kf->GetState().timestamp_;
        emitter << YAML::Key << "cloud_file" << YAML::Value << (std::to_string(id) + ".pcd");
        emitter << YAML::Key << "pose_lio" << YAML::Value << SE3ToVec7(kf->GetLIOPose());
        emitter << YAML::Key << "pose_opt" << YAML::Value << SE3ToVec7(kf->GetOptPose());
        emitter << YAML::Key << "pose_base" << YAML::Value << SE3ToVec7(kf->GetLIOPose());
        emitter << YAML::Key << "raw_floor_id" << YAML::Value << 1000;
        emitter << YAML::Key << "floor_id" << YAML::Value << 1;
        emitter << YAML::Key << "area_type" << YAML::Value << "floor";
        emitter << YAML::Key << "transition_id" << YAML::Value << -1;
        emitter << YAML::Key << "use_for_grid" << YAML::Value << true;
        emitter << YAML::Key << "is_session_anchor" << YAML::Value << (id == 0);
        emitter << YAML::Key << "anchor_pose" << YAML::Value << std::vector<double>{0, 0, 0, 0, 0, 0, 1};
        emitter << YAML::Key << "anchor_confidence" << YAML::Value << 0.95;
        emitter << YAML::EndMap;

        std::ofstream f(dir_kf + std::to_string(id) + ".yaml");
        f << emitter.c_str();
        f.close();
    }

    // === 2. 保存 3D 地图（全局 + 分块） ===
    CloudPtr global_map;
    SE3 start_pose = all_kfs.front()->GetOptPose();
    if (options_.use_super_lio_) {
        global_map = super_lio_->GetGlobalMap(false);
    } else {
        global_map = lio_->GetGlobalMap(!options_.with_loop_closing_);
    }
    if (global_map && !global_map->empty()) {
        // 全局点云
        pcl::io::savePCDFileBinaryCompressed(dir_3d + "global.pcd", *global_map);

        // 分块点云 + index.txt（由 TiledMap 自动生成）
        TiledMap::Options tm_options;
        tm_options.map_path_ = dir_3d;
        TiledMap tm(tm_options);
        tm.ConvertFromFullPCD(global_map, start_pose, dir_3d);
    }

    // === 3. 保存 2D 栅格地图 ===
    std::string map_image_rel = "2d/" + time_str + "/map.png";
    std::string map_yaml_rel = "2d/" + time_str + "/map.yaml";
    if (options_.with_gridmap_ && g2p5_ != nullptr) {
        auto map = g2p5_->GetNewestMap()->ToROS();
        int width = map.info.width;
        int height = map.info.height;

        cv::Mat nav_image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int8_t data = map.data[y * width + x];
                nav_image.at<uchar>(height - 1 - y, x) = (data == 0) ? 255 : (data == 100) ? 0 : 128;
            }
        }
        cv::imwrite(dir_2d + "map.png", nav_image);

        YAML::Emitter e;
        e << YAML::BeginMap;
        e << YAML::Key << "image" << YAML::Value << "map.png";
        e << YAML::Key << "mode" << YAML::Value << "trinary";
        e << YAML::Key << "width" << YAML::Value << width;
        e << YAML::Key << "height" << YAML::Value << height;
        e << YAML::Key << "resolution" << YAML::Value << map.info.resolution;
        e << YAML::Key << "origin" << YAML::Value
          << std::vector<double>{map.info.origin.position.x, map.info.origin.position.y, 0};
        e << YAML::Key << "negate" << YAML::Value << 0;
        e << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
        e << YAML::Key << "free_thresh" << YAML::Value << 0.25;
        e << YAML::EndMap;
        std::ofstream f(dir_2d + "map.yaml");
        f << e.c_str();
        f.close();
    }

    // === 4. 保存 factor_graph.yaml ===
    {
        YAML::Emitter e;
        e << YAML::BeginMap;
        e << YAML::Key << "last_check_kf_global_id" << YAML::Value << last_id;
        e << YAML::Key << "last_loop_success_kf_global_id" << YAML::Value << -1;

        // 运动边
        e << YAML::Key << "motion_edges" << YAML::Value << YAML::BeginSeq;
        for (size_t i = 1; i < all_kfs.size(); ++i) {
            SE3 motion = all_kfs[i - 1]->GetLIOPose().inverse() * all_kfs[i]->GetLIOPose();
            e << YAML::BeginMap;
            e << YAML::Key << "type" << YAML::Value << "motion";
            e << YAML::Key << "from_global_id" << YAML::Value << (int)all_kfs[i - 1]->GetID();
            e << YAML::Key << "to_global_id" << YAML::Value << (int)all_kfs[i]->GetID();
            e << YAML::Key << "measurement" << YAML::Value << SE3ToVec7(motion);
            e << YAML::Key << "information_diagonal" << YAML::Value
              << std::vector<int>{10000, 10000, 10000, 100, 100, 100};
            e << YAML::EndMap;
        }
        e << YAML::EndSeq;

        // 锚点边（第一帧先验）
        e << YAML::Key << "anchor_edges" << YAML::Value << YAML::BeginSeq;
        e << YAML::BeginMap;
        e << YAML::Key << "type" << YAML::Value << "prior";
        e << YAML::Key << "from_global_id" << YAML::Value << 0;
        e << YAML::Key << "to_global_id" << YAML::Value << 0;
        e << YAML::Key << "measurement" << YAML::Value << SE3ToVec7(all_kfs[0]->GetOptPose());
        e << YAML::Key << "information_diagonal" << YAML::Value << std::vector<int>{10000, 10000, 10000, 100, 100, 100};
        e << YAML::EndMap;
        e << YAML::EndSeq;

        e << YAML::EndMap;

        std::ofstream f(save_path + "session/factor_graph.yaml");
        f << e.c_str();
        f.close();
    }

    // === 5. 保存 manifest.yaml ===
    {
        SE3 start_pose = all_kfs.front()->GetOptPose();
        SE3 end_pose = all_kfs.back()->GetOptPose();
        double start_time = all_kfs.front()->GetState().timestamp_;
        double end_time = all_kfs.back()->GetState().timestamp_;

        YAML::Emitter e;
        e << YAML::BeginMap;
        e << YAML::Key << "version" << YAML::Value << 2;
        e << YAML::Key << "front_end" << YAML::Value << (options_.use_super_lio_ ? 3 : 1);
        e << YAML::Key << "back_end" << YAML::Value << 3;
        e << YAML::Key << "map_uuid" << YAML::Value << map_name_;
        e << YAML::Key << "session_count" << YAML::Value << 1;
        e << YAML::Key << "last_session_id" << YAML::Value << 0;
        e << YAML::Key << "keyframe_count" << YAML::Value << (int)all_kfs.size();
        e << YAML::Key << "last_global_id" << YAML::Value << last_id;
        e << YAML::Key << "global_map_file" << YAML::Value << "global.pcd";
        e << YAML::Key << "tile_index_file" << YAML::Value << "index.txt";
        e << YAML::Key << "map_image_file" << YAML::Value << "map.png";

        // multi_floor
        e << YAML::Key << "multi_floor" << YAML::Value << YAML::BeginMap;
        e << YAML::Key << "enabled" << YAML::Value << true;
        e << YAML::Key << "initial_raw_floor_id" << YAML::Value << 1000;
        e << YAML::Key << "normalization_offset" << YAML::Value << 999;
        e << YAML::Key << "floors" << YAML::Value << YAML::BeginSeq;
        e << YAML::BeginMap;
        e << YAML::Key << "raw_floor_id" << YAML::Value << 1000;
        e << YAML::Key << "floor_id" << YAML::Value << 1;
        e << YAML::Key << "representative_z" << YAML::Value << 0.0;
        e << YAML::Key << "first_global_id" << YAML::Value << 0;
        e << YAML::Key << "last_global_id" << YAML::Value << last_id;
        e << YAML::Key << "map_image_file" << YAML::Value << map_image_rel;
        e << YAML::Key << "map_yaml_file" << YAML::Value << map_yaml_rel;
        e << YAML::EndMap;
        e << YAML::EndSeq;
        e << YAML::EndMap;

        // keyframe_ids
        e << YAML::Key << "keyframe_ids" << YAML::Value << YAML::BeginSeq;
        for (auto& kf : all_kfs) {
            e << (int)kf->GetID();
        }
        e << YAML::EndSeq;

        // sessions
        e << YAML::Key << "sessions" << YAML::Value << YAML::BeginSeq;
        e << YAML::BeginMap;
        e << YAML::Key << "session_id" << YAML::Value << 0;
        e << YAML::Key << "start_time" << YAML::Value << start_time;
        e << YAML::Key << "end_time" << YAML::Value << end_time;
        e << YAML::Key << "start_pose" << YAML::Value << SE3ToVec7(start_pose);
        e << YAML::Key << "end_pose" << YAML::Value << SE3ToVec7(end_pose);
        e << YAML::Key << "anchor_pose" << YAML::Value << std::vector<double>{0, 0, 0, 0, 0, 0, 1};
        e << YAML::Key << "anchor_source" << YAML::Value << "localization";
        e << YAML::Key << "first_global_id" << YAML::Value << 0;
        e << YAML::Key << "last_global_id" << YAML::Value << last_id;
        e << YAML::EndMap;
        e << YAML::EndSeq;

        e << YAML::EndMap;

        std::ofstream f(save_path + "session/manifest.yaml");
        f << e.c_str();
        f.close();
    }

    // === 6. 保存 global_map.yaml ===
    {
        // 计算 z 范围
        double min_z = 1e9, max_z = -1e9, sum_z = 0;
        for (auto& kf : all_kfs) {
            double z = kf->GetOptPose().translation().z();
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
            sum_z += z;
        }
        double rep_z = all_kfs.empty() ? 0.0 : sum_z / all_kfs.size();

        YAML::Emitter e;
        e << YAML::BeginMap;
        e << YAML::Key << "version" << YAML::Value << 2;
        e << YAML::Key << "map_uuid" << YAML::Value << map_name_;
        e << YAML::Key << "frame_id" << YAML::Value << "map";
        e << YAML::Key << "multi_floor" << YAML::Value << true;
        e << YAML::Key << "initial_raw_floor_id" << YAML::Value << 1000;
        e << YAML::Key << "normalization_offset" << YAML::Value << 999;

        e << YAML::Key << "floors" << YAML::Value << YAML::BeginSeq;
        e << YAML::BeginMap;
        e << YAML::Key << "floor_id" << YAML::Value << 1;
        e << YAML::Key << "raw_floor_id" << YAML::Value << 1000;
        e << YAML::Key << "representative_z" << YAML::Value << rep_z;
        e << YAML::Key << "min_z" << YAML::Value << min_z;
        e << YAML::Key << "max_z" << YAML::Value << max_z;
        e << YAML::Key << "map_image_file" << YAML::Value << map_image_rel;
        e << YAML::Key << "map_yaml_file" << YAML::Value << map_yaml_rel;
        e << YAML::Key << "keyframes" << YAML::Value << YAML::BeginSeq;
        for (auto& kf : all_kfs) {
            e << (int)kf->GetID();
        }
        e << YAML::EndSeq;
        e << YAML::EndMap;
        e << YAML::EndSeq;

        e << YAML::EndMap;

        std::ofstream f(save_path + "global_map.yaml");
        f << e.c_str();
        f.close();
    }

    LOG(INFO) << "map saved (template) to " << save_path;
}

}  // namespace lightning
