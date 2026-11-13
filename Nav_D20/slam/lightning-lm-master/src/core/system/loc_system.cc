//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include <fstream>
#include <iomanip>
#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"

#include <geometry_msgs/msg/pose.hpp>

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
namespace {
// INS 位置类型常量定义(以 novatel_oem7_msgs/PositionOrVelocityType 的常量值为标准)
// novatel: INS_RTKFLOAT=55, INS_RTKFIXED=56, INS_PSRSP=53, PROPAGATED=19
// kygi665: INS_RTKFLOAT=55, INS_RTKFIXED=56, INS_PSRSP=53, INS=51
// kygi665 的 pos_type.data 需先归一化为 novatel 值再存储/比较
constexpr int INS_POS_TYPE_RTK_FIXED = 56;    // RTK 固定解
constexpr int INS_POS_TYPE_RTK_FLOAT = 55;    // RTK 浮点解
constexpr int INS_POS_TYPE_RTK_DR = 19;       // RTK 航位推算
constexpr int INS_POS_TYPE_SINGLE = 53;       // 单点定位
constexpr int INS_POS_TYPE_INVALID = 3;       // 无效解
constexpr int INS_INIT_TIMEOUT_SECONDS = 30;  // INS 初始化超时回退到 YAML 位姿的时间（秒）

/// 检测地图路径是否存在有效索引（支持旧格式 index.txt 和模板格式 3d/index.txt）
bool MapHasValidIndex(const std::string& map_path) {
    if (map_path.empty()) return false;
    std::string p = map_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    return std::filesystem::exists(p + "/index.txt") || std::filesystem::exists(p + "/3d/index.txt");
}

/// 解析模板格式下的 global.pcd 路径，兼容旧格式
std::string ResolveGlobalPcdPath(const std::string& map_path) {
    std::string p = map_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    if (std::filesystem::exists(p + "/global.pcd")) {
        return p + "/global.pcd";
    }
    return p + "/3d/global.pcd";
}

/// 解析模板格式下的 2d 栅格地图 (map.yaml + map.png) 路径
/// 返回 map.yaml 的完整路径；若未找到返回空字符串
std::string ResolveGridMapYamlPath(const std::string& map_path) {
    std::string p = map_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    // 优先旧格式：根目录下的 map.yaml
    if (std::filesystem::exists(p + "/map.yaml")) {
        return p + "/map.yaml";
    }
    // 模板格式：扫描 2d/ 子目录，取第一个有效的时间戳目录
    std::string dir_2d = p + "/2d";
    if (!std::filesystem::is_directory(dir_2d)) return "";
    for (const auto& entry : std::filesystem::directory_iterator(dir_2d)) {
        if (entry.is_directory()) {
            std::string yaml_path = entry.path().string() + "/map.yaml";
            if (std::filesystem::exists(yaml_path)) {
                return yaml_path;
            }
        }
    }
    return "";
}

/// 解析模板格式下的栅格地图图片路径，与 ResolveGridMapYamlPath 配套使用
std::string ResolveGridMapImagePath(const std::string& map_path) {
    std::string p = map_path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    if (std::filesystem::exists(p + "/map.png")) {
        return p + "/map.png";
    }
    if (std::filesystem::exists(p + "/map.pgm")) {
        return p + "/map.pgm";
    }
    // 模板格式：扫描 2d/ 子目录
    std::string dir_2d = p + "/2d";
    if (!std::filesystem::is_directory(dir_2d)) return "";
    for (const auto& entry : std::filesystem::directory_iterator(dir_2d)) {
        if (entry.is_directory()) {
            if (std::filesystem::exists(entry.path().string() + "/map.png")) {
                return entry.path().string() + "/map.png";
            }
            if (std::filesystem::exists(entry.path().string() + "/map.pgm")) {
                return entry.path().string() + "/map.pgm";
            }
        }
    }
    return "";
}

/// 将 kygi665 的 pos_type.data 归一化为 novatel 的 pos_type.type 值
uint32_t NormalizeKygi665PosType(uint32_t kygi665_type) {
    if (kygi665_type == 56) return 56;  // INS_RTKFIXED → RTK 固定解
    if (kygi665_type == 55) return 55;  // INS_RTKFLOAT → RTK 浮点解
    if (kygi665_type == 53) return 53;  // INS_PSRSP → 单点定位
    return 3;                           // 其他值保持不变(常量值一致的如 PROPAGATED=19)
}
}  // namespace

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

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() { loc_->Finish(); }

bool LocSystem::Init(const std::string& yaml_path, bool publish_occ_grid, bool publish_map_cloud,
                     bool auto_set_init_pose) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    YAML_IO yaml(yaml_path);

    std::string map_path = yaml.GetValue<std::string>("system", "map_path");

    // 检查配置的地图路径是否存在有效索引（兼容旧格式 index.txt 和模板格式 3d/index.txt）
    if (!MapHasValidIndex(map_path)) {
        // 回退到安装目录下的内置默认地图: <config所在目录>/../map/0c3f8a2d-...
        std::filesystem::path cfg_path(yaml_path);
        std::string default_map_path = (cfg_path.parent_path() / ".." / "map" / "0c3f8a2d-41eb-416d-8fca-aaad500e2be9")
                                           .lexically_normal()
                                           .string();
        LOG(WARNING) << "配置的地图路径无效(无 index.txt): " << map_path
                     << "，回退到内置默认地图: " << default_map_path;
        map_path = default_map_path;
        using_default_map_ = true;
    }

    // 从 yaml 读取地图发布配置（存在则覆盖函数参数默认值）
    try {
        publish_map_cloud = yaml.GetValue<bool>("system", "publish_map_cloud");
    } catch (...) {
        // yaml 中未配置，保持函数参数默认值
    }
    try {
        publish_occ_grid = yaml.GetValue<bool>("system", "publish_occ_grid");
    } catch (...) {
        // yaml 中未配置，保持函数参数默认值
    }
    if (!map_path.empty()) {
        std::string trimmed = map_path;
        while (!trimmed.empty() && trimmed.back() == '/') {
            trimmed.pop_back();
        }
        size_t pos = trimmed.find_last_of('/');
        current_map_uuid_ = (pos == std::string::npos) ? trimmed : trimmed.substr(pos + 1);
    }

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_slam");

    // 健康上报: 告警/事件发布器
    health_ = std::make_shared<HealthReporter>(node_);

    // 地图无效回退默认地图时, 上报地图数据损坏告警
    if (using_default_map_) {
        health_->SetAlarm(HealthReporter::ALARM_MAP_CORRUPT, true, rhr_interface::msg::AlarmInfo::ALARM_SERIOUS,
                          "地图数据损坏", "加载的地图文件格式错误或关键数据缺失");
    }

    imu_topic_ = yaml.GetValue<std::string>("common", "imu_topic");
    cloud_topic_ = yaml.GetValue<std::string>("common", "lidar_topic");
    livox_topic_ = yaml.GetValue<std::string>("common", "livox_lidar_topic");
    ins_topic_ = yaml.GetValue<std::string>("common", "ins_topic");
    std::string bestgnsspos_topic = yaml.GetValue<std::string>("common", "bestgnsspos_topic");

    // 读取组合导航类型: "novatel"(旧) 或 "kygi665"(新), 默认 "novatel"
    // 注: IMU 统一使用 sensor_msgs/Imu, 仅 INS 数据源随 ins_type 变化
    try {
        ins_type_ = yaml.GetValue<std::string>("common", "ins_type");
    } catch (...) {
        ins_type_ = "novatel";
    }
    try {
        kygi665_ins_topic_ = yaml.GetValue<std::string>("common", "kygi665_ins_topic");
    } catch (...) {
        kygi665_ins_topic_ = "/kygi665/inspvax";
    }
    LOG(INFO) << "组合导航类型: " << ins_type_;

    time_offset = yaml.GetValue<double>("fasterlio", "time_offset");
    try {
        ins_yaw_offset_deg_ = yaml.GetValue<double>("system", "ins_yaw_offset");
    } catch (...) {
        ins_yaw_offset_deg_ = 0.0;
    }
    LOG(INFO) << "kygi665 INS 朝向偏置 system.ins_yaw_offset = " << ins_yaw_offset_deg_ << " deg";
    rclcpp::QoS qos(10);

    // 初始化地理坐标转换器
    options_.with_ins_ = yaml.GetValue<bool>("system", "with_ins");
    options_.ins_only_mode_ = yaml.GetValue<bool>("system", "ins_only_mode");
    double ins_lat = yaml.GetValue<double>("system", "ins_lat");
    double ins_lon = yaml.GetValue<double>("system", "ins_lon");
    double ins_height = yaml.GetValue<double>("system", "ins_height");
    geo_converter_.Reset(ins_lat, ins_lon, ins_height);
    auto extrinsic_T = yaml.GetValue<std::vector<double>>("system", "init_t");
    auto extrinsic_R = yaml.GetValue<std::vector<double>>("system", "init_r");
    yaml_init_pose_ = SE3(math::MatFromArray<double>(extrinsic_R), math::VecFromArray<double>(extrinsic_T));
    // 读取 IMU-body 外参(yaml 中为 body->imu, 求逆得 imu->body)
    auto imu_ext_T = yaml.GetValue<std::vector<double>>("system", "extrinsic_T");
    auto imu_ext_R = yaml.GetValue<std::vector<double>>("system", "extrinsic_R");
    T_imu_body_ = SE3(math::MatFromArray<double>(imu_ext_R), math::VecFromArray<double>(imu_ext_T)).inverse();
    init_start_time_ = std::chrono::steady_clock::now();
    init_pose_set_ = false;

    // 创建回调组，使 IMU、LiDAR、INS 回调可并行处理
    imu_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    lidar_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    ins_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // 创建多线程执行器（3线程），使 IMU、LiDAR、INS 回调组可并行处理
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(rclcpp::ExecutorOptions(), 3);
    executor_->add_node(node_);
    LOG(INFO) << "使用 MultiThreadedExecutor (3线程)，IMU/LiDAR/INS 回调组并行处理";

    // ===== INS 直出模式: 不运行 LIO 和激光定位, 仅订阅 INS 并发布 odom =====
    if (options_.ins_only_mode_) {
        LOG(INFO) << "INS only mode: skip LIO and lidar localization, publish odom from INS directly.";
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/lightning/odom", 10);
        pub_ins_path_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/ins_path", 10);
        pub_map_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/map_cloud", 2);
        occ_grid_pub_ =
            node_->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).reliable().transient_local());
        ins_path_.header.frame_id = "map";
        if (options_.pub_tf_) {
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        }

        // 加载地图(仅用于发布, 不运行定位)
        bool map_ok = loc_->Init(yaml_path, map_path);
        if (map_ok) {
            // 发布地图点云
            if (publish_map_cloud) {
                std::string pcd_path = ResolveGlobalPcdPath(map_path);
                auto map_cloud = CloudPtr(new PointCloudType());
                if (pcl::io::loadPCDFile<PointType>(pcd_path, *map_cloud) == -1) {
                    LOG(ERROR) << "[INS only] 加载 PCD 失败: " << pcd_path;
                } else if (map_cloud->empty()) {
                    LOG(ERROR) << "[INS only] PCD 为空: " << pcd_path;
                } else {
                    auto sampled_cloud = math::VoxelGrid(map_cloud, 0.5);
                    sensor_msgs::msg::PointCloud2 map_msg;
                    pcl::toROSMsg(*sampled_cloud, map_msg);
                    map_msg.header.stamp = node_->now();
                    map_msg.header.frame_id = "map";
                    for (int i = 0; i < 3; i++) {
                        pub_map_cloud_->publish(map_msg);
                        sleep(1);
                    }
                    LOG(INFO) << "[INS only] Published map cloud: " << map_cloud->size() << " -> "
                              << sampled_cloud->size() << " points.";
                }
            }
            // 发布栅格地图
            if (publish_occ_grid && !map_path.empty()) {
                PublishOccupancyGrid(map_path);
            }
        } else {
            LOG(ERROR) << "[INS only] 地图加载失败: " << map_path;
        }

        // 订阅 IMU 和 INS (根据 ins_type_ 选择 novatel 或 kygi665)
        CreateINSOnlySubscriptions(qos);
        return true;
    }

    rclcpp::SubscriptionOptions imu_sub_opts;
    imu_sub_opts.callback_group = imu_cb_group_;

    rclcpp::SubscriptionOptions lidar_sub_opts;
    lidar_sub_opts.callback_group = lidar_cb_group_;

    rclcpp::SubscriptionOptions ins_sub_opts;
    ins_sub_opts.callback_group = ins_cb_group_;

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            LOG_EVERY_N(INFO, 500) << "IMU 回调线程ID: " << std::this_thread::get_id();
            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp) + time_offset;
            // 用 steady_clock 检测回调到达间隔，判断是否被阻塞
            double now_sec = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (last_imu_recv_time_ > 0) {
                double interval = now_sec - last_imu_recv_time_;
                if (interval > 0.04) {
                    LOG(ERROR) << "IMU 回调间隔过大: " << interval << " 秒，可能被阻塞";
                }
            }
            last_imu_recv_time_ = now_sec;
            // 帧率监测(含IMU帧率异常告警上报)
            MonitorIMUFrameRate(imu->timestamp, now_sec);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            // 角速度单位为 deg/s, 转换为 rad/s
            imu->angular_velocity =
                Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z) * M_PI / 180.0;
            ProcessIMU(imu);
        },
        imu_sub_opts);
    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, qos,
        [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            LOG_EVERY_N(INFO, 50) << "LiDAR 回调线程ID: " << std::this_thread::get_id();
            // 计算 LiDAR 时间戳
            double lidar_time = ToSec(cloud->header.stamp) + time_offset;
            double now_sec = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (last_lidar_recv_time_ > 0) {
                double interval = now_sec - last_lidar_recv_time_;
                if (interval > 0.2) {
                    LOG(ERROR) << "LiDAR 回调间隔过大: " << interval << " 秒，可能被阻塞";
                }
            }
            last_lidar_recv_time_ = now_sec;
            // 帧率/点云监测(含帧间隔抖动、点云骤降告警上报)
            MonitorLidarFrameRate(lidar_time, now_sec);
            MonitorLidarCloudSize(cloud->width * cloud->height);
            // 2026-11-13（T49 复核）：回放时 LiDAR 回调隔几秒就卡 0.9 s，/lightning/odom 跟着跳。分清是处理本身慢
            // （下面这条）还是回调没被调度、点云在上游丢了（只有上面"回调间隔过大"、没有这条）。只打日志。
            const auto proc_start = std::chrono::steady_clock::now();
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", false);
            const double proc_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - proc_start).count();
            if (proc_s > 0.2) {
                LOG(WARNING) << "ProcessLidar 用了 " << proc_s << " 秒（点云时间 " << std::fixed
                             << std::setprecision(3) << lidar_time << "）";
            }
        },
        lidar_sub_opts);

    // livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
    //     livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
    //         // 计算 LiDAR 时间戳（Livox 时间戳在第一个点中？使用消息时间戳）
    //         double lidar_time = ToSec(cloud->header.stamp) + time_offset;
    //         double current_time =
    //         std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    //         MonitorLidarFrameRate(lidar_time, current_time);
    //         Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
    //     });
    bool ret = loc_->Init(yaml_path, map_path);
    if (!ret) {
        LOG(ERROR) << "地图加载失败: " << map_path;
        // 地图加载失败, 上报地图数据损坏告警
        health_->SetAlarm(HealthReporter::ALARM_MAP_CORRUPT, true, rhr_interface::msg::AlarmInfo::ALARM_SERIOUS,
                          "地图数据损坏", "加载的地图文件格式错误或关键数据缺失");
    }

    // 传感器断流检测复用 location_timer_(1Hz), 见 PublishLocationTimer 中的 SensorWatchdogTimer()
    watchdog_start_sec_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // INS 订阅
    if (options_.with_ins_) {
        CreateINSSubscription(qos, ins_sub_opts);
    } else {
        // 未使用 INS，直接使用 YAML 初始位姿
        if (auto_set_init_pose) {
            SetInitPose(yaml_init_pose_);
        } else {
            LOG(INFO) << "auto_set_init_pose=false，跳过 YAML 初始位姿设置，等待外部调用设置初始位姿";
        }
    }
    // BESTGNSSPOS 订阅器 - 用于初始化定位
    // sub_bestgnsspos_ = node_->create_subscription<novatel_oem7_msgs::msg::BESTGNSSPOS>(
    //     bestgnsspos_topic, qos, [this](novatel_oem7_msgs::msg::BESTGNSSPOS::SharedPtr msg) {
    //         {
    //             std::lock_guard<std::recursive_mutex> lock(init_mutex_);
    //             if (!init_pose_set_) {
    //                 // 检查是否超时（10秒）
    //                 auto now = std::chrono::steady_clock::now();
    //                 auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - init_start_time_).count();
    //                 if (elapsed >= 10) {
    //                     // 超时，使用 YAML 位姿
    //                     LOG(WARNING) << "INS initialization timeout after " << elapsed << " seconds, using YAML init
    //                     pose."; SetInitPose(yaml_init_pose_);
    //                 } else if (msg->pos_type.type == 50 || msg->pos_type.type == 49 || msg->pos_type.type == 34) {
    //                     // INS 有效，使用 INS 位姿
    //                     LOG(INFO) << "Setting init pose from INS (valid).";
    //                     SetInitPose(SE3(rotation, enu));
    //                 }
    //             }
    //         }
    //     });

    // 点云发布器
    pub_cloud_body_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/current_scan_body", 10);
    pub_cloud_world_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/current_scan_world", 10);
    pub_map_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/lightning/map_cloud", 2);
    pub_lio_path_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/lio_path", 10);
    pub_ins_path_ = node_->create_publisher<nav_msgs::msg::Path>("/lightning/ins_path", 10);
    pub_location_ = node_->create_publisher<nav2_interface::msg::Location>("/nav/location", 10);
    occ_grid_pub_ =
        node_->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).reliable().transient_local());

    // 初始化路径
    lio_path_.header.frame_id = "map";
    ins_path_.header.frame_id = "map";

    // 定位命令服务（包括地图切换）
    // loc_cmd_service_ = node_->create_service<lightning::srv::LocCmd>(
    //     "/lightning/loc_cmd",
    //     std::bind(&LocSystem::HandleLocCmd, this, std::placeholders::_1, std::placeholders::_2));

    if (options_.pub_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        // loc_->SetTFCallback(
        //     [this](const geometry_msgs::msg::TransformStamped &pose) { tf_broadcaster_->sendTransform(pose); });
    }
    if (options_.pub_odom_) {
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/lightning/odom", 10);
        loc_->SetOdometryCallback([this](nav_msgs::msg::Odometry odom) {
            double sys_time = ToSec(node_->now());
            odom.pose.covariance[0] = sys_time;
            // 定位状态通过 covariance[1] 传递: 0=正常, 1=故障
            odom.pose.covariance[1] = loc_->IsLocFailed() ? 0.0 : 1.0;
            {
                std::lock_guard<std::mutex> lock(nav_state_mutex_);
                latest_nav_state_valid_ = loc_->IsLocFailed() ? 0.0 : 1.0;
                latest_odom_ = odom;
            }
            odom_pub_->publish(odom);
            if (tf_broadcaster_) {
                geometry_msgs::msg::TransformStamped tf_msg;
                tf_msg.header = odom.header;
                tf_msg.child_frame_id = "base_link";
                tf_msg.transform.translation.x = odom.pose.pose.position.x;
                tf_msg.transform.translation.y = odom.pose.pose.position.y;
                tf_msg.transform.translation.z = odom.pose.pose.position.z;
                tf_msg.transform.rotation = odom.pose.pose.orientation;
                tf_broadcaster_->sendTransform(tf_msg);
            }
            double now_sec = ToSec(node_->now());
            if (now_sec - last_save_time_ >= 1.0) {
                last_save_time_ = now_sec;
                // 首次写入覆盖旧文件，之后追加
                auto mode = traj_file_inited_ ? std::ios::app : std::ios::out;
                std::ofstream fout("/tmp/lightning_trajectory.txt", mode);
                traj_file_inited_ = true;
                if (fout.is_open()) {
                    fout << std::fixed << std::setprecision(6) << now_sec << " " << odom.pose.pose.position.x << " "
                         << odom.pose.pose.position.y << " " << odom.pose.pose.position.z << " "
                         << odom.pose.pose.orientation.x << " " << odom.pose.pose.orientation.y << " "
                         << odom.pose.pose.orientation.z << " " << odom.pose.pose.orientation.w << "\n";
                    fout.close();
                }
            }
        });
    }

    if (ret) {
        LOG(INFO) << "online loc node has been created.";
        // 设置点云发布回调
        loc_->SetPointcloudBodyCallback([this](const sensor_msgs::msg::PointCloud2& cloud) {
            if (pub_cloud_body_->get_subscription_count() > 0) {
                auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>(cloud);
                pub_cloud_body_->publish(*msg);
            }
        });
        loc_->SetPointcloudWorldCallback([this](const sensor_msgs::msg::PointCloud2& cloud) {
            if (pub_cloud_world_->get_subscription_count() > 0) {
                auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>(cloud);
                pub_cloud_world_->publish(*msg);
            }
        });
        // 设置 LIO 里程计轨迹回调

        // loc_->SetLidarOdomCallback([this](const NavState& state) {
        //     geometry_msgs::msg::PoseStamped pose_stamped;
        //     pose_stamped.header.stamp = math::FromSec(state.timestamp_);
        //     pose_stamped.header.frame_id = "map";
        //     pose_stamped.pose.position.x = state.pos_.x();
        //     pose_stamped.pose.position.y = state.pos_.y();
        //     pose_stamped.pose.position.z = state.pos_.z();
        //     pose_stamped.pose.orientation.x = state.rot_.unit_quaternion().x();
        //     pose_stamped.pose.orientation.y = state.rot_.unit_quaternion().y();
        //     pose_stamped.pose.orientation.z = state.rot_.unit_quaternion().z();
        //     pose_stamped.pose.orientation.w = state.rot_.unit_quaternion().w();
        //     lio_path_.poses.push_back(pose_stamped);
        //     // 限制路径长度，避免内存无限增长
        //     if (lio_path_.poses.size() > 1000) {
        //         lio_path_.poses.erase(lio_path_.poses.begin());
        //     }
        //     lio_path_.header.stamp = pose_stamped.header.stamp;
        //     if (pub_lio_path_->get_subscription_count() > 0) {
        //         pub_lio_path_->publish(lio_path_);
        //     }
        //     // 存储最新状态
        //     {
        //         std::lock_guard<std::mutex> lock(nav_state_mutex_);
        //         latest_nav_state_ = state;
        //         latest_nav_state_valid_ = true;
        //     }
        // });
        // 创建定时器，每秒发布定位消息
        location_timer_ =
            node_->create_wall_timer(std::chrono::seconds(1), std::bind(&LocSystem::PublishLocationTimer, this));

        // 发布地图点云（仅一次）
        if (publish_map_cloud) {
            std::string pcd_path = ResolveGlobalPcdPath(map_path);
            auto map_cloud = CloudPtr(new PointCloudType());
            if (pcl::io::loadPCDFile<PointType>(pcd_path, *map_cloud) == -1 || map_cloud->empty()) {
                LOG(WARNING) << "地图点云为空或加载失败: " << pcd_path;
            } else {
                auto sampled_cloud = math::VoxelGrid(map_cloud, 0.5);
                sensor_msgs::msg::PointCloud2 map_msg;
                pcl::toROSMsg(*sampled_cloud, map_msg);
                map_msg.header.stamp = node_->now();
                map_msg.header.frame_id = "map";
                for (int i = 0; i < 3; i++) {
                    pub_map_cloud_->publish(map_msg);
                    sleep(1);
                }
                LOG(INFO) << "Published map cloud: " << map_cloud->size() << " -> " << sampled_cloud->size()
                          << " points.";
            }
        }
    }

    // 发布栅格地图（自动启动定位时发布，重定位/建图后由 SwitchMapAndReinit 发布）

    if (ret && publish_occ_grid && !map_path.empty()) {
        PublishOccupancyGrid(map_path);
    }

    return ret;
}

void LocSystem::CreateINSSubscription(const rclcpp::QoS& qos, const rclcpp::SubscriptionOptions& ins_sub_opts) {
    if (ins_type_ == "kygi665") {
        LOG(INFO) << "使用 kygi665 组合导航, INS topic: " << kygi665_ins_topic_;
        sub_kygi665_ins_ = node_->create_subscription<kygi665_msgs::msg::INSPVAX>(
            kygi665_ins_topic_, qos,
            [this](kygi665_msgs::msg::INSPVAX::SharedPtr msg) {
                double ins_time = ToSec(msg->header.stamp) + time_offset;
                {
                    double now_sec =
                        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (last_ins_recv_time_ > 0) {
                        double interval = now_sec - last_ins_recv_time_;
                        if (interval > 0.1) {
                            LOG(ERROR) << "INS 回调间隔过大: " << interval << " 秒，可能被阻塞";
                        }
                    }
                    last_ins_recv_time_ = now_sec;
                }

                // 解析 INS 状态 (kygi665 pos_type.data 归一化为 novatel 常量值)
                // 要求 ins_status 为 INS_SOLUTION_GOOD(3) 才可用
                INSPtr ins = std::make_shared<Ins>();
                bool ins_solution_good = (msg->ins_status.data == 3);
                uint32_t pos_type = NormalizeKygi665PosType(msg->pos_type.data);
                if (ins_solution_good && pos_type == INS_POS_TYPE_RTK_FIXED) {
                    ins->state = INS_VALID;
                } else {
                    ins->state = INS_INVALID;
                }

                // 转换为 ENU 坐标
                Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                double azimuth_rad = (-msg->azimuth + ins_yaw_offset_deg_) * M_PI / 180.0;
                Eigen::Matrix3d rotation = (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                            Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                               .toRotationMatrix();
                SE3 pose = SE3(rotation, enu);

                // 存储最新 INS 地理数据
                {
                    std::lock_guard<std::mutex> lock(lla_mutex_);
                    latest_ins_geo_ = {msg->latitude, msg->longitude, msg->height, pos_type};
                    latest_ins_pose_ = pose;
                    ins_pose_valid_ =
                        ins_solution_good && (pos_type == INS_POS_TYPE_RTK_FIXED || pos_type == INS_POS_TYPE_RTK_FLOAT);
                }

                ins->stamp = ins_time;
                ins->pose = pose.matrix();
                ins->position_covariance =
                    Eigen::Vector3d(msg->latitude_stdev, msg->longitude_stdev, msg->height_stdev);
                ins->orientation_covariance = Eigen::Vector3d(msg->roll_stdev, msg->pitch_stdev, msg->azimuth_stdev);

                // 检查是否已设置初始位姿
                {
                    std::lock_guard<std::recursive_mutex> lock(init_mutex_);
                    if (!init_pose_set_) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - init_start_time_).count();
                        if (elapsed >= INS_INIT_TIMEOUT_SECONDS) {
                            LOG(WARNING) << "INS initialization timeout after " << elapsed
                                         << " seconds, using YAML init pose.";
                            SetInitPose(yaml_init_pose_);
                        } else if (ins_solution_good && pos_type == INS_POS_TYPE_RTK_FIXED) {
                            // RTK_FIXED: 厘米级精度，跳过 NDT 配准，直接信任 INS 位姿
                            LOG(INFO) << "Setting init pose from INS (RTK_FIXED, skip NDT).";
                            SetInitPose(SE3(rotation, enu), true);
                        } else if (ins_solution_good && pos_type == INS_POS_TYPE_RTK_FLOAT) {
                            // RTK_FLOAT: 亚米级精度，仍需 NDT 精配准
                            LOG(INFO) << "Setting init pose from INS (RTK_FLOAT).";
                            SetInitPose(SE3(rotation, enu), false);
                        }
                    }
                }

                // 存入缓冲区
                loc_->ins_buffer_.push_back(ins);
                if (loc_->ins_buffer_.size() > 100) {
                    loc_->ins_buffer_.pop_front();
                }

                // 更新 INS 路径
                if (ins->state == INS_VALID) {
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header.stamp = math::FromSec(ins->stamp);
                    pose_stamped.header.frame_id = "map";
                    pose_stamped.pose.position.x = enu.x();
                    pose_stamped.pose.position.y = enu.y();
                    pose_stamped.pose.position.z = enu.z();
                    Eigen::Quaterniond q(rotation);
                    pose_stamped.pose.orientation.x = q.x();
                    pose_stamped.pose.orientation.y = q.y();
                    pose_stamped.pose.orientation.z = q.z();
                    pose_stamped.pose.orientation.w = q.w();
                    ins_path_.poses.push_back(pose_stamped);
                    if (ins_path_.poses.size() > 1000) {
                        ins_path_.poses.erase(ins_path_.poses.begin());
                    }
                    ins_path_.header.stamp = pose_stamped.header.stamp;
                    if (pub_ins_path_->get_subscription_count() > 0) {
                        pub_ins_path_->publish(ins_path_);
                    }
                }
            },
            ins_sub_opts);
    } else {
        LOG(INFO) << "使用 novatel 组合导航, INS topic: " << ins_topic_;
        sub_ins_ = node_->create_subscription<novatel_oem7_msgs::msg::INSPVAX>(
            ins_topic_, qos,
            [this](novatel_oem7_msgs::msg::INSPVAX::SharedPtr msg) {
                double ins_time = ToSec(msg->header.stamp) + time_offset;
                {
                    double now_sec =
                        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (last_ins_recv_time_ > 0) {
                        double interval = now_sec - last_ins_recv_time_;
                        if (interval > 0.1) {
                            LOG(ERROR) << "INS 回调间隔过大: " << interval << " 秒，可能被阻塞";
                        }
                    }
                    last_ins_recv_time_ = now_sec;
                }

                // 解析 INS 状态
                INSPtr ins = std::make_shared<Ins>();
                uint32_t pos_type = msg->pos_type.type;
                if (pos_type == INS_POS_TYPE_RTK_FIXED) {
                    ins->state = INS_VALID;
                } else {
                    ins->state = INS_INVALID;
                }

                // 转换为 ENU 坐标
                Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                double azimuth_rad = (90.0 - msg->azimuth) * M_PI / 180.0;
                Eigen::Matrix3d rotation = (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                            Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                               .toRotationMatrix();
                SE3 pose = SE3(rotation, enu);

                // 存储最新 INS 地理数据
                {
                    std::lock_guard<std::mutex> lock(lla_mutex_);
                    latest_inspvax_ = *msg;
                    latest_ins_geo_ = {msg->latitude, msg->longitude, msg->height, pos_type};
                    latest_ins_pose_ = pose;
                    ins_pose_valid_ = (pos_type == INS_POS_TYPE_RTK_FIXED || pos_type == INS_POS_TYPE_RTK_FLOAT);
                }

                ins->stamp = ins_time;
                ins->pose = pose.matrix();
                ins->position_covariance =
                    Eigen::Vector3d(msg->latitude_stdev, msg->longitude_stdev, msg->height_stdev);
                ins->orientation_covariance = Eigen::Vector3d(msg->roll_stdev, msg->pitch_stdev, msg->azimuth_stdev);

                // 检查是否已设置初始位姿
                {
                    std::lock_guard<std::recursive_mutex> lock(init_mutex_);
                    if (!init_pose_set_) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - init_start_time_).count();
                        if (elapsed >= INS_INIT_TIMEOUT_SECONDS) {
                            LOG(WARNING) << "INS initialization timeout after " << elapsed
                                         << " seconds, using YAML init pose.";
                            SetInitPose(yaml_init_pose_);
                        } else if (pos_type == INS_POS_TYPE_RTK_FIXED) {
                            // RTK_FIXED: 厘米级精度，跳过 NDT 配准，直接信任 INS 位姿
                            LOG(INFO) << "Setting init pose from INS (RTK_FIXED, skip NDT).";
                            SetInitPose(SE3(rotation, enu), true);
                        } else if (pos_type == INS_POS_TYPE_RTK_FLOAT) {
                            // RTK_FLOAT: 亚米级精度，仍需 NDT 精配准
                            LOG(INFO) << "Setting init pose from INS (RTK_FLOAT).";
                            SetInitPose(SE3(rotation, enu), false);
                        }
                    }
                }

                // 存入缓冲区
                loc_->ins_buffer_.push_back(ins);
                if (loc_->ins_buffer_.size() > 100) {
                    loc_->ins_buffer_.pop_front();
                }

                // 更新 INS 路径
                if (ins->state == INS_VALID) {
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header.stamp = math::FromSec(ins->stamp);
                    pose_stamped.header.frame_id = "map";
                    pose_stamped.pose.position.x = enu.x();
                    pose_stamped.pose.position.y = enu.y();
                    pose_stamped.pose.position.z = enu.z();
                    Eigen::Quaterniond q(rotation);
                    pose_stamped.pose.orientation.x = q.x();
                    pose_stamped.pose.orientation.y = q.y();
                    pose_stamped.pose.orientation.z = q.z();
                    pose_stamped.pose.orientation.w = q.w();
                    ins_path_.poses.push_back(pose_stamped);
                    if (ins_path_.poses.size() > 1000) {
                        ins_path_.poses.erase(ins_path_.poses.begin());
                    }
                    ins_path_.header.stamp = pose_stamped.header.stamp;
                    if (pub_ins_path_->get_subscription_count() > 0) {
                        pub_ins_path_->publish(ins_path_);
                    }
                }
            },
            ins_sub_opts);
    }
}

void LocSystem::CreateINSOnlySubscriptions(const rclcpp::QoS& qos) {
    // 订阅 IMU, 仅缓存角速度(body系), 供 odom.twist 使用
    // 注: IMU 统一使用 sensor_msgs/Imu, 不再区分组合导航类型
    rclcpp::SubscriptionOptions imu_sub_opts;
    imu_sub_opts.callback_group = imu_cb_group_;

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(imu_gyro_mutex_);
            // 角速度单位为 deg/s, 转换为 rad/s
            latest_imu_gyro_ =
                Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z) * M_PI / 180.0;
            latest_imu_stamp_ = ToSec(msg->header.stamp) + time_offset;
        },
        imu_sub_opts);

    rclcpp::SubscriptionOptions ins_sub_opts;
    ins_sub_opts.callback_group = ins_cb_group_;
    if (ins_type_ == "kygi665") {
        LOG(INFO) << "[INS only] 使用 kygi665 组合导航, INS topic: " << kygi665_ins_topic_;
        sub_kygi665_ins_ = node_->create_subscription<kygi665_msgs::msg::INSPVAX>(
            kygi665_ins_topic_, qos,
            [this](kygi665_msgs::msg::INSPVAX::SharedPtr msg) {
                double ins_time = ToSec(msg->header.stamp) + time_offset;

                // 要求 ins_status 为 INS_SOLUTION_GOOD(3) 才可用
                bool ins_solution_good = (msg->ins_status.data == 3);
                uint32_t pos_type = NormalizeKygi665PosType(msg->pos_type.data);
                bool is_fixed = ins_solution_good && (pos_type == (uint32_t)INS_POS_TYPE_RTK_FIXED ||
                                                      pos_type == (uint32_t)INS_POS_TYPE_RTK_FLOAT);
                if (is_fixed) {
                    ins_fixed_once_ = true;
                    last_fixed_time_ = ins_time;
                    non_fixed_start_time_ = 0.0;
                } else if (ins_fixed_once_ && pos_type != (uint32_t)INS_POS_TYPE_RTK_DR) {
                    if (non_fixed_start_time_ == 0.0) {
                        non_fixed_start_time_ = ins_time;
                    }
                    if (ins_time - non_fixed_start_time_ > INS_LOSS_TIMEOUT_) {
                        ins_fixed_once_ = false;
                    }
                }
                bool ins_valid = ins_fixed_once_;
                if (!ins_valid) return;

                Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                double azimuth_rad = (-msg->azimuth + ins_yaw_offset_deg_) * M_PI / 180.0;
                Eigen::Matrix3d rotation = (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                            Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                               .toRotationMatrix();
                SE3 pose_imu = SE3(rotation, enu);
                SE3 pose_body = pose_imu * T_imu_body_;

                Eigen::Vector3d vel_enu(msg->east_velocity, msg->north_velocity, msg->up_velocity);
                Eigen::Vector3d vel_imu = rotation.transpose() * vel_enu;
                Eigen::Vector3d gyro_imu = Eigen::Vector3d::Zero();
                {
                    std::lock_guard<std::mutex> lock(imu_gyro_mutex_);
                    gyro_imu = latest_imu_gyro_;
                }
                Eigen::Vector3d t_imu_body = T_imu_body_.translation();
                Eigen::Vector3d vel_body = T_imu_body_.so3().inverse() * (vel_imu - gyro_imu.cross(t_imu_body));
                Eigen::Vector3d gyro_body = T_imu_body_.so3().inverse() * gyro_imu;

                nav_msgs::msg::Odometry odom;
                odom.header.stamp = math::FromSec(ins_time);
                odom.header.frame_id = "map";
                odom.child_frame_id = "base_link";
                odom.pose.pose.position.x = pose_body.translation().x();
                odom.pose.pose.position.y = pose_body.translation().y();
                odom.pose.pose.position.z = pose_body.translation().z();
                Eigen::Quaterniond q_body = pose_body.so3().unit_quaternion();
                odom.pose.pose.orientation.x = q_body.x();
                odom.pose.pose.orientation.y = q_body.y();
                odom.pose.pose.orientation.z = q_body.z();
                odom.pose.pose.orientation.w = q_body.w();
                odom.twist.twist.linear.x = vel_body.x();
                odom.twist.twist.linear.y = vel_body.y();
                odom.twist.twist.linear.z = vel_body.z();
                odom.twist.twist.angular.x = gyro_body.x();
                odom.twist.twist.angular.y = gyro_body.y();
                odom.twist.twist.angular.z = gyro_body.z();
                odom.pose.covariance[0] = ToSec(node_->now());
                odom_pub_->publish(odom);

                double now_sec = ToSec(node_->now());
                if (now_sec - last_save_time_ >= 1.0) {
                    last_save_time_ = now_sec;
                    auto mode = traj_file_inited_ ? std::ios::app : std::ios::out;
                    std::ofstream fout("/tmp/lightning_trajectory.txt", mode);
                    traj_file_inited_ = true;
                    if (fout.is_open()) {
                        fout << std::fixed << std::setprecision(6) << now_sec << " " << pose_body.translation().x()
                             << " " << pose_body.translation().y() << " " << pose_body.translation().z() << " "
                             << q_body.x() << " " << q_body.y() << " " << q_body.z() << " " << q_body.w() << "\n";
                        fout.close();
                    }
                }

                if (tf_broadcaster_) {
                    geometry_msgs::msg::TransformStamped tf_msg;
                    tf_msg.header = odom.header;
                    tf_msg.child_frame_id = "base_link";
                    tf_msg.transform.translation.x = pose_body.translation().x();
                    tf_msg.transform.translation.y = pose_body.translation().y();
                    tf_msg.transform.translation.z = pose_body.translation().z();
                    tf_msg.transform.rotation = odom.pose.pose.orientation;
                    tf_broadcaster_->sendTransform(tf_msg);
                }

                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header = odom.header;
                pose_stamped.pose = odom.pose.pose;
                ins_path_.poses.push_back(pose_stamped);
                if (ins_path_.poses.size() > 1000) ins_path_.poses.erase(ins_path_.poses.begin());
                ins_path_.header.stamp = odom.header.stamp;
                if (pub_ins_path_->get_subscription_count() > 0) pub_ins_path_->publish(ins_path_);
            },
            ins_sub_opts);
    } else {
        // novatel 组合导航
        LOG(INFO) << "[INS only] 使用 novatel 组合导航, INS topic: " << ins_topic_;
        sub_ins_ = node_->create_subscription<novatel_oem7_msgs::msg::INSPVAX>(
            ins_topic_, qos,
            [this](novatel_oem7_msgs::msg::INSPVAX::SharedPtr msg) {
                double ins_time = ToSec(msg->header.stamp) + time_offset;

                bool is_fixed =
                    (msg->pos_type.type == INS_POS_TYPE_RTK_FIXED || msg->pos_type.type == INS_POS_TYPE_RTK_FLOAT);
                if (is_fixed) {
                    ins_fixed_once_ = true;
                    last_fixed_time_ = ins_time;
                    non_fixed_start_time_ = 0.0;
                } else if (ins_fixed_once_ && msg->pos_type.type != INS_POS_TYPE_RTK_DR) {
                    if (non_fixed_start_time_ == 0.0) {
                        non_fixed_start_time_ = ins_time;
                    }
                    if (ins_time - non_fixed_start_time_ > INS_LOSS_TIMEOUT_) {
                        ins_fixed_once_ = false;
                    }
                }
                bool ins_valid = ins_fixed_once_;
                if (!ins_valid) return;

                Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->height);
                Eigen::Vector3d enu = lla_to_enu(lla, geo_converter_);
                double azimuth_rad = (90.0 - msg->azimuth) * M_PI / 180.0;
                Eigen::Matrix3d rotation = (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                            Eigen::AngleAxisd(azimuth_rad, Eigen::Vector3d::UnitZ()))
                                               .toRotationMatrix();
                SE3 pose_imu = SE3(rotation, enu);
                SE3 pose_body = pose_imu * T_imu_body_;

                Eigen::Vector3d vel_enu(msg->east_velocity, msg->north_velocity, msg->up_velocity);
                Eigen::Vector3d vel_imu = rotation.transpose() * vel_enu;
                Eigen::Vector3d gyro_imu = Eigen::Vector3d::Zero();
                {
                    std::lock_guard<std::mutex> lock(imu_gyro_mutex_);
                    gyro_imu = latest_imu_gyro_;
                }
                Eigen::Vector3d t_imu_body = T_imu_body_.translation();
                Eigen::Vector3d vel_body = T_imu_body_.so3().inverse() * (vel_imu - gyro_imu.cross(t_imu_body));
                Eigen::Vector3d gyro_body = T_imu_body_.so3().inverse() * gyro_imu;

                nav_msgs::msg::Odometry odom;
                odom.header.stamp = math::FromSec(ins_time);
                odom.header.frame_id = "map";
                odom.child_frame_id = "base_link";
                odom.pose.pose.position.x = pose_body.translation().x();
                odom.pose.pose.position.y = pose_body.translation().y();
                odom.pose.pose.position.z = pose_body.translation().z();
                Eigen::Quaterniond q_body = pose_body.so3().unit_quaternion();
                odom.pose.pose.orientation.x = q_body.x();
                odom.pose.pose.orientation.y = q_body.y();
                odom.pose.pose.orientation.z = q_body.z();
                odom.pose.pose.orientation.w = q_body.w();
                odom.twist.twist.linear.x = vel_body.x();
                odom.twist.twist.linear.y = vel_body.y();
                odom.twist.twist.linear.z = vel_body.z();
                odom.twist.twist.angular.x = gyro_body.x();
                odom.twist.twist.angular.y = gyro_body.y();
                odom.twist.twist.angular.z = gyro_body.z();
                odom.pose.covariance[0] = ToSec(node_->now());
                odom_pub_->publish(odom);

                double now_sec = ToSec(node_->now());
                if (now_sec - last_save_time_ >= 1.0) {
                    last_save_time_ = now_sec;
                    auto mode = traj_file_inited_ ? std::ios::app : std::ios::out;
                    std::ofstream fout("/tmp/lightning_trajectory.txt", mode);
                    traj_file_inited_ = true;
                    if (fout.is_open()) {
                        fout << std::fixed << std::setprecision(6) << now_sec << " " << pose_body.translation().x()
                             << " " << pose_body.translation().y() << " " << pose_body.translation().z() << " "
                             << q_body.x() << " " << q_body.y() << " " << q_body.z() << " " << q_body.w() << "\n";
                        fout.close();
                    }
                }

                if (tf_broadcaster_) {
                    geometry_msgs::msg::TransformStamped tf_msg;
                    tf_msg.header = odom.header;
                    tf_msg.child_frame_id = "base_link";
                    tf_msg.transform.translation.x = pose_body.translation().x();
                    tf_msg.transform.translation.y = pose_body.translation().y();
                    tf_msg.transform.translation.z = pose_body.translation().z();
                    tf_msg.transform.rotation = odom.pose.pose.orientation;
                    tf_broadcaster_->sendTransform(tf_msg);
                }

                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header = odom.header;
                pose_stamped.pose = odom.pose.pose;
                ins_path_.poses.push_back(pose_stamped);
                if (ins_path_.poses.size() > 1000) ins_path_.poses.erase(ins_path_.poses.begin());
                ins_path_.header.stamp = odom.header.stamp;
                if (pub_ins_path_->get_subscription_count() > 0) pub_ins_path_->publish(ins_path_);
            },
            ins_sub_opts);
    }
}

void LocSystem::SetInitPose(const SE3& pose, bool skip_ndt) {
    std::lock_guard<std::recursive_mutex> lock(init_mutex_);
    if (init_pose_set_) {
        LOG(WARNING) << "Init pose already set, ignoring duplicate call.";
        return;
    }
    LOG(INFO) << "set init pose" << (skip_ndt ? " (skip NDT)" : "") << ": " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();
    loc_->SetExternalPose(pose.unit_quaternion(), pose.translation(), skip_ndt);
    loc_started_ = true;
    init_pose_set_ = true;
}

void LocSystem::ProcessIMU(const IMUPtr& imu) {
    if (loc_started_) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}

void LocSystem::Spin() {
    if (executor_ != nullptr) {
        executor_->spin();
    }
}

void LocSystem::SpinSome() {
    if (executor_ != nullptr) {
        executor_->spin_some();
    }
}

void LocSystem::CancelSpin() {
    if (executor_ != nullptr) {
        executor_->cancel();
    }
}

rclcpp::TimerBase::SharedPtr LocSystem::CreateWallTimer(std::chrono::nanoseconds period,
                                                        std::function<void()> callback) {
    if (node_ != nullptr) {
        return node_->create_wall_timer(period, callback, imu_cb_group_);
    }
    return nullptr;
}
void LocSystem::CheckInitTimeout() {
    if (loc_started_) return;
    std::lock_guard<std::recursive_mutex> lock(init_mutex_);
    if (init_pose_set_) return;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - init_start_time_).count();
    if (elapsed >= INS_INIT_TIMEOUT_SECONDS) {
        LOG(WARNING) << "定位初始化超时 (" << elapsed << "s)，使用 YAML 位姿启动定位";
        SetInitPose(yaml_init_pose_);
    }
}
void LocSystem::SensorWatchdogTimer() {
    // 1Hz 看门狗: IMU/点云完全断流时由本定时器触发告警(数据回调不触发也能发布)
    double now_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double elapsed = now_sec - watchdog_start_sec_;

    // IMU 断流 → 复用"上装IMU帧率异常"告警(帧率为0是帧率异常的极端情况)
    bool imu_lost;
    if (imu_cnt_ == 0) {
        imu_lost = elapsed > SENSOR_STARTUP_GRACE_;  // 启动后从未收到数据
    } else {
        imu_lost = (now_sec - last_imu_recv_time_) > SENSOR_LOST_TIMEOUT_;
    }
    if (imu_lost) {
        LOG(ERROR) << "IMU 数据中断超过 " << SENSOR_LOST_TIMEOUT_ << " 秒";
    }
    health_->SetAlarm(HealthReporter::ALARM_IMU_FRAME_RATE, imu_lost,
                      rhr_interface::msg::AlarmInfo::ALARM_SERIOUS, "上装IMU帧率异常",
                      imu_lost ? "检测到上装IMU帧率异常" : "IMU数据恢复正常");

    // 点云断流 → 复用"激光雷达帧间隔抖动"告警(间隔无穷大)
    bool lidar_lost;
    if (lidar_cnt_ == 0) {
        lidar_lost = elapsed > SENSOR_STARTUP_GRACE_;
    } else {
        lidar_lost = (now_sec - last_lidar_recv_time_) > SENSOR_LOST_TIMEOUT_;
    }
    if (lidar_lost) {
        LOG(ERROR) << "点云数据中断超过 " << SENSOR_LOST_TIMEOUT_ << " 秒";
    }
    health_->SetAlarm(HealthReporter::ALARM_LIDAR_JITTER, lidar_lost,
                      rhr_interface::msg::AlarmInfo::ALARM_GENERAL, "帧间隔抖动",
                      lidar_lost ? "帧间隔大于理论值的150%" : "点云数据恢复正常");
}

void LocSystem::MonitorIMUFrameRate(double sensor_time, double sys_time) {
    const double expected_interval = 0.01;  // 100 Hz
    if (first_imu_time_ == 0.0) {
        first_imu_time_ = sensor_time;
        last_imu_time_ = sensor_time;
        first_imu_sys_time_ = sys_time;
        last_imu_sys_time_ = sys_time;
        imu_cnt_ = 1;
        return;
    }
    double interval = sensor_time - last_imu_time_;
    double sys_interval = sys_time - last_imu_sys_time_;
    last_imu_time_ = sensor_time;
    last_imu_sys_time_ = sys_time;
    ++imu_cnt_;
    // 丢帧检测（使用传感器时间）
    if (interval > expected_interval * 1.5) {
        LOG(WARNING) << "IMU 丢帧！间隔 " << interval << " 秒，超过 " << expected_interval * 1.5 << " 秒";
    }
    // 零间隔检测（卡帧）
    if (interval < 1e-6) {
        LOG(WARNING) << "IMU 卡帧！时间戳未变化";
    }
    // 时间戳倒流
    if (interval < 0.0) {
        LOG(WARNING) << "IMU 时间戳倒流！间隔 " << interval << " 秒";
    }
    // 粘包检测（间隔过短）
    if (interval > 0.0 && interval < expected_interval * 0.5) {
        LOG(WARNING) << "IMU 粘包！间隔 " << interval << " 秒，过短";
    }
    // 每1000帧统计一次平均帧率（使用系统时间）
    if (imu_cnt_ % 1000 == 0) {
        double total_sys_time = sys_time - first_imu_sys_time_;
        if (total_sys_time > 0.0) {
            double avg_freq = imu_cnt_ / total_sys_time;
            LOG(INFO) << "IMU 平均频率: " << avg_freq << " Hz";
        }
    }
    // 帧率评估: 每5秒计算窗口平均帧率, 偏离[80,120]Hz时上报告警, 恢复后清除 (0x011ANN02)
    if (imu_freq_check_time_ == 0.0) {
        imu_freq_check_time_ = sys_time;
        imu_win_cnt_ = imu_cnt_;
    } else if (sys_time - imu_freq_check_time_ >= 5.0) {
        double avg_freq = (imu_cnt_ - imu_win_cnt_) / (sys_time - imu_freq_check_time_);
        imu_freq_check_time_ = sys_time;
        imu_win_cnt_ = imu_cnt_;
        bool abnormal = (avg_freq < 80.0);
        if (abnormal) {
            LOG(WARNING) << "IMU 帧率异常: " << avg_freq << " Hz";
        }
        health_->SetAlarm(HealthReporter::ALARM_IMU_FRAME_RATE, abnormal, rhr_interface::msg::AlarmInfo::ALARM_SERIOUS,
                          "上装IMU帧率异常",
                          abnormal ? "检测到上装IMU帧率异常": "IMU帧率恢复正常");
    }
}

void LocSystem::MonitorLidarFrameRate(double sensor_time, double sys_time) {
    const double expected_interval = 0.1;  // 10 Hz
    if (first_lidar_time_ == 0.0) {
        first_lidar_time_ = sensor_time;
        last_lidar_time_ = sensor_time;
        first_lidar_sys_time_ = sys_time;
        last_lidar_sys_time_ = sys_time;
        lidar_cnt_ = 1;
        return;
    }
    double interval = sensor_time - last_lidar_time_;
    last_lidar_time_ = sensor_time;
    last_lidar_sys_time_ = sys_time;
    ++lidar_cnt_;
    if (interval > expected_interval * 1.5) {
        LOG(WARNING) << "LiDAR 丢帧！间隔 " << interval << " 秒，超过 " << expected_interval * 1.5 << " 秒";
    }
    if (interval < 1e-6) {
        LOG(WARNING) << "LiDAR 卡帧！时间戳未变化";
    }
    if (interval < 0.0) {
        LOG(WARNING) << "LiDAR 时间戳倒流！间隔 " << interval << " 秒";
    }
    if (interval > 0.0 && interval < expected_interval * 0.5) {
        LOG(WARNING) << "LiDAR 粘包！间隔 " << interval << " 秒，过短";
    }
    // 帧间隔抖动检测: 间隔超出[0.5,1.5]倍期望值为抖动, 连续5帧抖动上报, 连续10帧正常恢复 (0x0103NN08)
    bool jitter = (interval <= 0.0 || interval < expected_interval * 0.5 || interval > expected_interval * 1.5);
    if (jitter) {
        if (lidar_jitter_cnt_ < 5) {
            ++lidar_jitter_cnt_;
        }
    } else if (lidar_jitter_cnt_ > 0) {
        --lidar_jitter_cnt_;
    }
    bool jitter_alarm = (lidar_jitter_cnt_ >= 5);
    health_->SetAlarm(
        HealthReporter::ALARM_LIDAR_JITTER, jitter_alarm, rhr_interface::msg::AlarmInfo::ALARM_GENERAL,
        "帧间隔抖动",
        jitter_alarm ? "帧间隔大于理论值的150%" : "点云帧间隔恢复正常");
    if (lidar_cnt_ % 100 == 0) {
        double total_sys_time = sys_time - first_lidar_sys_time_;
        if (total_sys_time > 0.0) {
            double avg_freq = lidar_cnt_ / total_sys_time;
            LOG(INFO) << "LiDAR 平均频率: " << avg_freq << " Hz";
        }
    }
}

void LocSystem::MonitorLidarCloudSize(size_t cloud_size) {
    // 点云数目基线: 指数滑动平均, 异常帧不参与更新, 避免污染基线
    if (lidar_cloud_baseline_ <= 0.0) {
        lidar_cloud_baseline_ = static_cast<double>(cloud_size);
        return;
    }
    bool dropped = (static_cast<double>(cloud_size) < lidar_cloud_baseline_ * 0.5);
    if (dropped) {
        if (lidar_cloud_drop_cnt_ < 3) {
            ++lidar_cloud_drop_cnt_;
        }
        if (lidar_cloud_drop_cnt_ >= 3) {
            LOG(WARNING) << "点云数据骤降: " << cloud_size << " (基线 " << lidar_cloud_baseline_ << ")";
        }
    } else {
        if (lidar_cloud_drop_cnt_ > 0) {
            --lidar_cloud_drop_cnt_;
        }
        lidar_cloud_baseline_ = 0.95 * lidar_cloud_baseline_ + 0.05 * static_cast<double>(cloud_size);
    }
    // 连续3帧骤降上报, 连续10帧正常恢复 (0x0103NN04)
    bool drop_alarm = (lidar_cloud_drop_cnt_ >= 5);
    health_->SetAlarm(HealthReporter::ALARM_LIDAR_CLOUD_DROP, drop_alarm, rhr_interface::msg::AlarmInfo::ALARM_GENERAL,
                      "点云数据骤降",
                      drop_alarm ? ("点云数目骤降至基线50%以下: " + std::to_string(cloud_size)) : "点云数据恢复正常");
}

void LocSystem::MonitorINSFrameRate(double sensor_time, double sys_time) {
    const double expected_interval = 0.01;  // 100 Hz
    if (first_ins_time_ == 0.0) {
        first_ins_time_ = sensor_time;
        last_ins_time_ = sensor_time;
        first_ins_sys_time_ = sys_time;
        last_ins_sys_time_ = sys_time;
        ins_cnt_ = 1;
        return;
    }
    double interval = sensor_time - last_ins_time_;
    last_ins_time_ = sensor_time;
    last_ins_sys_time_ = sys_time;
    ++ins_cnt_;
    if (interval > expected_interval * 1.5) {
        LOG(WARNING) << "INS 丢帧！间隔 " << interval << " 秒，超过 " << expected_interval * 1.5 << " 秒";
    }
    if (interval < 1e-6) {
        LOG(WARNING) << "INS 卡帧！时间戳未变化";
    }
    if (interval < 0.0) {
        LOG(WARNING) << "INS 时间戳倒流！间隔 " << interval << " 秒";
    }
    if (interval > 0.0 && interval < expected_interval * 0.5) {
        LOG(WARNING) << "INS 粘包！间隔 " << interval << " 秒，过短";
    }
    if (ins_cnt_ % 1000 == 0) {
        double total_sys_time = sys_time - first_ins_sys_time_;
        if (total_sys_time > 0.0) {
            double avg_freq = ins_cnt_ / total_sys_time;
            LOG(INFO) << "INS 平均频率: " << avg_freq << " Hz";
        }
    }
}

void LocSystem::PublishLocationTimer() {
    // 备份安全网：如果 LIO 触发了自动重初始化且上层还没提供初始位姿，尝试用 INS 覆盖
    if (loc_started_ && loc_->WasReinitTriggered() && !loc_->IsInitPoseSet()) {
        std::lock_guard<std::mutex> lock(lla_mutex_);
        if (ins_pose_valid_ && latest_ins_geo_.pos_type == INS_POS_TYPE_RTK_FIXED) {
            LOG(INFO) << "PublishLocationTimer: reinit detected, using INS RTK_FIXED pose (skip NDT).";
            loc_->SetExternalPose(latest_ins_pose_.unit_quaternion(), latest_ins_pose_.translation(), true);
        }
    }

    // 创建 Location 消息
    nav2_interface::msg::Location location_msg;
    location_msg.header.frame_id = "map";
    location_msg.pose_geometry.map_uuid = current_map_uuid_;  // 当前加载的地图UUID
    // 设置几何位姿 (geometry pose)
    // 地图加载失败和初始化失败均上报定位失败
    if (using_default_map_) {
        // 使用内置默认地图时，定位结果视为无效
        location_msg.pose_geometry.status = common_interface::msg::GeometryPose::STATE_MAP_FAILED;
    } else if (latest_nav_state_valid_ && loc_started_) {
        location_msg.pose_geometry.status = common_interface::msg::GeometryPose::STATE_MAP_SUCCESS;
        {
            std::lock_guard<std::mutex> lock(nav_state_mutex_);
            location_msg.header.stamp = latest_odom_.header.stamp;
            location_msg.pose_geometry.point.x = latest_odom_.pose.pose.position.x;
            location_msg.pose_geometry.point.y = latest_odom_.pose.pose.position.y;
            location_msg.pose_geometry.point.z = latest_odom_.pose.pose.position.z;
            location_msg.pose_geometry.orient_enable = true;
            location_msg.pose_geometry.orient.x = latest_odom_.pose.pose.orientation.x;
            location_msg.pose_geometry.orient.y = latest_odom_.pose.pose.orientation.y;
            location_msg.pose_geometry.orient.z = latest_odom_.pose.pose.orientation.z;
            location_msg.pose_geometry.orient.w = latest_odom_.pose.pose.orientation.w;
        }
        // 地理位姿：使用最新的INS数据
        {
            std::lock_guard<std::mutex> lock(lla_mutex_);
            if (latest_ins_geo_.pos_type == INS_POS_TYPE_RTK_FIXED ||
                latest_ins_geo_.pos_type == INS_POS_TYPE_RTK_FLOAT) {
                location_msg.pose_geograpic.status = 2;
                location_msg.pose_geograpic.orient_enable = true;
            } else {
                location_msg.pose_geograpic.status = -1;
                location_msg.pose_geograpic.orient_enable = false;
            }
            location_msg.pose_geograpic.latitude = latest_ins_geo_.latitude;
            location_msg.pose_geograpic.longitude = latest_ins_geo_.longitude;
            location_msg.pose_geograpic.altitude = latest_ins_geo_.height;
        }

        // 速度
        // auto vel_b = state.rot_ * state.vel_;
        location_msg.twist.linear.x = latest_odom_.twist.twist.linear.x;
        location_msg.twist.linear.y = latest_odom_.twist.twist.linear.y;
        location_msg.twist.linear.z = latest_odom_.twist.twist.linear.z;
        location_msg.twist.angular.x = ToSec(node_->now());
        location_msg.twist.angular.y = latest_odom_.twist.twist.angular.y;
        location_msg.twist.angular.z = latest_odom_.twist.twist.angular.z;
    } else if (!loc_started_) {
        location_msg.pose_geometry.status = common_interface::msg::GeometryPose::STATUS_MAP_LOCATING;
    } else {
        location_msg.pose_geometry.status = common_interface::msg::GeometryPose::STATE_MAP_FAILED;
    }

    // 健康上报: 传感器断流检测(IMU/点云完全断流时由本定时器兜底, 不依赖数据回调)
    if (health_) {
        SensorWatchdogTimer();
    }

    // 健康上报: 定位丢失/重定位成功/重定位失败/地图匹配失败事件
    if (health_) {
        bool localized = !using_default_map_ && loc_started_ && latest_nav_state_valid_;
        double now_sec = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (localized) {
            // 定位恢复正常: 此前上报过异常才发布重定位成功
            if (loc_lost_reported_ || reloc_fail_reported_ || map_match_fail_reported_) {
                health_->PublishEvent(HealthReporter::EVENT_RELOC_SUCCESS, rhr_interface::msg::EventInfo::EVENT_SERIOUS,
                                      "重定位成功", "定位恢复正常");
            }
            loc_lost_reported_ = false;
            reloc_fail_reported_ = false;
            map_match_fail_reported_ = false;
            loc_lost_since_ = 0.0;
        } else if (loc_started_) {
            if (loc_lost_since_ == 0.0) {
                loc_lost_since_ = now_sec;               // 定位异常计时开始
                lost_after_localized_ = was_localized_;  // 区分"丢失"与"从未匹配成功"
            }
            double lost_dur = now_sec - loc_lost_since_;
            if (lost_after_localized_) {
                // 定位后丢失: 去抖1s上报定位丢失, 持续10s上报重定位失败
                if (!loc_lost_reported_ && lost_dur >= LOC_LOST_DEBOUNCE_) {
                    health_->PublishEvent(HealthReporter::EVENT_LOC_LOST, rhr_interface::msg::EventInfo::EVENT_SERIOUS,
                                          "定位丢失", "机器人定位丢失");
                    loc_lost_reported_ = true;
                } else if (loc_lost_reported_ && !reloc_fail_reported_ && lost_dur >= RELOC_FAIL_TIMEOUT_) {
                    health_->PublishEvent(HealthReporter::EVENT_RELOC_FAIL,
                                          rhr_interface::msg::EventInfo::EVENT_SERIOUS, "重定位失败",
                                          "机器人重定位失败");
                    reloc_fail_reported_ = true;
                }
            } else if (!map_match_fail_reported_ && lost_dur >= MAP_MATCH_FAIL_TIMEOUT_) {
                // 定位启动后一直未能匹配成功
                health_->PublishEvent(HealthReporter::EVENT_MAP_MATCH_FAIL,
                                      rhr_interface::msg::EventInfo::EVENT_SERIOUS, "地图匹配失败",
                                      "导航时未匹配到当前区域的地图");
                map_match_fail_reported_ = true;
            }
        }
        was_localized_ = localized;
    }

    // 发布
    pub_location_->publish(location_msg);
}
void LocSystem::PublishOccupancyGrid(const std::string& map_path) {
    // 兼容旧格式（map.yaml 在根目录）和模板格式（map.yaml 在 2d/<timestamp>/）
    std::string yaml_path = ResolveGridMapYamlPath(map_path);
    std::string image_path = ResolveGridMapImagePath(map_path);
    if (!std::filesystem::exists(yaml_path) || !std::filesystem::exists(image_path)) {
        LOG(WARNING) << "栅格地图文件不存在: " << yaml_path << " 或 " << image_path;
        return;
    }

    try {
        // 解析 map.yaml
        YAML::Node yaml = YAML::LoadFile(yaml_path);
        double resolution = yaml["resolution"].as<double>(0.1);
        auto origin_seq = yaml["origin"].as<std::vector<double>>(std::vector<double>{0.0, 0.0, 0.0});
        double occupied_thresh = yaml["occupied_thresh"].as<double>(0.65);
        double free_thresh = yaml["free_thresh"].as<double>(0.25);

        // 读取地图图片（灰度）
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            LOG(ERROR) << "无法读取地图图片: " << image_path;
            return;
        }

        // width/height 优先从 yaml 读取，缺失时回退到图片尺寸
        int width = yaml["width"] ? yaml["width"].as<int>() : img.cols;
        int height = yaml["height"] ? yaml["height"].as<int>() : img.rows;

        // 构建 OccupancyGrid 消息
        auto grid_msg = std::make_unique<nav_msgs::msg::OccupancyGrid>();
        grid_msg->header.stamp = node_->now();
        grid_msg->header.frame_id = "map";
        grid_msg->info.resolution = resolution;
        grid_msg->info.width = width;
        grid_msg->info.height = height;
        grid_msg->info.origin.position.x = origin_seq[0];
        grid_msg->info.origin.position.y = origin_seq[1];
        grid_msg->info.origin.position.z = origin_seq.size() > 2 ? origin_seq[2] : 0.0;
        grid_msg->info.origin.orientation.w = 1.0;

        // 像素值转 OccupancyGrid data
        // ROS 约定: 0=free(白), 100=occupied(黑), -1=unknown(灰)
        grid_msg->data.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // 图片行从上到下，ROS 地图行从下到上，需要翻转 Y 轴
                int img_y = height - 1 - y;
                int pixel = img.at<uchar>(img_y, x);
                int8_t occ;
                if (pixel >= 255 * (1.0 - free_thresh)) {
                    occ = 0;  // free
                } else if (pixel <= 255 * (1.0 - occupied_thresh)) {
                    occ = 100;  // occupied
                } else {
                    occ = -1;  // unknown
                }
                grid_msg->data[y * width + x] = occ;
            }
        }

        occ_grid_pub_->publish(std::move(grid_msg));
        LOG(INFO) << "已发布栅格地图: " << width << "x" << height << ", resolution=" << resolution
                  << ", map_path=" << map_path;
    } catch (const std::exception& e) {
        LOG(ERROR) << "发布栅格地图失败: " << e.what();
    }
}
// void LocSystem::HandleLocCmd(const lightning::srv::LocCmd::Request::SharedPtr request,
//                       lightning::srv::LocCmd::Response::SharedPtr response) {
//     LOG(INFO) << "Received loc cmd request: id=" << (int)request->id
//               << ", map_id=" << request->map_id
//               << ", x=" << request->x << ", y=" << request->y << ", z=" << request->z;

//     if (!loc_) {
//         response->result = 1; // 非零表示失败
//         LOG(ERROR) << "Localization not initialized";
//         return;
//     }

//     // 根据命令ID处理
//     switch (request->id) {
//         case lightning::srv::LocCmd::Request::CMD_LOC_SWITCH_MAP_INIT: {
//             // 切换地图并重新初始化位姿
//             // 方向保持为零（单位四元数）
//             Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
//             Eigen::Vector3d t(request->x, request->y, request->z);
//             SE3 init_pose(q, t);

//             // 调用 localization 的地图切换和重新初始化函数
//             // switch_map 设为 true
//             bool success = loc_->SwitchMapAndReinit(request->map_id, init_pose, true);
//             if (success) {
//                 response->result = 0; // 成功
//                 LOG(INFO) << "Map switch and reinitialization successful";
//             } else {
//                 response->result = 2; // 其他失败
//                 LOG(ERROR) << "Map switch and reinitialization failed";
//             }
//             break;
//         }
//         // 其他命令可以在这里扩展
//         default:
//             response->result = 3; // 未知命令
//             LOG(ERROR) << "Unknown loc cmd id: " << (int)request->id;
//             break;
//     }
// }

void LocSystem::LockInitPose() {
    std::lock_guard<std::recursive_mutex> lock(init_mutex_);
    // 标记初始位姿已设置，阻止 INS 回调在 SpinSome 期间抢占初始化
    init_pose_set_ = true;
    LOG(INFO) << "已锁定初始位姿状态，阻止 INS 回调自动初始化";
}

bool LocSystem::SwitchMapAndReinit(const std::string& map_id, const SE3& init_pose) {
    if (!loc_) {
        LOG(ERROR) << "Localization not initialized";
        return false;
    }
    // 从 map_id 提取 UUID（最后一级目录名），如 "data/xxx-xxx" → "xxx-xxx"
    {
        std::string trimmed = map_id;
        while (!trimmed.empty() && trimmed.back() == '/') {
            trimmed.pop_back();
        }
        size_t pos = trimmed.find_last_of('/');
        current_map_uuid_ = (pos == std::string::npos) ? trimmed : trimmed.substr(pos + 1);
    }
    // 优先使用 INS 位姿（若有效），否则使用 backup 位姿（如 YAML init_t 或 SLAM 关键帧）
    SE3 pose_to_use = init_pose;
    {
        std::lock_guard<std::mutex> lock(lla_mutex_);
        if (ins_pose_valid_) {
            pose_to_use = latest_ins_pose_;
            LOG(INFO) << "使用 INS 位姿重定位: " << pose_to_use.translation().transpose();
        }
    }
    bool success = loc_->SwitchMapAndReinit(map_id, pose_to_use);
    if (success) {
        LOG(INFO) << "Map switch and reinitialization successful";
        // 切换到新地图成功，清除默认地图标志
        using_default_map_ = false;
        // 重置初始化状态，防止 INS 超时逻辑用 YAML 位姿覆盖重定位的 init_pose
        {
            std::lock_guard<std::recursive_mutex> lock(init_mutex_);
            init_pose_set_ = true;                                // 标记初始位姿已设置，阻止 INS 超时分支
            init_start_time_ = std::chrono::steady_clock::now();  // 重置超时计时基准
        }
        loc_started_ = true;
        // 重置健康上报的定位事件状态(新地图重新开始匹配, 避免切换期误报定位丢失)
        was_localized_ = false;
        loc_lost_reported_ = false;
        reloc_fail_reported_ = false;
        map_match_fail_reported_ = false;
        loc_lost_since_ = 0.0;
        // 发布栅格地图
        PublishOccupancyGrid(map_id);
        // 新地图加载成功, 清除地图数据损坏告警
        health_->SetAlarm(HealthReporter::ALARM_MAP_CORRUPT, false, rhr_interface::msg::AlarmInfo::ALARM_SERIOUS,
                          "地图数据损坏", "加载的地图文件格式错误或关键数据缺失");
    } else {
        LOG(ERROR) << "Map switch and reinitialization failed";
        // 地图切换失败, 上报地图数据损坏告警
        health_->SetAlarm(HealthReporter::ALARM_MAP_CORRUPT, true, rhr_interface::msg::AlarmInfo::ALARM_SERIOUS,
                          "地图数据损坏", "加载的地图文件格式错误或关键数据缺失");
    }
    return success;
}

bool LocSystem::IsLocalized() const { return loc_started_ && latest_nav_state_valid_; }

int LocSystem::GetLocStatus() const {
    if (loc_started_ && latest_nav_state_valid_) {
        return static_cast<int>(loc::LocalizationStatus::GOOD);
    } else if (!loc_started_) {
        return static_cast<int>(loc::LocalizationStatus::INITIALIZING);
    }
    return static_cast<int>(loc::LocalizationStatus::FAIL);
}

double LocSystem::GetConfidence() const {
    if (loc_) {
        return loc_->GetConfidence();
    }
    return 0.0;
}

bool LocSystem::IsNDTInited() const {
    if (loc_) {
        return loc_->IsNDTInited();
    }
    return false;
}

}  // namespace lightning