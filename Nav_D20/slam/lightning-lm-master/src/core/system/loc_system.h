//
// Created by xiang on 25-9-8.
//

#ifndef LIGHTNING_LOC_SYSTEM_H
#define LIGHTNING_LOC_SYSTEM_H

#include <tf2_ros/transform_broadcaster.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/ins.h"
#include "common/keyframe.h"
// #include "msg_interfaces/msg/hcrawimub.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include "nav_msgs/msg/path.hpp"
#include <novatel_oem7_msgs/msg/inspvax.hpp>
#include <novatel_oem7_msgs/msg/bestgnsspos.hpp>
#include <kygi665_msgs/msg/inspvax.hpp>
#include <nav2_interface/msg/location.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl/io/pcd_io.h>
#include <common_interface/msg/geometry_pose.hpp>
#include <deque>
#include <chrono>
#include <mutex>
#include <GeographicLib/LocalCartesian.hpp>
#include "common/ins.h" 
#include "lightning/srv/loc_cmd.hpp"
#include "core/localization/localization_result.h"
#include "core/system/health_reporter.h"
namespace lightning {
namespace loc {
class Localization;
}
class LocSystem {
   public:
    struct Options {
        bool pub_tf_ = true;  // 是否发布tf
        bool with_ins_ = false; // 是否使用INS数据
        bool pub_odom_ = true;
        bool ins_only_mode_ = false;  // INS直出模式: 不运行LIO和激光定位, 直接用INS位姿发布odom
    };

    explicit LocSystem(Options options);
    ~LocSystem();

    /// 初始化，地图路径在yaml里配置
    /// auto_set_init_pose: 是否在 Init 内部自动设置初始位姿（重定位场景应传 false）
    bool Init(const std::string& yaml_path, bool publish_occ_grid = true,
              bool publish_map_cloud = true, bool auto_set_init_pose = true);

    /// 设置初始化位姿
    /// @param skip_ndt: true 时跳过 NDT 配准，直接信任外部给的位姿（如 INS RTK_FIXED）
    void SetInitPose(const SE3& pose, bool skip_ndt = false);

    /// 处理IMU
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin（阻塞）
    void Spin();
    /// 非阻塞spin，处理待处理事件
    void SpinSome();

    /// 取消 Spin（用于退出阻塞的 spin）
    void CancelSpin();

    /// 创建 wall timer（回调在 executor 线程中执行）
    rclcpp::TimerBase::SharedPtr CreateWallTimer(
        std::chrono::nanoseconds period, std::function<void()> callback);

    /// 切换地图并重新初始化位姿
    bool SwitchMapAndReinit(const std::string& map_id, const SE3& init_pose);

    /// 阻止 INS 回调自动设置初始位姿（重定位前调用，防止抢占）
    void LockInitPose();

    /// 检查初始化超时，若超时则用 YAML 位姿启动定位
    void CheckInitTimeout();

    /// 查询定位是否正常（处于 GOOD 状态）
    bool IsLocalized() const;

    /// 获取最近一次定位状态（对应 loc::LocalizationStatus 枚举值）
    int GetLocStatus() const;

    /// 获取最近一次定位的 NDT 匹配置信度
    double GetConfidence() const;

    /// NDT位置初始化是否完成
    bool IsNDTInited() const;

    /// 获取当前加载的地图 UUID
    const std::string& GetCurrentMapUuid() const { return current_map_uuid_; }

    /// 是否使用了内置默认地图
    bool IsUsingDefaultMap() const { return using_default_map_; }

   private:
    Options options_;

    std::shared_ptr<loc::Localization> loc_ = nullptr;  // 定位接口

    std::atomic_bool loc_started_ = false;  // 是否开启定位
    std::atomic_bool map_loaded_ = false;   // 地图是否已载入
    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;

    // 多线程执行器，使 IMU、LiDAR、INS 回调组可并行处理
    rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_;
    rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
    rclcpp::CallbackGroup::SharedPtr lidar_cb_group_;
    rclcpp::CallbackGroup::SharedPtr ins_cb_group_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_= nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_world_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_lio_path_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_ins_path_ = nullptr;
    rclcpp::Publisher<nav2_interface::msg::Location>::SharedPtr pub_location_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_pub_ = nullptr;

    // 轨迹保存
    double last_save_time_ = 0.0;  // 上次保存时间
    bool traj_file_inited_ = false;  // 轨迹文件是否已初始化（首次覆盖）
    rclcpp::TimerBase::SharedPtr location_timer_ = nullptr;

    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;
    double time_offset=0.0;
    //rclcpp::Subscription<msg_interfaces::msg::Hcrawimub>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;
    rclcpp::Subscription<novatel_oem7_msgs::msg::INSPVAX>::SharedPtr sub_ins_ = nullptr;
    rclcpp::Subscription<novatel_oem7_msgs::msg::BESTGNSSPOS>::SharedPtr sub_bestgnsspos_ = nullptr;
    rclcpp::Subscription<kygi665_msgs::msg::INSPVAX>::SharedPtr sub_kygi665_ins_ = nullptr;
    // std::deque<INSPtr> ins_buffer_;
    GeographicLib::LocalCartesian geo_converter_;
    std::string ins_topic_;
    std::string ins_type_;  // "novatel" 或 "kygi665"
    std::string kygi665_ins_topic_;
    // system.ins_yaw_offset [deg]：kygi665 的 INS 朝向 = -azimuth + 这个。没写就是 0（狗上天线 1016 起接正了）；
    // 0921 那批包录的时候双天线接反，回放要 180。老版本写死 +180，2026-11 的新版改成了 0（T48 复核）
    double ins_yaw_offset_deg_ = 0.0;

    /// 定位命令服务（包括地图切换）
    rclcpp::Service<lightning::srv::LocCmd>::SharedPtr loc_cmd_service_ = nullptr;
    // 传感器频率监测
    double first_imu_time_ = 0.0;
    double last_imu_time_ = 0.0;
    double first_imu_sys_time_ = 0.0;
    double last_imu_sys_time_ = 0.0;
    size_t imu_cnt_ = 0;
    size_t imu_win_cnt_ = 0;           // 上次帧率评估窗口的帧计数
    double imu_freq_check_time_ = 0.0;  // 上次IMU帧率评估的系统时间
    double first_lidar_time_ = 0.0;
    double last_lidar_time_ = 0.0;
    double first_lidar_sys_time_ = 0.0;
    double last_lidar_sys_time_ = 0.0;
    size_t lidar_cnt_ = 0;
    int lidar_jitter_cnt_ = 0;          // 连续帧间隔超限的帧数
    double lidar_cloud_baseline_ = 0.0; // 点云数目基线(EMA)
    int lidar_cloud_drop_cnt_ = 0;      // 连续点云骤降的帧数
    double first_ins_time_ = 0.0;
    double last_ins_time_ = 0.0;
    double first_ins_sys_time_ = 0.0;
    double last_ins_sys_time_ = 0.0;
    size_t ins_cnt_ = 0;

    /// 健康上报(告警/事件)
    std::shared_ptr<HealthReporter> health_ = nullptr;
    bool was_localized_ = false;        // 上一周期定位是否正常
    bool lost_after_localized_ = false; // 本次异常是"定位后丢失"还是"从未匹配成功"
    bool loc_lost_reported_ = false;    // 本轮定位丢失是否已上报
    bool reloc_fail_reported_ = false;  // 本轮定位丢失是否已上报重定位失败
    bool map_match_fail_reported_ = false;  // 本轮是否已上报地图匹配失败
    double loc_lost_since_ = 0.0;       // 定位异常开始的系统时间
    static constexpr double LOC_LOST_DEBOUNCE_ = 1.0;    // 丢失去抖时长(秒)
    static constexpr double RELOC_FAIL_TIMEOUT_ = 10.0;  // 丢失持续该时长上报重定位失败(秒)
    static constexpr double MAP_MATCH_FAIL_TIMEOUT_ = 30.0;  // 启动后该时长未定位成功上报地图匹配失败(秒)

    /// LIO 路径轨迹
    nav_msgs::msg::Path lio_path_;
    /// INS 路径轨迹
    nav_msgs::msg::Path ins_path_;

    /// 初始化位姿相关
    SE3 yaml_init_pose_;                                      // 从YAML读取的初始位姿
    std::chrono::steady_clock::time_point init_start_time_;   // 初始化开始时间
    bool init_pose_set_ = false;                              // 是否已设置初始位姿
    std::recursive_mutex init_mutex_;                         // 保护初始化状态（可重入）

    // 回调到达间隔检测（steady_clock，不受时钟基准影响）
    double last_imu_recv_time_ = 0.0;
    double last_lidar_recv_time_ = 0.0;
    double last_ins_recv_time_ = 0.0;

    /// 传感器断流检测: 由 location_timer_(1Hz) 在 PublishLocationTimer 中调用,
    /// IMU/点云完全断流时兜底触发告警(不依赖数据回调)
    double watchdog_start_sec_ = 0.0;  // 启动时刻(用于启动宽限期)
    static constexpr double SENSOR_LOST_TIMEOUT_ = 3.0;   // 断流判定时长(秒)
    static constexpr double SENSOR_STARTUP_GRACE_ = 20.0; // 启动宽限期(秒)
    void SensorWatchdogTimer();

    /// 监测函数
    void MonitorIMUFrameRate(double sensor_time, double sys_time);
    void MonitorLidarFrameRate(double sensor_time, double sys_time);
    void MonitorLidarCloudSize(size_t cloud_size);
    void MonitorINSFrameRate(double sensor_time, double sys_time);

    /// 创建 INS 订阅(根据 ins_type_ 选择 novatel 或 kygi665)
    void CreateINSSubscription(const rclcpp::QoS& qos, const rclcpp::SubscriptionOptions& ins_sub_opts);
    /// 创建 INS only 模式的订阅(根据 ins_type_ 选择 novatel 或 kygi665)
    void CreateINSOnlySubscriptions(const rclcpp::QoS& qos);

    /// 定位命令服务回调
    void HandleLocCmd(const lightning::srv::LocCmd::Request::SharedPtr request, 
                      lightning::srv::LocCmd::Response::SharedPtr response);
    /// 定时发布定位消息
    void PublishLocationTimer();
    /// 从地图目录加载栅格地图并发布 OccupancyGrid
    void PublishOccupancyGrid(const std::string& map_path);
    /// 存储最新的导航状态
    NavState latest_nav_state_;
    nav_msgs::msg::Odometry latest_odom_;
    mutable std::mutex nav_state_mutex_;
    bool latest_nav_state_valid_ = false;
    /// 存储最新的INS地理坐标

    novatel_oem7_msgs::msg::INSPVAX latest_inspvax_;
    sensor_msgs::msg::Imu     latest_imu_;
    std::mutex lla_mutex_;
    bool latest_lla_valid_ = false;
    int8_t latest_ins_status_ = -1; // 对应GeograpicPose状态
        /// 最新 INS 位姿（ENU），重定位时优先使用
    SE3 latest_ins_pose_;
    bool ins_pose_valid_ = false;

    /// 最新 INS 地理坐标(通用, 不依赖具体消息类型)
    struct INSGeoData {
        double latitude = 0;
        double longitude = 0;
        double height = 0;
        uint32_t pos_type = 0;  // 位置类型(novatel: pos_type.type, kygi665: pos_type.data)
    };
    INSGeoData latest_ins_geo_;

    /// 最新 IMU 角速度(body系), 用于 INS only 模式输出 twist
    Vec3d latest_imu_gyro_ = Vec3d::Zero();
    double latest_imu_stamp_ = 0.0;
    std::mutex imu_gyro_mutex_;

    /// IMU 到 body 的外参(从 yaml 读取)
    SE3 T_imu_body_ = SE3();

    /// INS 固定解有效性状态机(INS only 模式)
    bool ins_fixed_once_ = false;           // 是否曾收到过固定解
    double last_fixed_time_ = 0.0;          // 上次固定解时间戳
    double non_fixed_start_time_ = 0.0;     // 当前连续非固定解的起始时间, 0 表示无连续非固定
    static constexpr double INS_LOSS_TIMEOUT_ = 3.0;  // 连续非固定超时阈值(秒)

    /// 当前加载的地图 UUID（map_path 的最后一级目录名）
    std::string current_map_uuid_;

    /// 是否使用了内置默认地图（配置地图无效时回退），此时定位结果视为无效
    bool using_default_map_ = false;
};

};  // namespace lightning

#endif  // LIGHTNING_LOC_SYSTEM_H
