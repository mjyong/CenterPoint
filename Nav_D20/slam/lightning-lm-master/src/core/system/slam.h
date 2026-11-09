//
// Created by xiang on 25-5-6.
//

#ifndef LIGHTNING_SLAM_H
#define LIGHTNING_SLAM_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <string>
#include <deque>
#include <chrono>
#include "lightning/srv/save_map.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/ins.h"
#include "common/keyframe.h"
#include "novatel_oem7_msgs/msg/inspvax.hpp"
#include "kygi665_msgs/msg/inspvax.hpp"
#include <GeographicLib/LocalCartesian.hpp>
// #include "msg_interfaces/msg/hcrawimub.hpp"
#include "core/g2p5/g2p5_map.h"
namespace lightning {

class LaserMapping;  //  lio 前端
class LaserMappingSuperLIO;    // super_lio 前端
class LoopClosing;   // 回环检测

namespace ui {
class RvizVisualizer;
}

namespace g2p5 {
class G2P5;
}

/**
 * SLAM 系统调用接口
 */
class SlamSystem {
   public:
    struct Options {
        Options() {}

        bool online_mode_ = true;  // 在线模式，在线模式下会起一些子线程来做异步处理

        bool with_cc_ = true;               // 是否需要带交叉验证
        bool with_gridmap_ = true;          // 是否需要2D栅格
        bool with_loop_closing_ = true;     // 是否需要回环检测
        bool with_visualization_ = true;    // 是否需要可视化UI
        bool with_2dvisualization_ = true;  // 是否需要2D可视化UI

        bool step_on_kf_ = true;  // 是否在关键帧处暂停p
        bool with_ins_ = true;     // 是否使用INS数据
        bool use_super_lio_ = true;
    };

    using SaveMapService = srv::SaveMap;

    SlamSystem(Options options);
    ~SlamSystem();

    /// 初始化
    bool Init(const std::string& yaml_path);

    /// 对外部交互接口
    /// 开始建图，输入地图名称
    void StartSLAM(std::string map_name);

    /// 保存地图，默认保存至./data/地图名/ 下方
    void SaveMap(const std::string& path = "");
    /// 按模板格式保存地图（global_map.yaml + session + 2d + 3d）
    void SaveMapByTemplate(const std::string& path = "");   
    /// 获取最后一帧关键帧的优化位姿（建图结束后用于初始化定位）
    SE3 GetLastKeyframePose();
    /// 处理IMU
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();
    void SpinSome();
    void CancelSpin();

    /// 地图监控回调，参数为uint8数组、宽度、高度、栅格分辨率（米/像素）、地图原点坐标x、y（米）
    using MapDataCallback = std::function<void(const std::vector<uint8_t>&, int width, int height, float resolution, float origin_x, float origin_y)>;

    /// 启动地图监控线程，当位移超过translation_threshold_m米或旋转超过rotation_threshold_deg度时触发回调
    void StartMapMonitor(float translation_threshold_m = 3.0, float rotation_threshold_deg = 30.0, MapDataCallback callback = nullptr);

    /// 停止地图监控
    void StopMapMonitor();

   private:
    /// ros端保存地图的实现
    void SaveMap(const SaveMapService::Request::SharedPtr request, SaveMapService::Response::SharedPtr response);
    /// 在ins_buffer_中查找时间戳最接近给定时间戳的INS数据
    INSPtr FindNearestINS(double stamp);
    /// 移除时间戳早于给定时间戳的旧INS数据
    void RemoveOldINS(double stamp);
    Options options_;
    std::atomic_bool running_ = false;

    rclcpp::Service<SaveMapService>::SharedPtr savemap_service_ = nullptr;

    std::string map_name_;  // 地图名

    std::shared_ptr<LaserMapping> lio_ = nullptr;       // lio 前端
    std::shared_ptr<LaserMappingSuperLIO> super_lio_ = nullptr;  // super_lio 前端
    std::shared_ptr<LoopClosing> lc_ = nullptr;         // 回环检测
    std::shared_ptr<ui::RvizVisualizer> ui_ = nullptr;  // ui
    std::shared_ptr<g2p5::G2P5> g2p5_ = nullptr;        // 栅格地图

    Keyframe::Ptr cur_kf_ = nullptr;

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::deque<INSPtr> ins_buffer_;
    std::mutex ins_buffer_mutex_;  // 保护 ins_buffer_ 的并发访问
    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;
    std::string ins_topic_;
    std::string ins_type_ = "novatel";  // "novatel" 或 "kygi665"
    std::string kygi665_ins_topic_;     // kygi665 INS 话题
    double time_offset=0.0;
        /// 激光雷达帧率监测（丢帧检测使用传感器时间，帧率统计使用系统时间）
    double first_lidar_time_ = 0.0;
    double last_lidar_time_ = 0.0;
    double first_lidar_sys_time_ = 0.0;
    double last_lidar_sys_time_ = 0.0;
    int frame_count_ = 0;
    double frame_rate_ = 0.0;
    double expected_interval_ = 0.1;  // 默认10Hz
    int dropped_frames_ = 0;
    double max_interval_ = 0.0;
    double min_interval_ = 1e6;
    /// 检测丢帧和帧率稳定性的函数
    void MonitorLidarFrameRate(double sensor_time, double sys_time);

    /// IMU 数据频率监测
    double first_imu_time_ = 0.0;
    double last_imu_time_ = 0.0;
    double first_imu_sys_time_ = 0.0;
    double last_imu_sys_time_ = 0.0;
    int imu_count_ = 0;
    double imu_rate_ = 0.0;
    double imu_expected_interval_ = 0.01;  // 默认200Hz
    int imu_dropped_ = 0;
    double imu_max_interval_ = 0.0;
    double imu_min_interval_ = 1e6;
    void MonitorIMUFrameRate(double sensor_time, double sys_time);

    /// INS 数据频率监测
    double first_ins_time_ = 0.0;
    double last_ins_time_ = 0.0;
    double first_ins_sys_time_ = 0.0;
    double last_ins_sys_time_ = 0.0;
    int ins_count_ = 0;
    double ins_rate_ = 0.0;
    double ins_expected_interval_ = 0.1;  // 默认10Hz (Novatel INSPVAX)
    int ins_dropped_ = 0;
    double ins_max_interval_ = 0.0;
    double ins_min_interval_ = 1e6;
    void MonitorINSFrameRate(double sensor_time, double sys_time);

    /// INS 到 LIO 坐标系的变换（第一帧）
    SE3 T_ins_lio_ = SE3();
    bool T_ins_lio_computed_ = false;
    /// INS 是否已初始化（有效INS且外参已计算）
    bool ins_initialized_ = false;
    /// 用于SVD计算外参的位姿对（收集INS和LIO位姿）
    std::vector<SE3> ins_poses_for_calib_;
    std::vector<SE3> lio_poses_for_calib_;
    /// 是否已通过SVD计算外参
    bool svd_calibrated_ = false;
    GeographicLib::LocalCartesian geo_converter_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<novatel_oem7_msgs::msg::INSPVAX>::SharedPtr sub_ins_ = nullptr;
    // kygi665 组合导航订阅器 (仅 INS 数据源, IMU 统一使用 sensor_msgs/Imu)
    rclcpp::Subscription<kygi665_msgs::msg::INSPVAX>::SharedPtr sub_kygi665_ins_ = nullptr;
    // rclcpp::Subscription<msg_interfaces::msg::Hcrawimub>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;

    rclcpp::CallbackGroup::SharedPtr imu_cb_group_ = nullptr;
    rclcpp::CallbackGroup::SharedPtr lidar_cb_group_ = nullptr;
    rclcpp::CallbackGroup::SharedPtr ins_cb_group_ = nullptr;

    /// 多线程执行器，使 IMU 与 LiDAR 回调可并行处理
    rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_ = nullptr;

    /// INS 可视化发布器
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ins_pose_pub_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ins_path_pub_ = nullptr;
    nav_msgs::msg::Path ins_path_;

    /// 地图监控相关
    std::mutex monitor_mutex_;
    bool monitor_enabled_ = false;
    float trans_threshold_ = 3.0;
    float rot_threshold_ = 30.0;
    MapDataCallback map_callback_;
    SE3 last_trigger_pose_;
    bool trigger_initialized_ = false;
    std::chrono::steady_clock::time_point last_trigger_time_;

    /// 内部方法：检查运动条件并触发回调
    void CheckMotionAndTrigger(const SE3& current_pose);
    /// 将G2P5地图转换为uint8数组
    std::vector<uint8_t> ConvertMapToUint8Array(const g2p5::G2P5MapPtr& map);
};
}  // namespace lightning

#endif  // LIGHTNING_SLAM_H
