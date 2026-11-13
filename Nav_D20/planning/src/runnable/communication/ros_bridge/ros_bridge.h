#pragma once

#include "grid_map_ros/GridMapRosConverter.hpp"
#include "src/common/util/debug_datas.h"
#include "src/common/util/math.h"
#include "src/common/util/mean_filter.h"
#include "src/common/util/tf_buffer.h"
#include "src/common/util/velocity_calibration.h"
#include "src/runnable/communication/communication_base.h"
#include "src/runnable/communication/ros_bridge/diagnostic_bridge.h"
#include "src/runnable/communication/ros_bridge/following_bridge.h"
#include "src/runnable/communication/ros_bridge/recharging_bridge.h"
#include "src/runnable/communication/ros_bridge/skill_bridge.h"
#include "src/runnable/communication/ros_bridge/timeout_state.h"
#include "src/runnable/communication/ros_bridge/tracking_bridge.h"
#include <common_interface/msg/path_file.hpp>
#include <functional>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mc_interface/msg/mode.hpp>
#include <mc_interface/msg/motion.hpp>
#include <mc_interface/msg/motion_skill.hpp>
#include <mc_interface/msg/motion_state.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <perception_interface/msg/lidar_object_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
#include <visualization_msgs/msg/marker_array.hpp>

namespace planning {
namespace communication {
using planning::common::util::DebugDatas;
using planning::common::util::normalize_angle;
using planning::common::util::TfBuffer;
using planning::common::util::VelocityCalibration;

using std::placeholders::_1;
using std::placeholders::_2;
class RosBridge : public CommunicationBase {
public:
  RosBridge(const CommunicationConf::Conf &conf);
  ~RosBridge();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  bool Publish() override;
  bool ok() override;
  const Odometry *odometry() override { return &odometry_; }
  const grid_map::GridMap *occupied_map_2d() override { return &grid_map_; }
  bool lidar_objects(double *stamp, uint64_t *seq, std::vector<algorithm::yield::Object> *objects) override;
  bool checkTimeout();

private:
  // subscriber
  void sync_callback(const nav_msgs::msg::Odometry::SharedPtr &odom_msg,
                     const nav_msgs::msg::OccupancyGrid::SharedPtr &occ_msg);
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void occupancy_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void local_octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg);
  void
  global_occupancy_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  void map_switch_callback(const std_msgs::msg::String::SharedPtr msg);
  void lidar_objects_callback(const perception_interface::msg::LidarObjectArray::SharedPtr msg);
  void
  map_update_callback(const common_interface::msg::PathFile::SharedPtr msg);
  // debug
  void OnGoalPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void OnInitPose(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

private:
  CommunicationConf::RosBridgeConf config_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Rate rate_;

  using OdomSub = message_filters::Subscriber<nav_msgs::msg::Odometry>;
  using OccSub = message_filters::Subscriber<nav_msgs::msg::OccupancyGrid>;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      nav_msgs::msg::Odometry, nav_msgs::msg::OccupancyGrid>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
  std::unique_ptr<OdomSub> odom_sub_;
  std::unique_ptr<OccSub> occ_sub_;

  std::unique_ptr<Synchronizer> sync_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      sub_joint_state_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr
      sub_local_octomap_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_global_occ_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_map_switch_;
  rclcpp::Subscription<perception_interface::msg::LidarObjectArray>::SharedPtr sub_lidar_objects_;
  std::mutex lidar_objects_mutex_;
  double lidar_objects_stamp_ = -1.0;
  uint64_t lidar_objects_seq_ = 0;
  std::vector<algorithm::yield::Object> lidar_objects_;
  rclcpp::Subscription<common_interface::msg::PathFile>::SharedPtr
      sub_map_update_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
  rclcpp::Publisher<mc_interface::msg::Motion>::SharedPtr pub_motion_;

  TimeoutStates timeout_states_;

  // Mission-specific action server bridges (constructed in Init(), their
  // Publish() is called every cycle from RosBridge::Publish()).
  std::unique_ptr<TrackingBridge> tracking_bridge_;
  std::unique_ptr<RechargingBridge> recharging_bridge_;
  std::unique_ptr<FollowingBridge> following_bridge_;
  std::unique_ptr<SkillBridge> skill_bridge_;

  Odometry odometry_;
  grid_map::GridMap grid_map_;
  double timeout_pause_start_time_ = 0.0;
  double timeout_resume_start_time_ = 0.0;
  bool is_timeout_paused_ = false;

  // Fault edge event publisher (AlarmInfo/EventInfo to rhr)
  std::unique_ptr<DiagnosticBridge> diagnostic_bridge_;

  // debug
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      init_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_pose_sub_;
};
} // namespace communication
} // namespace planning
