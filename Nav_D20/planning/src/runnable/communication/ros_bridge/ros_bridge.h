#pragma once

#include "src/common/util/mean_filter.h"
#include "src/runnable/communication/communication_base.h"
#include <functional>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mc_interface/msg/mode.hpp>
#include <mc_interface/msg/motion.hpp>
#include <mc_interface/msg/motion_skill.hpp>
#include <mc_interface/msg/motion_state.hpp>
#include <mc_interface/msg/state.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav2_interface/action/back_charge.hpp>
#include <nav2_interface/action/follow_object.hpp>
#include <nav2_interface/action/navigation.hpp>
#include <nav2_interface/msg/nav_state.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <perception_interface/msg/objects.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rte_interface/msg/battery_states.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
#include <visualization_msgs/msg/marker_array.hpp>
#include <woosh_robot_msgs/srv/twist.hpp>
namespace planning {
namespace communication {
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

  // Navigation action server
  using NavigationAction = nav2_interface::action::Navigation;
  using GoalHandleNavigation =
      rclcpp_action::ServerGoalHandle<NavigationAction>;
  rclcpp_action::GoalResponse
  handle_navigation_goal(const rclcpp_action::GoalUUID &uuid,
                         std::shared_ptr<const NavigationAction::Goal> goal);
  rclcpp_action::CancelResponse handle_navigation_cancel(
      const std::shared_ptr<GoalHandleNavigation> goal_handle);
  void handle_navigation_accepted(
      const std::shared_ptr<GoalHandleNavigation> goal_handle);
  void TrackingFeedback();
  void TrackingInitPipe();

  // Follow action server
  using FollowAction = nav2_interface::action::FollowObject;
  using GoalHandleFollow = rclcpp_action::ServerGoalHandle<FollowAction>;
  rclcpp_action::GoalResponse
  follow_handle_goal(const rclcpp_action::GoalUUID &uuid,
                     std::shared_ptr<const FollowAction::Goal> goal);
  rclcpp_action::CancelResponse
  follow_handle_cancel(const std::shared_ptr<GoalHandleFollow> goal_handle);
  void
  follow_handle_accepted(const std::shared_ptr<GoalHandleFollow> goal_handle);
  void FollowingFeedback();
  void FollowingInitPipe();
  void
  objects_callback(const perception_interface::msg::Objects::SharedPtr msg);

  // Recharging action server
  using BackChargeAction = nav2_interface::action::BackCharge;
  using GoalHandleBackCharge =
      rclcpp_action::ServerGoalHandle<BackChargeAction>;
  rclcpp_action::GoalResponse
  back_charge_handle_goal(const rclcpp_action::GoalUUID &uuid,
                          std::shared_ptr<const BackChargeAction::Goal> goal);
  rclcpp_action::CancelResponse back_charge_handle_cancel(
      const std::shared_ptr<GoalHandleBackCharge> goal_handle);
  void back_charge_handle_accepted(
      const std::shared_ptr<GoalHandleBackCharge> goal_handle);
  void RechargingFeedback();
  void RechargingInitPipe();
  void charging_goal_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void battery_states_callback(
      const rte_interface::msg::BatteryStates::SharedPtr msg);
  void motion_state_callback(const mc_interface::msg::State::SharedPtr msg);
  void map_switch_callback(const std_msgs::msg::String::SharedPtr msg);
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
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      sub_charging_pose_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_global_occ_;
  rclcpp::Subscription<rte_interface::msg::BatteryStates>::SharedPtr
      sub_battery_states_;
  rclcpp::Subscription<mc_interface::msg::State>::SharedPtr sub_motion_state_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_map_switch_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
  rclcpp::Publisher<mc_interface::msg::Motion>::SharedPtr pub_motion_;

  // Navigation action server
  rclcpp_action::Server<NavigationAction>::SharedPtr navigation_server_;
  std::shared_ptr<GoalHandleNavigation> current_navigation_handle_;

  // Follow action server
  rclcpp_action::Server<FollowAction>::SharedPtr follow_server_;
  std::shared_ptr<GoalHandleFollow> current_follow_handle_;
  rclcpp::Subscription<perception_interface::msg::Objects>::SharedPtr
      sub_objects_;

  // BackCharge action server
  rclcpp_action::Server<BackChargeAction>::SharedPtr back_charge_server_;
  std::shared_ptr<GoalHandleBackCharge> current_back_charge_handle_;
  uint32_t last_back_charge_feedback_state_{0xFFFFFFFFu};
  uint32_t back_charge_final_state_{0};

  // for r11-w
  rclcpp::Client<woosh_robot_msgs::srv::Twist>::SharedPtr woosh_twist_client_;

  struct TimeoutState {
    double last_time = 0.0;
    bool timed_out = true;
    double threshold = 0.0;
    std::function<bool()> is_active;
  };
  std::unordered_map<std::string, TimeoutState> timeout_states_;

  Odometry odometry_;
  grid_map::GridMap grid_map_;
  double timeout_pause_start_time_ = 0.0;
  double timeout_resume_start_time_ = 0.0;
  bool is_timeout_paused_ = false;

  // debug
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      init_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_pose_sub_;
};
} // namespace communication
} // namespace planning
