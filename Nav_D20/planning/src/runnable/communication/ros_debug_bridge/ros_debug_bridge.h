/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-03-05 17:31:33
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-09-12 17:51:15
 */
#pragma once
#include "rclcpp/rclcpp.hpp"
#include "src/runnable/communication/communication_base.h"
#include "src/runnable/communication/ros_debug_bridge/visualization_util.h"
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <plotjuggler_msgs/msg/data_points.hpp>
#include <plotjuggler_msgs/msg/dictionary.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
namespace planning {
namespace communication {
class RosDebugBridge : public CommunicationBase {

public:
  RosDebugBridge(const CommunicationConf::Conf &conf);
  ~RosDebugBridge();

  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  bool Publish() override;

private:
  void OnGoal(const geometry_msgs::msg::PointStamped::SharedPtr msg);
  void PathTrackingDebug();
  void GoalPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void TEBDebug();
  void
  SetPubFlag(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
             std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  Eigen::Vector3d transformToGlobal(const Eigen::Vector3d &local_point,
                                    const Eigen::Vector3d &robot_pos,
                                    double robot_yaw);

private:
  PlanningEntry *planning_entry_{nullptr};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_pose_sub_;
  rclcpp::Publisher<plotjuggler_msgs::msg::Dictionary>::SharedPtr
      pub_plotjuggler_dictionary_;
  rclcpp::Publisher<plotjuggler_msgs::msg::DataPoints>::SharedPtr
      pub_plotjuggler_data_points_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_debug_string_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      path_tracking_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      pub_diagnostic_msg_;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      collision_checker_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_grid_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr local_occmap_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_occmap_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      frenet_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      gradient_vec_pub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_pub_flag_srv_;
  bool pub_flag_ = true;
  size_t last_global_occmap_sub_count_{0};

  visualization_msgs::msg::Marker goal_pose_marker_;
};

} // namespace communication
} // namespace planning
