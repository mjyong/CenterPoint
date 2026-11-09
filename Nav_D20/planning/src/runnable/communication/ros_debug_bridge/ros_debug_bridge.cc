/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2024-03-08 15:57:06
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2025-09-06 09:59:12
 */
#include "src/runnable/communication/ros_debug_bridge/ros_debug_bridge.h"
#include "src/common/util/debug_datas.h"
#include "src/common/util/diagnostic.h"
#include "src/common/util/rosbag_recorder.h"
#include "src/common/util/tf_buffer.h"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/tracking/tracking.h"
#include <grid_map_ros/GridMapRosConverter.hpp>

using planning::common::util::DebugDatas;
using planning::common::util::Diagnostic;
using planning::common::util::RosbagRecorder;
using planning::common::util::TfBuffer;

namespace planning {
namespace communication {

RosDebugBridge::RosDebugBridge(const CommunicationConf::Conf &conf) {
  conf_ = conf;
}

RosDebugBridge::~RosDebugBridge() {}

bool RosDebugBridge::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  node_ = std::make_shared<rclcpp::Node>("planning_debug_bridge");
  goal_pose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 1,
      std::bind(&RosDebugBridge::GoalPose, this, std::placeholders::_1));
  pub_plotjuggler_dictionary_ =
      node_->create_publisher<plotjuggler_msgs::msg::Dictionary>(
          "/planning/debug/plotjuggler/dictionary", 1);
  pub_plotjuggler_data_points_ =
      node_->create_publisher<plotjuggler_msgs::msg::DataPoints>(
          "/planning/debug/plotjuggler/data_points", 1);
  DebugDatas::Instance();
  pub_debug_string_ = node_->create_publisher<std_msgs::msg::String>(
      "/planning/debug/plotjuggler/string", 1);
  pub_diagnostic_msg_ =
      node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
          "/planning/debug/diagnostic", 1);

  path_tracking_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planning/debug/path_tracking", 1);

  goal_pose_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
      "/planning/debug/goal_pose", 1);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

  collision_checker_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planning/debug/collision_checker", 10);
  local_grid_map_pub_ = node_->create_publisher<grid_map_msgs::msg::GridMap>(
      "/planning/debug/cost_map_production/grid_map", 1);
  local_occmap_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/planning/debug/cost_map_production/occmap", 1);
  global_occmap_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/planning/debug/global_map_manager/occmap", 1);
  frenet_debug_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planning/debug/frenet_boundary", 1);
  gradient_vec_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
      "/planning/debug/gradient_vector", 1);

  pub_flag_ = conf_.ros_debug_bridge_conf().pub_flag();
  set_pub_flag_srv_ = node_->create_service<std_srvs::srv::SetBool>(
      "/planning/debug/set_pub_flag",
      std::bind(&RosDebugBridge::SetPubFlag, this, std::placeholders::_1,
                std::placeholders::_2));
  return true;
}

bool RosDebugBridge::Proc() { return true; }

bool RosDebugBridge::Publish() {
  rclcpp::spin_some(node_);
  std_msgs::msg::String debug_string;
  debug_string.data = DebugDatas::ConvertToString("planning");
  pub_debug_string_->publish(debug_string);
  RosbagRecorder::Instance()->add(debug_string,
                                  "/planning/debug/plotjuggler/string",
                                  "std_msgs/msg/String");

  if (!pub_flag_) {
    DebugDatas::Instance()->clear();
    return true;
  }
  plotjuggler_msgs::msg::Dictionary dictionary;
  plotjuggler_msgs::msg::DataPoints data_points;
  DebugDatas::ConvertToRosMsg(dictionary, data_points);
  pub_plotjuggler_dictionary_->publish(dictionary);
  pub_plotjuggler_data_points_->publish(data_points);
  RosbagRecorder::Instance()->add(dictionary,
                                  "/planning/debug/plotjuggler/dictionary",
                                  "plotjuggler_msgs/msg/Dictionary");
  RosbagRecorder::Instance()->add(data_points,
                                  "/planning/debug/plotjuggler/data_points",
                                  "plotjuggler_msgs/msg/DataPoints");

  DebugDatas::Instance()->clear();
  diagnostic_msgs::msg::DiagnosticArray diagnostic_msg;
  Diagnostic::Instance()->ConvertToRosMsg(diagnostic_msg);
  pub_diagnostic_msg_->publish(diagnostic_msg);

  nav_msgs::msg::OccupancyGrid occmap_msg;
  grid_map::GridMapRosConverter::toOccupancyGrid(
      *planning_entry_->preprocess_entry_ptr()
           ->cost_map_production()
           ->GridMap(),
      "obstacle", 0.0, 1.0, occmap_msg);
  RosbagRecorder::Instance()->add(occmap_msg,
                                  "/planning/debug/cost_map_production/occmap",
                                  "nav_msgs/msg/OccupancyGrid");
  local_occmap_pub_->publish(std::move(occmap_msg));

  auto grid_map_msg = grid_map::GridMapRosConverter::toMessage(
      *planning_entry_->preprocess_entry_ptr()
           ->cost_map_production()
           ->GridMap());
  // RosbagRecorder::Instance()->add(grid_map_msg,
  //                                 "/planning/debug/cost_map_production/grid_map",
  //                                 "grid_map_msgs/msg/GridMap");
  local_grid_map_pub_->publish(*grid_map_msg);
  // pub obstacle gradient vector
  {
    auto *grid_map = planning_entry_->preprocess_entry_ptr()
                         ->cost_map_production()
                         ->GridMap();
    visualization_msgs::msg::Marker gradient_vec_marker;
    gradient_vec_marker.header.frame_id = "base_link";
    gradient_vec_marker.header.stamp = node_->now();
    gradient_vec_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    gradient_vec_marker.action = visualization_msgs::msg::Marker::ADD;
    gradient_vec_marker.ns = "composite_gradient";
    gradient_vec_marker.id = 0;
    gradient_vec_marker.scale.x = 0.01;
    gradient_vec_marker.color.r = 1.0;
    gradient_vec_marker.color.g = 0.0;
    gradient_vec_marker.color.b = 0.0;
    gradient_vec_marker.color.a = 0.6;
    gradient_vec_marker.points.clear();
    constexpr double SCALE = 0.05;
    for (grid_map::GridMapIterator iterator(*grid_map); !iterator.isPastEnd();
         ++iterator) {
      Eigen::Vector3d position;
      grid_map->getPosition3("composite_cost", *iterator, position);
      geometry_msgs::msg::Point start_point;
      start_point.x = position.x();
      start_point.y = position.y();
      start_point.z = position.z();
      gradient_vec_marker.points.push_back(start_point);
      double grad_x = grid_map->at("composite_gradient_x", *iterator);
      double grad_y = grid_map->at("composite_gradient_y", *iterator);
      geometry_msgs::msg::Point end_point;
      end_point.x = start_point.x + SCALE * grad_x;
      end_point.y = start_point.y + SCALE * grad_y;
      end_point.z = start_point.z;
      gradient_vec_marker.points.push_back(end_point);
    }
    gradient_vec_pub_->publish(gradient_vec_marker);
  }

  Diagnostic::Instance()->clear();
  const auto &tf_msg = TfBuffer::Instance()->toTfMessage();
  tf_broadcaster_->sendTransform(tf_msg);
  tf2_msgs::msg::TFMessage tf2_msg;
  tf2_msg.transforms = tf_msg;
  RosbagRecorder::Instance()->add(tf2_msg, "/tf", "tf2_msgs/msg/TFMessage");
  PathTrackingDebug();
  goal_pose_pub_->publish(goal_pose_marker_);
  TEBDebug();

  auto collision_checker =
      planning_entry_->preprocess_entry_ptr()->collision_checker();
  if (collision_checker) {
    visualization_msgs::msg::MarkerArray marker_array;
    for (size_t i = 0; i < collision_checker->contacts().size(); i++) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = collision_checker->contacts()[i].x();
      pose.position.y = collision_checker->contacts()[i].y();
      pose.position.z = collision_checker->contacts()[i].z();
      auto contact_marker = VisualizationUtil::createMarkerSphere(
          "base_link", "contact", i, pose,
          VisualizationUtil::createMarkerScale(0.1, 0.1, 0.1),
          VisualizationUtil::getColor(VisualizationUtil::YELLOW));
      marker_array.markers.emplace_back(contact_marker);
    }
    collision_checker_pub_->publish(marker_array);
  }

  if (conf_.ros_debug_bridge_conf().enable_global_map_pub()) {
    auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
    if (gmm) {
      const size_t cur_count = global_occmap_pub_->get_subscription_count();
      if (cur_count > last_global_occmap_sub_count_) {
        nav_msgs::msg::OccupancyGrid occmap_msg;
        grid_map::GridMapRosConverter::toOccupancyGrid(
            gmm->global_map(), "obstacle", 0.0, 1.0, occmap_msg);
        global_occmap_pub_->publish(std::move(occmap_msg));
      }
      last_global_occmap_sub_count_ = cur_count;
    }
  }
  return true;
}

void RosDebugBridge::SetPubFlag(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  pub_flag_ = request->data;
  response->success = true;
  response->message = pub_flag_ ? "pub_flag enabled" : "pub_flag disabled";
  AINFO << "SetPubFlag: " << response->message;
}

void RosDebugBridge::PathTrackingDebug() {
  const auto &mission_base_tracking =
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Tracking);
  auto mission_tracking =
      dynamic_cast<planning::mission::Tracking *>(mission_base_tracking);
  visualization_msgs::msg::MarkerArray marker_array;
  int32_t marker_id = 1;
  auto global_path_2d = mission_tracking->global_path_2d();
  auto global_path_marker = VisualizationUtil::createLineStripMarker(
      "map", "global_path_2d", marker_id++,
      VisualizationUtil::getColor(VisualizationUtil::BLUE), *global_path_2d);
  marker_array.markers.push_back(global_path_marker);

  // for (const auto &pt : *global_path_2d) {
  //   geometry_msgs::msg::Pose2D ros_pose2d;
  //   ros_pose2d.x = pt.position.x();
  //   ros_pose2d.y = pt.position.y();
  //   ros_pose2d.theta = pt.theta;
  //   auto footprint_marker = VisualizationUtil::createMarkerFootprint(
  //       "map", "global_path_footprint", marker_id++, ros_pose2d,
  //       VisualizationUtil::getColor(VisualizationUtil::BLUE));
  //   marker_array.markers.push_back(footprint_marker);
  // }
  auto local_path_2d = mission_tracking->local_path_2d();
  auto local_path_marker = VisualizationUtil::createLineStripMarker(
      "map", "local_path_2d", marker_id++,
      VisualizationUtil::GetColorByGait(Gait::WALK), *local_path_2d);

  marker_array.markers.push_back(local_path_marker);
  for (const auto &pt : *local_path_2d) {
    geometry_msgs::msg::Pose2D ros_pose2d;
    ros_pose2d.x = pt.position.x();
    ros_pose2d.y = pt.position.y();
    ros_pose2d.theta = pt.theta;
    auto footprint_marker = VisualizationUtil::createMarkerFootprint(
        "map", "local_path_footprint", marker_id++, ros_pose2d,
        VisualizationUtil::getColor(VisualizationUtil::GREEN));
    marker_array.markers.push_back(footprint_marker);
  }

  auto target_pose = mission_tracking->target_pose();
  geometry_msgs::msg::Pose target_pt_msg;
  target_pt_msg.position.x = target_pose.position.x();
  target_pt_msg.position.y = target_pose.position.y();
  target_pt_msg.position.z = target_pose.position.z();
  target_pt_msg.orientation.w = target_pose.orientation.w();
  target_pt_msg.orientation.x = target_pose.orientation.x();
  target_pt_msg.orientation.y = target_pose.orientation.y();
  target_pt_msg.orientation.z = target_pose.orientation.z();
  auto target_pt_marker = VisualizationUtil::createMarkerArrow(
      "map", "target_pt", marker_id++, target_pt_msg,
      VisualizationUtil::createMarkerScale(0.5, 0.1, 0.1),
      VisualizationUtil::getColor(mission_tracking->constraint_theta()
                                      ? VisualizationUtil::RED
                                      : VisualizationUtil::GREEN));
  marker_array.markers.push_back(target_pt_marker);

  auto odometry =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  geometry_msgs::msg::Pose2D base_pose2d;
  base_pose2d.x = odometry->pose.position.x();
  base_pose2d.y = odometry->pose.position.y();
  base_pose2d.theta = odometry->pose.euler()[2];
  auto footprint_marker = VisualizationUtil::createMarkerFootprint(
      "map", "robot_footprint", marker_id++, base_pose2d,
      VisualizationUtil::getColor(VisualizationUtil::CYAN));
  marker_array.markers.push_back(footprint_marker);

  VisualizationUtil::appendMarkerArray(
      VisualizationUtil::createMarkerFootprintCircle(
          "map", "robot_circle", marker_id++, base_pose2d,
          VisualizationUtil::getColor(VisualizationUtil::PINK)),
      &marker_array);

  geometry_msgs::msg::Pose robot_heading_pose;
  robot_heading_pose.position.x = odometry->pose.position.x();
  robot_heading_pose.position.y = odometry->pose.position.y();
  robot_heading_pose.position.z = odometry->pose.position.z();
  robot_heading_pose.orientation.w = odometry->pose.orientation.w();
  robot_heading_pose.orientation.x = odometry->pose.orientation.x();
  robot_heading_pose.orientation.y = odometry->pose.orientation.y();
  robot_heading_pose.orientation.z = odometry->pose.orientation.z();
  auto heading_arrow_marker = VisualizationUtil::createMarkerArrow(
      "map", "robot_heading", marker_id++, robot_heading_pose,
      VisualizationUtil::createMarkerScale(0.5, 0.1, 0.1),
      VisualizationUtil::getColor(VisualizationUtil::CYAN));
  marker_array.markers.push_back(heading_arrow_marker);

  RosbagRecorder::Instance()->add(marker_array, "/planning/debug/path_tracking",
                                  "visualization_msgs/msg/MarkerArray");
  path_tracking_pub_->publish(marker_array);
}

void RosDebugBridge::GoalPose(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  goal_pose_marker_ = VisualizationUtil::createMarkerArrow(
      "map", "goal_pose", 1, msg->pose,
      VisualizationUtil::createMarkerScale(0.5, 0.1, 0.1),
      VisualizationUtil::getColor(VisualizationUtil::GREEN));
}

void RosDebugBridge::TEBDebug() {
  const auto &mission_base_tracking =
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Tracking);
  auto mission_tracking =
      dynamic_cast<planning::mission::Tracking *>(mission_base_tracking);

  if (mission_tracking == nullptr ||
      mission_tracking->teb_planner() == nullptr) {
    return;
  }

  auto teb_planner = mission_tracking->teb_planner();
  visualization_msgs::msg::MarkerArray marker_array;
  int32_t marker_id = 1;

  auto odometry =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  Eigen::Vector3d robot_pos(odometry->pose.position.x(),
                            odometry->pose.position.y(),
                            odometry->pose.position.z());
  double robot_yaw = odometry->pose.euler()[2];
  auto coarse_path = teb_planner->getCoarsePath();
  if (!coarse_path.empty()) {
    std::vector<Eigen::Vector3d> coarse_path_global;
    for (const auto &pose : coarse_path) {
      coarse_path_global.emplace_back(
          transformToGlobal(pose, robot_pos, robot_yaw));
    }
    auto coarse_path_marker = VisualizationUtil::createLineStripMarker(
        "map", "frenet_coarse_path", marker_id++,
        VisualizationUtil::getColor(VisualizationUtil::CYAN),
        coarse_path_global);
    marker_array.markers.push_back(coarse_path_marker);

    for (size_t i = 0; i < coarse_path_global.size(); ++i) {
      geometry_msgs::msg::Pose arrow_pose;
      arrow_pose.position.x = coarse_path_global[i].x();
      arrow_pose.position.y = coarse_path_global[i].y();
      arrow_pose.position.z = 0.0;
      double yaw = coarse_path_global[i].z();
      arrow_pose.orientation.w = std::cos(yaw * 0.5);
      arrow_pose.orientation.x = 0.0;
      arrow_pose.orientation.y = 0.0;
      arrow_pose.orientation.z = std::sin(yaw * 0.5);
      auto arrow_marker = VisualizationUtil::createMarkerArrow(
          "map", "frenet_coarse_arrow", marker_id++, arrow_pose,
          VisualizationUtil::createMarkerScale(0.3, 0.06, 0.06),
          VisualizationUtil::getColor(VisualizationUtil::CYAN));
      marker_array.markers.push_back(arrow_marker);
    }
  }

  auto optimize_path = teb_planner->getOptimizePath();
  if (!optimize_path.empty()) {
    std::vector<Eigen::Vector3d> optimize_path_global;
    for (const auto &pose : optimize_path) {
      optimize_path_global.emplace_back(
          transformToGlobal(pose, robot_pos, robot_yaw));
    }
    auto optimize_path_marker = VisualizationUtil::createLineStripMarker(
        "map", "frenet_optimize_path", marker_id++,
        VisualizationUtil::getColor(VisualizationUtil::LIME_GREEN),
        optimize_path_global);
    marker_array.markers.push_back(optimize_path_marker);

    for (size_t i = 0; i < optimize_path_global.size(); ++i) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = optimize_path_global[i].x();
      pose.position.y = optimize_path_global[i].y();
      pose.position.z = 0.0;
      double yaw = optimize_path_global[i].z();
      pose.orientation.w = std::cos(yaw * 0.5);
      pose.orientation.x = 0.0;
      pose.orientation.y = 0.0;
      pose.orientation.z = std::sin(yaw * 0.5);
      auto point_marker = VisualizationUtil::createMarkerSphere(
          "map", "frenet_opt_point", marker_id++, pose,
          VisualizationUtil::createMarkerScale(0.08, 0.08, 0.08),
          VisualizationUtil::getColor(VisualizationUtil::GREEN));
      marker_array.markers.push_back(point_marker);
    }
  }

  frenet_debug_pub_->publish(marker_array);
}

Eigen::Vector3d
RosDebugBridge::transformToGlobal(const Eigen::Vector3d &local_point,
                                  const Eigen::Vector3d &robot_pos,
                                  double robot_yaw) {
  double cos_yaw = std::cos(robot_yaw);
  double sin_yaw = std::sin(robot_yaw);
  Eigen::Vector3d global_point;
  global_point.x() =
      cos_yaw * local_point.x() - sin_yaw * local_point.y() + robot_pos.x();
  global_point.y() =
      sin_yaw * local_point.x() + cos_yaw * local_point.y() + robot_pos.y();
  global_point.z() = local_point.z() + robot_yaw;
  return global_point;
}
} // namespace communication
} // namespace planning
