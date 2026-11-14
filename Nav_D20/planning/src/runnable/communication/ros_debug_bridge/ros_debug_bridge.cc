#include "src/runnable/communication/ros_debug_bridge/ros_debug_bridge.h"
#include <algorithm>
#include <cmath>
#include "src/common/util/debug_datas.h"
#include "src/common/util/diagnostic.h"
#include "src/common/util/rosbag_recorder.h"
#include "src/common/util/tf_buffer.h"
#include "src/runnable/entry/planning_entry.h"
#include "src/runnable/mission/following/following.h"
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

  // ours (T55 review): the crosswalk yield as the planner sees it, for Foxglove on the dog (planning_live.json)
  crosswalk_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/planning/debug/crosswalk", 1);
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
  teb_debug_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planning/debug/teb_planner", 1);
  gradient_vec_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
      "/planning/debug/gradient_vector", 1);
  following_debug_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/planning/debug/following", 1);

  pub_flag_ = conf_.ros_debug_bridge_conf().pub_flag();
  set_pub_flag_srv_ = node_->create_service<std_srvs::srv::SetBool>(
      "/planning/debug/set_pub_flag",
      std::bind(&RosDebugBridge::SetPubFlag, this, std::placeholders::_1,
                std::placeholders::_2));
  return true;
}

bool RosDebugBridge::Proc() { return true; }

builtin_interfaces::msg::Time RosDebugBridge::DebugStamp() const {
  const auto *odometry =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  if (odometry == nullptr || odometry->timestamp <= 0.0) {
    return node_->now();
  }
  const int64_t ns = static_cast<int64_t>(odometry->timestamp * 1e9);
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(ns / 1000000000);
  stamp.nanosec = static_cast<uint32_t>(ns % 1000000000);
  return stamp;
}

void RosDebugBridge::StampMarkers(
    visualization_msgs::msg::MarkerArray *marker_array) const {
  const auto stamp = DebugStamp();
  for (auto &marker : marker_array->markers) {
    marker.header.stamp = stamp;
  }
}

bool RosDebugBridge::Publish() {
  rclcpp::spin_some(node_);
  std_msgs::msg::String debug_string;
  debug_string.data = DebugDatas::ConvertToString("planning");
  pub_debug_string_->publish(debug_string);
  RosbagRecorder::Instance()->add(debug_string,
                                  "/planning/debug/plotjuggler/string",
                                  "std_msgs/msg/String");
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
  if (!pub_flag_) {
    // Edge detection cycle boundary must run even without debug publishing
    Diagnostic::Instance()->clear();
    return true;
  }

  diagnostic_msgs::msg::DiagnosticArray diagnostic_msg;
  Diagnostic::Instance()->ConvertToRosMsg(diagnostic_msg);
  pub_diagnostic_msg_->publish(diagnostic_msg);
  // Edge detection cycle boundary (after the conversion has read the set)
  Diagnostic::Instance()->clear();

  nav_msgs::msg::OccupancyGrid occmap_msg;
  grid_map::GridMapRosConverter::toOccupancyGrid(
      *planning_entry_->preprocess_entry_ptr()
           ->cost_map_production()
           ->GridMap(),
      "obstacle", 255.0, 0.0, occmap_msg);
  // the obstacle layer holds 255 = free, 0 = obstacle (CostMapProduction::Proc, GlobalMapManager::LoadPGMMap):
  // dataMin 255, dataMax 0 maps free to 0 and obstacles to 100. With (0, 1) every free cell read 100 and the
  // obstacles 0: Foxglove showed the road as the obstacle (T47 review).
  // the grid map is stamped with the wall clock; its base_link pose on /tf has the odometry's stamp
  const auto debug_stamp = DebugStamp();
  occmap_msg.header.stamp = debug_stamp;
  RosbagRecorder::Instance()->add(occmap_msg,
                                  "/planning/debug/cost_map_production/occmap",
                                  "nav_msgs/msg/OccupancyGrid");
  local_occmap_pub_->publish(std::move(occmap_msg));

  // auto grid_map_msg = grid_map::GridMapRosConverter::toMessage(
  //     *planning_entry_->preprocess_entry_ptr()
  //          ->cost_map_production()
  //          ->GridMap());
  // RosbagRecorder::Instance()->add(grid_map_msg,
  //                                 "/planning/debug/cost_map_production/grid_map",
  //                                 "grid_map_msgs/msg/GridMap");
  // local_grid_map_pub_->publish(*grid_map_msg);
  // // pub obstacle gradient vector
  // {
  //   auto *grid_map = planning_entry_->preprocess_entry_ptr()
  //                        ->cost_map_production()
  //                        ->GridMap();
  //   visualization_msgs::msg::Marker gradient_vec_marker;
  //   gradient_vec_marker.header.frame_id = "base_link";
  //   gradient_vec_marker.header.stamp = node_->now();
  //   gradient_vec_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  //   gradient_vec_marker.action = visualization_msgs::msg::Marker::ADD;
  //   gradient_vec_marker.ns = "composite_gradient";
  //   gradient_vec_marker.id = 0;
  //   gradient_vec_marker.scale.x = 0.01;
  //   gradient_vec_marker.color.r = 1.0;
  //   gradient_vec_marker.color.g = 0.0;
  //   gradient_vec_marker.color.b = 0.0;
  //   gradient_vec_marker.color.a = 0.6;
  //   gradient_vec_marker.points.clear();
  //   constexpr double SCALE = 0.05;
  //   for (grid_map::GridMapIterator iterator(*grid_map); !iterator.isPastEnd();
  //        ++iterator) {
  //     Eigen::Vector3d position;
  //     grid_map->getPosition3("composite_cost", *iterator, position);
  //     geometry_msgs::msg::Point start_point;
  //     start_point.x = position.x();
  //     start_point.y = position.y();
  //     start_point.z = position.z();
  //     gradient_vec_marker.points.push_back(start_point);
  //     double grad_x = grid_map->at("composite_gradient_x", *iterator);
  //     double grad_y = grid_map->at("composite_gradient_y", *iterator);
  //     geometry_msgs::msg::Point end_point;
  //     end_point.x = start_point.x + SCALE * grad_x;
  //     end_point.y = start_point.y + SCALE * grad_y;
  //     end_point.z = start_point.z;
  //     gradient_vec_marker.points.push_back(end_point);
  //   }
  //   gradient_vec_pub_->publish(gradient_vec_marker);
  // }

  const auto &tf_msg = TfBuffer::Instance()->toTfMessage();
  tf_broadcaster_->sendTransform(tf_msg);
  tf2_msgs::msg::TFMessage tf2_msg;
  tf2_msg.transforms = tf_msg;
  RosbagRecorder::Instance()->add(tf2_msg, "/tf", "tf2_msgs/msg/TFMessage");
  PathTrackingDebug();
  CrosswalkDebug();
  if (!goal_pose_marker_.header.frame_id.empty()) {   // none before the first /goal_pose (an empty frame_id)
    goal_pose_marker_.header.stamp = DebugStamp();
    goal_pose_pub_->publish(goal_pose_marker_);
  }
  TEBDebug();
  FollowingDebug();

  auto collision_checker =
      planning_entry_->preprocess_entry_ptr()->collision_checker();
  if (collision_checker) {
    visualization_msgs::msg::MarkerArray marker_array;
    VisualizationUtil::appendDeleteAll(&marker_array);
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
    StampMarkers(&marker_array);
    collision_checker_pub_->publish(marker_array);
  }

  if (conf_.ros_debug_bridge_conf().enable_global_map_pub()) {
    auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
    if (gmm) {
      const size_t cur_count = global_occmap_pub_->get_subscription_count();
      if (cur_count > last_global_occmap_sub_count_) {
        nav_msgs::msg::OccupancyGrid occmap_msg;
        grid_map::GridMapRosConverter::toOccupancyGrid(
            gmm->global_map(), "obstacle", 255.0, 0.0, occmap_msg);   // 255 = free, as above
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

namespace {
std_msgs::msg::ColorRGBA CwColor(float r, float g, float b, float a) {
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}
}  // namespace

// ours (T55 review): in the map frame, the crosswalks near the robot (the one it is at or heading to coloured by the
// mode: approach yellow, align / wait red, go / cross green), perception's vehicles where the gate places them
// (red: the gate waits for it; an arrow for the velocity) and a line of text over the robot (mode, reason, how many
// it waits for). If the extrinsic were wrong the boxes would miss the vehicles' cells in the cost map.
void RosDebugBridge::CrosswalkDebug() {
  using planning::mission::CrosswalkYieldOutput;
  auto *mission_tracking = dynamic_cast<planning::mission::Tracking *>(
      planning_entry_->mission_entry_ptr()->GetMissionPtr(MissionConf::Tracking));
  auto *gmm = planning_entry_->preprocess_entry_ptr()->global_map_manager();
  if (mission_tracking == nullptr || gmm == nullptr) return;
  const auto odometry = planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  if (!odometry) return;
  const double rx = odometry->pose.position.x(), ry = odometry->pose.position.y();
  const CrosswalkYieldOutput &o = mission_tracking->crosswalk_yield();
  visualization_msgs::msg::MarkerArray arr;
  VisualizationUtil::appendDeleteAll(&arr);
  int32_t id = 1;
  auto base = [&](const std::string &ns, int32_t type) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.ns = ns;
    m.id = id++;
    m.type = type;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    return m;
  };
  std_msgs::msg::ColorRGBA mode_color = CwColor(0.6f, 0.6f, 0.6f, 0.8f);
  switch (o.mode) {
    case CrosswalkYieldOutput::kApproach: mode_color = CwColor(1.0f, 0.85f, 0.1f, 0.95f); break;
    case CrosswalkYieldOutput::kAlign:
    case CrosswalkYieldOutput::kWait: mode_color = CwColor(1.0f, 0.2f, 0.2f, 0.95f); break;
    case CrosswalkYieldOutput::kGo:
    case CrosswalkYieldOutput::kCross: mode_color = CwColor(0.2f, 0.9f, 0.3f, 0.95f); break;
    default: break;
  }
  const auto &crosswalks = gmm->crosswalks();
  for (size_t k = 0; k < crosswalks.size(); ++k) {
    const auto &poly = crosswalks[k];
    if (poly.empty()) continue;
    double cx = 0.0, cy = 0.0;
    for (const auto &p : poly) {
      cx += p.x();
      cy += p.y();
    }
    cx /= poly.size();
    cy /= poly.size();
    if (std::hypot(cx - rx, cy - ry) > 80.0) continue;   // only those near the robot
    const bool current = static_cast<int>(k) == o.crosswalk;
    auto m = base("crosswalks", visualization_msgs::msg::Marker::LINE_STRIP);
    m.scale.x = current ? 0.25 : 0.1;
    m.color = current ? mode_color : CwColor(1.0f, 1.0f, 1.0f, 0.6f);
    for (size_t i = 0; i <= poly.size(); ++i) {
      geometry_msgs::msg::Point p;
      p.x = poly[i % poly.size()].x();
      p.y = poly[i % poly.size()].y();
      p.z = 0.05;
      m.points.push_back(p);
    }
    arr.markers.push_back(m);
  }
  std::vector<planning::algorithm::yield::Object> objs;
  if (mission_tracking->crosswalk_objects_in_map(&objs)) {
    for (const auto &ob : objs) {
      if (ob.label != "vehicle" && ob.label != "cyclist") continue;   // what the gate yields to
      const bool waited = std::find(o.decision.ids.begin(), o.decision.ids.end(), ob.id) != o.decision.ids.end();
      auto box = base("vehicles", visualization_msgs::msg::Marker::CUBE);
      box.pose.position.x = ob.position.x();
      box.pose.position.y = ob.position.y();
      box.pose.position.z = 0.8;
      box.pose.orientation.z = std::sin(0.5 * ob.yaw);
      box.pose.orientation.w = std::cos(0.5 * ob.yaw);
      box.scale.x = std::max(ob.length, 0.5);
      box.scale.y = std::max(ob.width, 0.5);
      box.scale.z = 1.6;
      box.color = waited ? CwColor(1.0f, 0.15f, 0.15f, 0.6f) : CwColor(0.3f, 0.6f, 1.0f, 0.45f);
      arr.markers.push_back(box);
      if (ob.velocity.norm() > 0.3) {
        auto arrow = base("vehicle_velocity", visualization_msgs::msg::Marker::ARROW);
        geometry_msgs::msg::Point a, b;
        a.x = ob.position.x();
        a.y = ob.position.y();
        a.z = b.z = 1.7;
        b.x = a.x + ob.velocity.x();   // where it is in 1 s
        b.y = a.y + ob.velocity.y();
        arrow.points = {a, b};
        arrow.scale.x = 0.15;
        arrow.scale.y = 0.35;
        arrow.scale.z = 0.35;
        arrow.color = box.color;
        arrow.color.a = 0.95f;
        arr.markers.push_back(arrow);
      }
    }
  }
  if (o.mode != CrosswalkYieldOutput::kNone) {
    auto text = base("state", visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
    text.pose.position.x = rx;
    text.pose.position.y = ry;
    text.pose.position.z = 2.0;
    text.scale.z = 0.5;
    text.color = mode_color;
    text.text = std::string(planning::mission::ModeName(o.mode)) + " (" + o.decision.reason + ")" +
                (o.decision.ids.empty() ? "" : ", waiting for " + std::to_string(o.decision.ids.size())) +
                (o.stale ? ", perception stale" : "") + (o.unreleased ? ", on it without a go" : "");
    arr.markers.push_back(text);
  }
  StampMarkers(&arr);
  RosbagRecorder::Instance()->add(arr, "/planning/debug/crosswalk", "visualization_msgs/msg/MarkerArray");
  crosswalk_pub_->publish(arr);
}

void RosDebugBridge::PathTrackingDebug() {
  const auto &mission_base_tracking =
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Tracking);
  auto mission_tracking =
      dynamic_cast<planning::mission::Tracking *>(mission_base_tracking);
  visualization_msgs::msg::MarkerArray marker_array;
  // the footprints vary in number: clear the previous ones instead of waiting for their lifetime
  VisualizationUtil::appendDeleteAll(&marker_array);
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
  // the local planner's input: the reference cut from the global path (Tracking::spin_s_active)
  const auto &local_reference_2d = mission_tracking->local_reference_2d();
  if (local_reference_2d.size() >= 2) {
    marker_array.markers.push_back(VisualizationUtil::createLineStripMarker(
        "map", "local_reference_2d", marker_id++,
        VisualizationUtil::getColor(VisualizationUtil::PURPLE),
        local_reference_2d));
  }
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

  StampMarkers(&marker_array);
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
  VisualizationUtil::appendDeleteAll(&marker_array);
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

  StampMarkers(&marker_array);
  teb_debug_pub_->publish(marker_array);
}

void RosDebugBridge::FollowingDebug() {
  const auto &mission_base_following =
      planning_entry_->mission_entry_ptr()->GetMissionPtr(
          MissionConf::Following);
  auto mission_following =
      dynamic_cast<planning::mission::Following *>(mission_base_following);
  auto planner = mission_following == nullptr
                     ? nullptr
                     : mission_following->target_follow_planner();
  visualization_msgs::msg::MarkerArray marker_array;
  int32_t marker_id = 1;
  // 目标当前位置(红色)：通信桥最新观测的目标位置
  if (mission_following != nullptr &&
      mission_following->target_position_valid()) {
    const auto &tp = mission_following->target_position();
    geometry_msgs::msg::Pose tp_pose;
    tp_pose.position.x = tp.x();
    tp_pose.position.y = tp.y();
    tp_pose.position.z = 0.0;
    tp_pose.orientation.w = 1.0;
    geometry_msgs::msg::Vector3 tp_scale;
    tp_scale.x = 0.2;
    tp_scale.y = 0.2;
    tp_scale.z = 0.2;
    marker_array.markers.push_back(VisualizationUtil::createMarkerSphere(
        "map", "following_target_position", marker_id++, tp_pose, tp_scale,
        VisualizationUtil::getColor(VisualizationUtil::RED)));
  }
  
  if (planner == nullptr || planner->target_path().empty()) {
    if (!marker_array.markers.empty()) {
      StampMarkers(&marker_array);
      following_debug_pub_->publish(marker_array);
    }
    return;
  }
  // 原始入集点(橙色)：未截取、未平滑，用于调阈值观察滤波/限长效果
  const auto &raw_points = planner->raw_target_points();
  if (raw_points.size() >= 2) {
    const std::vector<Eigen::Vector2d> raw_pts(raw_points.begin(),
                                               raw_points.end());
    marker_array.markers.push_back(VisualizationUtil::createLineStripMarker(
        "map", "following_target_raw_path", marker_id++,
        VisualizationUtil::getColor(VisualizationUtil::ORANGE), raw_pts));
  }
  // 提取的目标路径(紫色)：截取+平滑后的输出
  const auto &target_path = planner->target_path();
  const std::vector<Eigen::Vector2d> path_pts(target_path.begin(),
                                              target_path.end());
  marker_array.markers.push_back(VisualizationUtil::createLineStripMarker(
      "map", "following_target_path", marker_id++,
      VisualizationUtil::getColor(VisualizationUtil::PURPLE), path_pts));
  // 跟随目标点(青色)：轨迹末端按时距回退 gap 的落点
  const auto &goal = planner->follow_goal();
  geometry_msgs::msg::Pose goal_pose;
  goal_pose.position.x = goal.position.x();
  goal_pose.position.y = goal.position.y();
  goal_pose.position.z = 0.0;
  goal_pose.orientation.w = 1.0;
  geometry_msgs::msg::Vector3 scale;
  scale.x = 0.15;
  scale.y = 0.15;
  scale.z = 0.15;
  marker_array.markers.push_back(VisualizationUtil::createMarkerSphere(
      "map", "following_goal", marker_id++,
      goal_pose, scale,
      VisualizationUtil::getColor(VisualizationUtil::CYAN)));
  // TEB 优化轨迹(绿色)：车身系优化轨迹转 map 系显示
  const auto &opt_path = planner->teb_optimize_path();
  if (opt_path.size() >= 2) {
    auto odometry =
        planning_entry_->communication_entry_ptr()
            ->MainBridge()
            ->odometry();
    if (odometry != nullptr) {
      const Eigen::Vector3d robot_pos(odometry->pose.position.x(),
                                      odometry->pose.position.y(), 0.0);
      const double robot_yaw = odometry->pose.euler()[2];
      std::vector<Eigen::Vector2d> traj_pts;
      traj_pts.reserve(opt_path.size());
      for (const auto &pose : opt_path) {
        const Eigen::Vector3d g = transformToGlobal(
            Eigen::Vector3d(pose.x(), pose.y(), 0.0),
            robot_pos, robot_yaw);
        traj_pts.emplace_back(g.x(), g.y());
      }
      marker_array.markers.push_back(VisualizationUtil::createLineStripMarker(
          "map", "following_teb_optimize_path", marker_id++,
          VisualizationUtil::getColor(VisualizationUtil::GREEN), traj_pts));
    }
  }
  StampMarkers(&marker_array);
  RosbagRecorder::Instance()->add(marker_array, "/planning/debug/following",
                                  "visualization_msgs/msg/MarkerArray");
  following_debug_pub_->publish(marker_array);
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
