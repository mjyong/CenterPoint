#include "../include/lidar_3d_detection.h"
#include "../include/logging.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <yaml-cpp/yaml.h>
#ifdef LIDAR3D_HAVE_KYGI
#include <kygi665_msgs/msg/inspvax.hpp>
#endif

#include "lidar3d/parallel.hpp"
#include "ros_convert.hpp"
#ifdef LIDAR3D_WITH_CUDA
#include "lidar3d/detection/gpu_detector.hpp"
#include "lidar3d/detection/trt_engine.hpp"
#endif

namespace seres {
namespace perception {

namespace {

std::string OptionalString(const YAML::Node& node, const char* key, const std::string& fallback) {
  if (!node || !node[key]) return fallback;
  return node[key].as<std::string>();
}

}  // namespace

Lidar3dDetection::Lidar3dDetection() : Node("lidar_3d_detection") {}

std::string Lidar3dDetection::DefaultConfig() {
  const char* root = std::getenv("SERES_PERCEPTION_INSTALL_DIR");
  if (root && *root) return std::string(root) + "/lidar_3d_detection/config/lidar_3d_detection.yaml";
  try {
    return ament_index_cpp::get_package_prefix("lidar_3d_detection") + "/config/lidar_3d_detection.yaml";
  } catch (const std::exception&) {
    return "config/lidar_3d_detection.yaml";
  }
}

bool Lidar3dDetection::Init() {
  // an empty parameter (the launch file's default) also means the installed config
  config_path_ = declare_parameter<std::string>("config", "");
  if (config_path_.empty()) config_path_ = DefaultConfig();
  model_dir_ = declare_parameter<std::string>("model_dir", "");
  precision_ = declare_parameter<std::string>("precision", "fp16");
  local_frame_ = declare_parameter<std::string>("local_frame", "lidar_det");
  gpu_ = declare_parameter<bool>("gpu", true);
  publish_markers_ = declare_parameter<bool>("publish_markers", true);
  lidar3d::set_num_threads(static_cast<int>(declare_parameter<int64_t>("threads", 1)));

  YAML::Node root_yaml;
  try {
    root_yaml = YAML::LoadFile(config_path_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "failed to read config " << config_path_ << ": " << e.what();
    return false;
  }
  const YAML::Node node = root_yaml["node"];
  if (model_dir_.empty()) model_dir_ = OptionalString(node, "model_dir", "");
  if (!model_dir_.empty() && model_dir_.front() != '/') {
    const auto package_root = std::filesystem::path(config_path_).parent_path().parent_path();
    model_dir_ = (package_root / model_dir_).lexically_normal().string();
  }
  objects_topic_ = OptionalString(node, "objects_topic", objects_topic_);
  tracks_topic_ = OptionalString(node, "tracks_topic", tracks_topic_);
  predictions_topic_ = OptionalString(node, "predictions_topic", predictions_topic_);
  cloud_topic_ = OptionalString(node, "cloud_topic", cloud_topic_);
  if (model_dir_.empty()) {
    LOG(ERROR) << "model_dir is empty: set the ROS parameter or node.model_dir in " << config_path_;
    return false;
  }

  lidar3d::PipelineConfig cfg;
  lidar3d::NodeConfig nc;
  try {
    cfg = lidar3d::load_config(config_path_);
    nc = lidar3d::load_node_config(config_path_);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }
  world_frame_ = nc.world_frame;
  publish_cloud_ = nc.publish_cloud;

#ifndef LIDAR3D_WITH_CUDA
  (void)gpu_;
  (void)precision_;
  (void)cfg;
  LOG(ERROR) << "lidar_3d_detection was built without LIDAR3D_WITH_CUDA: no detector";
  return false;
#else
  try {
    detector_ = gpu_ ? lidar3d::make_gpu_detector(model_dir_, precision_, cfg.detection_filter)
                     : lidar3d::make_trt_detector(model_dir_, precision_, cfg.detection_filter);
  } catch (const std::exception& e) {
    LOG(ERROR) << "detector init failed: " << e.what();
    return false;
  }
  pipe_ = std::make_unique<lidar3d::PerceptionPipeline>(cfg, detector_.get());
  inputs_ = std::make_unique<lidar3d::OnlineInputs>(nc, *pipe_, [](lidar3d::LogLevel level, const std::string& text) {
    if (level == lidar3d::LogLevel::Error) LOG(ERROR) << text;
    else if (level == lidar3d::LogLevel::Warn) LOG(WARNING) << text;
    else LOG(INFO) << text;
  });

  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      nc.imu_topic, rclcpp::QoS(400), [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
        inputs_->on_imu(lidar3d::ros::seconds(msg->header.stamp), lidar3d::ros::angular_velocity(*msg));
      });
  if (nc.localization == "inspvax") {
#ifdef LIDAR3D_HAVE_KYGI
    pose_sub_ = create_subscription<kygi665_msgs::msg::INSPVAX>(
        nc.inspvax_topic, rclcpp::QoS(200), [this](kygi665_msgs::msg::INSPVAX::ConstSharedPtr msg) {
          inputs_->on_inspvax(lidar3d::ros::ins_sample(*msg));
        });
#else
    LOG(ERROR) << "localization inspvax needs a build with kygi665_msgs";
    return false;
#endif
  } else {
    pose_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        nc.odom_topic, rclcpp::QoS(50), [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
          inputs_->on_odom(lidar3d::ros::odom(*msg));
        });
  }
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      nc.lidar_topic, rclcpp::SensorDataQoS().keep_last(5),
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { OnCloud(msg); });

  objects_pub_ = create_publisher<perception_interface::msg::LidarObjectArray>(objects_topic_, 5);
  tracks_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(tracks_topic_, 5);
  preds_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(predictions_topic_, 5);
  cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_, 2);

  LOG(INFO) << "lidar_3d_detection up: config " << config_path_ << ", model " << model_dir_ << " (" << precision_
            << ", " << (gpu_ ? "GPU" : "host") << "), lidar " << nc.lidar_topic << ", " << nc.localization
            << " " << (nc.localization == "inspvax" ? nc.inspvax_topic : nc.odom_topic)
            << ", imu " << nc.imu_topic << ", objects " << objects_topic_;
  return true;
#endif
}

void Lidar3dDetection::OnCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  const auto now = std::chrono::steady_clock::now();
  std::optional<lidar3d::PipelineOutput> out;
  try {
    out = inputs_->on_cloud(lidar3d::ros::cloud_view(*msg));
  } catch (const std::exception& e) {
    if (now - last_drop_ >= std::chrono::seconds(2)) {
      last_drop_ = now;
      LOG(ERROR) << "scan dropped: " << e.what();
    }
    return;
  }
  if (!out) return;
  objects_pub_->publish(lidar3d::ros::object_array(*pipe_, *out, local_frame_, world_frame_));
  if (publish_markers_) {
    tracks_pub_->publish(lidar3d::ros::track_markers(*out, world_frame_));
    preds_pub_->publish(lidar3d::ros::prediction_markers(*out, world_frame_));
  }
  if (publish_cloud_) cloud_pub_->publish(lidar3d::ros::det_cloud(*out, local_frame_));
  // once a second, as deploy/ros2's node (tools/deploy/soak_report.py reads these lines)
  if (now - last_stats_ < std::chrono::seconds(1)) return;
  last_stats_ = now;
  std::string t;
  for (const char* k : {"preprocess_ms", "det_gpu_ms", "det_post_ms", "track_ms", "predict_ms", "total_ms"}) {
    auto it = out->timings.find(k);
    if (it == out->timings.end()) continue;
    char buf[48];
    std::snprintf(buf, sizeof(buf), " %s %.1f", k, it->second);
    t += buf;
  }
  LOG(INFO) << out->detections.size() << " dets, " << out->tracks.size() << " tracks |" << t;
}

}  // namespace perception
}  // namespace seres
