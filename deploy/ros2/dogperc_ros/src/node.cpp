// The dog perception node (docs/DEPLOY_ORIN.md section 8): the C++ pipeline of deploy/cpp on live topics.
//
//   ros2 run dogperc_ros dogperc_node --ros-args -p config:=kygi.yaml -p model_dir:=nusc_pp_dyn
//
// config: tools/deploy/export_cpp_config.py --robot kygi (the pipeline and the node: section: topics,
// localization, time format); model_dir: the exported model with engines built on this machine.
// Subscribes to the config's lidar, IMU and INSPVAX / odometry topics (remap them with -r as usual).
// Publishes
//   ~/objects       dog_perception_msgs/ObjectArray  tracks and predicted paths in the det frame, for the planner
//   ~/tracks        visualization_msgs/MarkerArray   boxes, ids, speed (world frame), as tools/dog/ros2_node.py
//   ~/predictions   visualization_msgs/MarkerArray   one line per predicted mode
//   ~/cloud         sensor_msgs/PointCloud2          the detector's input (node.publish_cloud)
//   /tf             world_frame -> local_frame, world_frame -> the lidar frame, only with publish_tf (viewers on a
//                   replayed bag: Foxglove, RViz; off on the robot, whose own TF tree has the lidar frame)
// Callbacks run on one thread in arrival order, like the Python node, so its replay of a bag
// (cpp_parity.py python --bag) is the reference; dogperc_bag_replay checks that offline.
#include <cstdio>
#include <memory>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <tf2_ros/transform_broadcaster.h>

#include "dogperc/parallel.hpp"
#include "ros_convert.hpp"
#ifdef DOGPERC_WITH_CUDA
#include "dogperc/detection/gpu_detector.hpp"
#include "dogperc/detection/trt_engine.hpp"
#endif

namespace dogperc::ros {

class DogPerceptionNode : public rclcpp::Node {
 public:
  DogPerceptionNode() : Node("dog_perception") {
    const auto config = declare_parameter<std::string>("config", "");
    const auto model_dir = declare_parameter<std::string>("model_dir", "");
    const auto precision = declare_parameter<std::string>("precision", "fp16");
    const bool gpu = declare_parameter<bool>("gpu", true);
    local_frame_ = declare_parameter<std::string>("local_frame", "dog_det");
    publish_markers_ = declare_parameter<bool>("publish_markers", true);
    if (declare_parameter<bool>("publish_tf", false)) tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    set_num_threads(static_cast<int>(declare_parameter<int64_t>("threads", 1)));
    if (config.empty()) throw std::runtime_error("parameter config: the YAML of tools/deploy/export_cpp_config.py --robot");
    if (model_dir.empty()) throw std::runtime_error("parameter model_dir: the exported model directory with its engines");
    const PipelineConfig cfg = load_config(config);
    const NodeConfig nc = load_node_config(config);
#ifdef DOGPERC_WITH_CUDA
    detector_ = gpu ? make_gpu_detector(model_dir, precision, cfg.detection_filter)
                    : make_trt_detector(model_dir, precision, cfg.detection_filter);
#else
    (void)gpu;
    (void)precision;
    throw std::runtime_error("dogperc_ros was built without DOGPERC_WITH_CUDA: no detector");
#endif
    pipe_ = std::make_unique<PerceptionPipeline>(cfg, detector_.get());
    inputs_ = std::make_unique<OnlineInputs>(nc, *pipe_, [this](LogLevel level, const std::string& text) {
      if (level == LogLevel::Error) RCLCPP_ERROR(get_logger(), "%s", text.c_str());
      else if (level == LogLevel::Warn) RCLCPP_WARN(get_logger(), "%s", text.c_str());
      else RCLCPP_INFO(get_logger(), "%s", text.c_str());
    });
    world_frame_ = nc.world_frame;
    publish_cloud_ = nc.publish_cloud;

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(nc.imu_topic, rclcpp::QoS(400), [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {
      inputs_->on_imu(seconds(m->header.stamp), angular_velocity(*m));
    });
    if (nc.localization == "inspvax") {
#ifdef DOGPERC_HAVE_KYGI
      ins_sub_ = create_subscription<kygi665_msgs::msg::INSPVAX>(
          nc.inspvax_topic, rclcpp::QoS(200),
          [this](kygi665_msgs::msg::INSPVAX::ConstSharedPtr m) { inputs_->on_inspvax(ins_sample(*m)); });
#else
      throw std::runtime_error("localization inspvax: dogperc_ros was built without kygi665_msgs");
#endif
    } else {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          nc.odom_topic, rclcpp::QoS(50), [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { inputs_->on_odom(odom(*m)); });
    }
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        nc.lidar_topic, rclcpp::SensorDataQoS().keep_last(5),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_cloud(*m); });
    objects_pub_ = create_publisher<dog_perception_msgs::msg::ObjectArray>("~/objects", 5);
    tracks_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/tracks", 5);
    preds_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/predictions", 5);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud", 2);
    RCLCPP_INFO(get_logger(), "dog perception up: %s, model %s (%s, %s), lidar %s, localization %s", config.c_str(),
                model_dir.c_str(), precision.c_str(), gpu ? "GPU pre-processing" : "host pre-processing",
                nc.lidar_topic.c_str(), nc.localization.c_str());
  }

 private:
  void on_cloud(const sensor_msgs::msg::PointCloud2& msg) {
    std::optional<PipelineOutput> out;
    try {
      out = inputs_->on_cloud(cloud_view(msg));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "scan dropped: %s", e.what());
      return;
    }
    if (!out) return;
    objects_pub_->publish(object_array(*pipe_, *out, local_frame_, world_frame_));
    if (publish_markers_) {
      tracks_pub_->publish(track_markers(*out, world_frame_));
      preds_pub_->publish(prediction_markers(*out, world_frame_));
    }
    if (publish_cloud_) cloud_pub_->publish(det_cloud(*out, local_frame_));
    if (tf_) tf_->sendTransform(viewer_transforms(*pipe_, *out, msg, local_frame_, world_frame_));
    std::string t;
    for (const char* k : {"preprocess_ms", "det_gpu_ms", "det_post_ms", "track_ms", "predict_ms", "total_ms"}) {
      auto it = out->timings.find(k);
      if (it == out->timings.end()) continue;
      char buf[48];
      std::snprintf(buf, sizeof(buf), " %s %.1f", k, it->second);
      t += buf;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "%zu dets, %zu tracks |%s", out->detections.size(),
                         out->tracks.size(), t.c_str());
  }

  std::unique_ptr<Detector> detector_;
  std::unique_ptr<PerceptionPipeline> pipe_;
  std::unique_ptr<OnlineInputs> inputs_;
  std::string local_frame_, world_frame_;
  bool publish_markers_ = true, publish_cloud_ = false;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
#ifdef DOGPERC_HAVE_KYGI
  rclcpp::Subscription<kygi665_msgs::msg::INSPVAX>::SharedPtr ins_sub_;
#endif
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<dog_perception_msgs::msg::ObjectArray>::SharedPtr objects_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tracks_pub_, preds_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace dogperc::ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<dogperc::ros::DogPerceptionNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("dog_perception"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
