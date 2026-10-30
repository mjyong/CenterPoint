#ifndef SERES_PERCEPTION_LIDAR_3D_DETECTION_H_
#define SERES_PERCEPTION_LIDAR_3D_DETECTION_H_

#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "perception_interface/msg/lidar_object_array.hpp"

#include "lidar3d/online.hpp"
#include "lidar3d/pipeline.hpp"
#ifdef LIDAR3D_WITH_CUDA
#include "lidar3d/detection/detector.hpp"
#endif

namespace seres {
namespace perception {

class Lidar3dDetection : public rclcpp::Node {
 public:
  Lidar3dDetection();
  bool Init();

 private:
  void OnCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  // config/lidar_3d_detection.yaml of this install: $SERES_PERCEPTION_INSTALL_DIR/lidar_3d_detection, else the
  // package's own prefix (ament index)
  static std::string DefaultConfig();

  std::string config_path_;
  std::string model_dir_;
  std::string precision_ = "fp16";
  std::string local_frame_ = "lidar_det";
  std::string world_frame_ = "camera_init";
  bool publish_markers_ = true;
  bool publish_cloud_ = false;
  bool gpu_ = true;

  std::string objects_topic_ = "/perception/lidar_3d/objects";
  std::string tracks_topic_ = "/perception/lidar_3d/tracks";
  std::string predictions_topic_ = "/perception/lidar_3d/predictions";
  std::string cloud_topic_ = "/perception/lidar_3d/cloud";

#ifdef LIDAR3D_WITH_CUDA
  std::unique_ptr<lidar3d::Detector> detector_;
#endif
  std::unique_ptr<lidar3d::PerceptionPipeline> pipe_;
  std::unique_ptr<lidar3d::OnlineInputs> inputs_;
  std::chrono::steady_clock::time_point last_stats_{}, last_drop_{};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::SubscriptionBase::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<perception_interface::msg::LidarObjectArray>::SharedPtr objects_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tracks_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr preds_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  // publish_tf: world_frame -> local_frame and -> the lidar frame, for viewers on a replayed bag (README)
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace perception
}  // namespace seres

#endif
