#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/header.hpp"

namespace
{
constexpr int kUnvisited = -99;
constexpr int kNoise = -1;

struct Point3D
{
  double x;
  double y;
  double z;
};

const sensor_msgs::msg::PointField * findField(
  const sensor_msgs::msg::PointCloud2 & msg,
  const std::string & name)
{
  for (const auto & field : msg.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

const sensor_msgs::msg::PointField * findIntensityField(
  const sensor_msgs::msg::PointCloud2 & msg)
{
  const auto * field = findField(msg, "intensity");
  if (field != nullptr) {
    return field;
  }
  return findField(msg, "I");
}

template<typename T>
T readScalar(const uint8_t * data, bool is_bigendian)
{
  T value{};
  if (!is_bigendian) {
    std::memcpy(&value, data, sizeof(T));
    return value;
  }

  uint8_t reversed[sizeof(T)];
  for (size_t i = 0; i < sizeof(T); ++i) {
    reversed[i] = data[sizeof(T) - 1 - i];
  }
  std::memcpy(&value, reversed, sizeof(T));
  return value;
}

double readFieldAsDouble(
  const uint8_t * point_data,
  const sensor_msgs::msg::PointField & field,
  bool is_bigendian)
{
  const uint8_t * data = point_data + field.offset;

  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return static_cast<double>(readScalar<int8_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::UINT8:
      return static_cast<double>(readScalar<uint8_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::INT16:
      return static_cast<double>(readScalar<int16_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::UINT16:
      return static_cast<double>(readScalar<uint16_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::INT32:
      return static_cast<double>(readScalar<int32_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::UINT32:
      return static_cast<double>(readScalar<uint32_t>(data, is_bigendian));
    case sensor_msgs::msg::PointField::FLOAT32:
      return static_cast<double>(readScalar<float>(data, is_bigendian));
    case sensor_msgs::msg::PointField::FLOAT64:
      return readScalar<double>(data, is_bigendian);
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

double squaredDistance3D(const Point3D & a, const Point3D & b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

std::vector<int> regionQuery(
  const std::vector<Point3D> & points,
  int index,
  double eps_squared)
{
  std::vector<int> neighbors;
  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    if (squaredDistance3D(points[index], points[i]) <= eps_squared) {
      neighbors.push_back(i);
    }
  }
  return neighbors;
}

std::vector<int> dbscan(
  const std::vector<Point3D> & points,
  double eps,
  int min_samples)
{
  std::vector<int> labels(points.size(), kUnvisited);
  const double eps_squared = eps * eps;
  int cluster_id = 0;

  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    if (labels[i] != kUnvisited) {
      continue;
    }

    auto neighbors = regionQuery(points, i, eps_squared);
    if (static_cast<int>(neighbors.size()) < min_samples) {
      labels[i] = kNoise;
      continue;
    }

    labels[i] = cluster_id;
    std::deque<int> seeds(neighbors.begin(), neighbors.end());

    while (!seeds.empty()) {
      const int current = seeds.front();
      seeds.pop_front();

      if (labels[current] == kNoise) {
        labels[current] = cluster_id;
      }
      if (labels[current] != kUnvisited) {
        continue;
      }

      labels[current] = cluster_id;
      auto current_neighbors = regionQuery(points, current, eps_squared);
      if (static_cast<int>(current_neighbors.size()) >= min_samples) {
        for (const int neighbor : current_neighbors) {
          if (labels[neighbor] == kUnvisited || labels[neighbor] == kNoise) {
            seeds.push_back(neighbor);
          }
        }
      }
    }

    ++cluster_id;
  }

  return labels;
}

Point3D computeCentroid(const std::vector<Point3D> & points, const std::vector<int> & indices)
{
  Point3D centroid{0.0, 0.0, 0.0};
  for (const int index : indices) {
    centroid.x += points[index].x;
    centroid.y += points[index].y;
    centroid.z += points[index].z;
  }

  const double inv_size = 1.0 / static_cast<double>(indices.size());
  centroid.x *= inv_size;
  centroid.y *= inv_size;
  centroid.z *= inv_size;
  return centroid;
}
}  // namespace

class PerceptionRechargeNode : public rclcpp::Node
{
public:
  PerceptionRechargeNode()
  : Node("perception_recharge_node")
  {
    // 参数从 YAML 配置文件读取，代码中的默认值仅在未加载配置文件时兜底
    input_topic_ =
      declare_parameter<std::string>("input_topic", "/robot/topic/rdap/top_point_cloud2");
    output_topic_ =
      declare_parameter<std::string>("output_topic", "/reflector_position");
    intensity_threshold_ = declare_parameter("intensity_threshold", 250.0);
    dbscan_eps_ = declare_parameter("dbscan_eps", 0.1);
    declare_parameter("dbscan_min_samples", rclcpp::ParameterValue(20));
    dbscan_min_samples_ = static_cast<int>(get_parameter("dbscan_min_samples").as_int());
    min_reflector_distance_ = declare_parameter("min_reflector_distance", 0.5);
    max_reflector_distance_ = declare_parameter("max_reflector_distance", 1.5);
    position_x_offset_ = declare_parameter("position_x_offset", (323.7897 - 820.0) * 0.001);

    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      rclcpp::QoS(10),
      std::bind(&PerceptionRechargeNode::pointCloudCallback, this, std::placeholders::_1));

    position_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      output_topic_,
      rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "perception_recharge_node started");
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto * x_field = findField(*msg, "x");
    const auto * y_field = findField(*msg, "y");
    const auto * z_field = findField(*msg, "z");
    const auto * intensity_field = findIntensityField(*msg);

    if (x_field == nullptr || y_field == nullptr || z_field == nullptr || intensity_field == nullptr) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "PointCloud2 must contain x, y, z and intensity/I fields");
      return;
    }

    std::vector<Point3D> points;
    points.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height));

    const size_t point_count = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);
    for (size_t i = 0; i < point_count; ++i) {
      const uint8_t * point_data = msg->data.data() + i * msg->point_step;
      const double x_user = readFieldAsDouble(point_data, *x_field, msg->is_bigendian);
      const double y_user = readFieldAsDouble(point_data, *y_field, msg->is_bigendian);
      const double z_user = readFieldAsDouble(point_data, *z_field, msg->is_bigendian);
      const double intensity = readFieldAsDouble(point_data, *intensity_field, msg->is_bigendian);

      if (!std::isfinite(x_user) || !std::isfinite(y_user) || !std::isfinite(z_user) ||
        !std::isfinite(intensity))
      {
        continue;
      }
      if (intensity <= intensity_threshold_) {
        continue;
      }

      points.push_back(Point3D{-y_user, -x_user, z_user});
    }

    if (points.empty()) {
      return;
    }

    const auto labels = dbscan(points, dbscan_eps_, dbscan_min_samples_);
    std::unordered_map<int, std::vector<int>> clusters;
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
      if (labels[i] >= 0) {
        clusters[labels[i]].push_back(i);
      }
    }

    std::vector<std::vector<int>> valid_clusters;
    valid_clusters.reserve(clusters.size());
    for (auto & cluster : clusters) {
      if (static_cast<int>(cluster.second.size()) >= dbscan_min_samples_) {
        valid_clusters.push_back(std::move(cluster.second));
      }
    }

    std::sort(
      valid_clusters.begin(), valid_clusters.end(),
      [](const auto & lhs, const auto & rhs) {return lhs.size() > rhs.size();});

    if (valid_clusters.size() < 2) {
      return;
    }

    const Point3D centroid1 = computeCentroid(points, valid_clusters[0]);
    const Point3D centroid2 = computeCentroid(points, valid_clusters[1]);
    const double distance = std::sqrt(squaredDistance3D(centroid1, centroid2));

    if (distance < min_reflector_distance_ || distance > max_reflector_distance_) {
      RCLCPP_DEBUG(
        get_logger(), "Cluster distance %.2fm not in range %.2f-%.2fm",
        distance, min_reflector_distance_, max_reflector_distance_);
      return;
    }

    RCLCPP_INFO(get_logger(), "Found 2 reflectors, distance: %.2fm", distance);
    publishReflectorPosition(centroid1, centroid2, msg->header);
  }

  void publishReflectorPosition(
    const Point3D & centroid1,
    const Point3D & centroid2,
    const std_msgs::msg::Header & header)
  {
    Point3D midpoint{
      (centroid1.x + centroid2.x) * 0.5,
      (centroid1.y + centroid2.y) * 0.5,
      (centroid1.z + centroid2.z) * 0.5};

    const double direction_x = centroid2.x - centroid1.x;
    const double direction_y = centroid2.y - centroid1.y;

    const double perp1_x = -direction_y;
    const double perp1_y = direction_x;
    const double perp2_x = direction_y;
    const double perp2_y = -direction_x;

    const double dot1 = perp1_x * -midpoint.x + perp1_y * -midpoint.y;
    const double dot2 = perp2_x * -midpoint.x + perp2_y * -midpoint.y;

    double chosen_x = perp2_x;
    double chosen_y = perp2_y;
    if (dot1 > dot2) {
      chosen_x = perp1_x;
      chosen_y = perp1_y;
    }

    const double norm = std::sqrt(chosen_x * chosen_x + chosen_y * chosen_y);
    if (norm > 0.0) {
      chosen_x /= norm;
      chosen_y /= norm;
    }

    double yaw = std::atan2(chosen_y, chosen_x);
    yaw *= -1.0;

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = header;
    pose_msg.pose.position.x = midpoint.x + position_x_offset_;
    pose_msg.pose.position.y = midpoint.y * -1.0;
    pose_msg.pose.position.z = midpoint.z;
    pose_msg.pose.orientation.x = 0.0;
    pose_msg.pose.orientation.y = 0.0;
    pose_msg.pose.orientation.z = std::sin(yaw * 0.5);
    pose_msg.pose.orientation.w = std::cos(yaw * 0.5);

    position_pub_->publish(pose_msg);

    RCLCPP_INFO(
      get_logger(), "Published target pose at (%.2f, %.2f, %.2f)",
      pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr position_pub_;

  // 从 YAML 配置文件读取的参数
  std::string input_topic_;
  std::string output_topic_;
  double intensity_threshold_;
  double dbscan_eps_;
  int dbscan_min_samples_;
  double min_reflector_distance_;
  double max_reflector_distance_;
  double position_x_offset_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PerceptionRechargeNode>());
  rclcpp::shutdown();
  return 0;
}
