// The online node's input handling without ROS (lidar_3d_detection/online.py, ros_utils.py): PointCloud2
// bytes -> LidarScan, the RTK IMU's units, INSPVAX -> poses (ENU origin at the first good fix, lever arm,
// heading check) or LIO odometry. The ROS 2 node (deploy/ros2/lidar_3d_detection) copies its messages into these
// structs; lidar_3d_replay --messages feeds the message log tools/deploy/cpp_parity.py python --bag writes.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "lidar3d/localization.hpp"
#include "lidar3d/pipeline.hpp"

namespace lidar3d {

// the node: section of tools/deploy/export_cpp_config.py --robot (lidar_3d_detection/node_config.py); the rest
// of the node parameters (extrinsic, base height, self filter) are already in the pipeline sections
struct NodeConfig {
  std::string lidar_topic = "/lidar_points", time_field = "timestamp", time_mode = "absolute";
  int noise_tag_mask = 0;                     // drop points whose tag has these bits (Livox: 3)
  double lidar_time_offset = 0.0;             // [s] added to the lidar times
  std::string imu_topic = "/imu/data";
  std::string imu_convention = "ros";         // ros: rad/s in the body frame | kygi: the RTK IMU (deg/s)
  std::string localization = "odom";          // odom: nav_msgs/Odometry | inspvax: kygi665_msgs/INSPVAX
  std::string odom_topic = "/Odometry", odom_twist = "none";   // none | world | body
  std::string inspvax_topic = "/rte/rdap/rtk/inspvax";
  Vec3 ins_lever_arm = Vec3::Zero();          // [m] INSPVAX position point in the body frame (resolved)
  double ins_azimuth_offset_deg = 180.0;
  std::string world_frame = "camera_init";
  bool publish_cloud = false;
};

NodeConfig parse_node_config(const std::string& yaml_text);
NodeConfig load_node_config(const std::string& yaml_path);   // throws when the file has no node: section

// sensor_msgs/PointField datatypes
enum : uint8_t { kInt8 = 1, kUint8 = 2, kInt16 = 3, kUint16 = 4, kInt32 = 5, kUint32 = 6, kFloat32 = 7, kFloat64 = 8 };

struct CloudField {
  std::string name;
  uint32_t offset = 0;
  uint8_t datatype = 0;
  uint32_t count = 1;
};

// a PointCloud2's layout and bytes, without ROS (the node points into the message, no copy)
struct CloudView {
  double stamp = 0.0;                         // header stamp [s]
  uint32_t width = 0, height = 0, point_step = 0;
  bool is_bigendian = false;
  std::vector<CloudField> fields;
  const uint8_t* data = nullptr;
  size_t size = 0;                            // bytes at data
};

// ros_utils.scan_from_msg: time_mode absolute (s) | absolute_ns | relative (s after the header stamp) |
// relative_ns; points at the origin or not finite dropped, and those whose tag has noise_tag_mask bits
LidarScan scan_from_cloud(const CloudView& cloud, const std::string& time_field, const std::string& time_mode,
                          int noise_tag_mask = 0, const std::string& intensity_field = "intensity");

struct OdomMsg {
  double t = 0.0;
  Vec3 p = Vec3::Zero();
  Eigen::Vector4d q_xyzw{0.0, 0.0, 0.0, 1.0};
  Vec3 v_linear = Vec3::Zero();               // twist.linear
};

enum class LogLevel { Info, Warn, Error };
using LogFn = std::function<void(LogLevel, const std::string&)>;

// a topic whose stamps jump back more than this (a bag played again, a clock reset) restarts everything, as a
// new node: the pose buffer would drop every new pose as out of order, and past its 3 s no scan gets a pose. A
// message of another topic still from before the jump (the last cloud of a loop arriving after the next loop's
// first INS) is dropped, not taken as a second jump
constexpr double kTimeJumpBack = 1.0;       // [s]

class OnlineInputs {
 public:
  // log: messages already throttled (each kind of warning at most every 2 or 5 s, as the Python node)
  OnlineInputs(NodeConfig cfg, PerceptionPipeline& pipe, LogFn log = {});
  void on_imu(double t, const Vec3& angular_velocity);
  void on_odom(const OdomMsg& m);
  void on_inspvax(const InsSample& s);
  LidarScan scan(const CloudView& cloud) const;   // with the lidar time offset
  // none while no pose covers the scan; `detections` (det frame) bypass the detector
  std::optional<PipelineOutput> on_cloud(const CloudView& cloud, const Detections* detections = nullptr);
  // on_cloud after scan(): the replays look up the given detections by the scan's stamp
  std::optional<PipelineOutput> on_scan(const LidarScan& scan, const Detections* detections = nullptr);
  void reset();                               // the pipeline, the ENU origin, the heading check
  bool have_origin() const { return origin_.has_value(); }
  const NodeConfig& config() const { return cfg_; }

 private:
  void log(LogLevel level, const std::string& key, double every_s, const std::string& text);
  bool stamp(const char* kind, double t);     // resets on a jump back; false: from before the restart, dropped

  NodeConfig cfg_;
  PerceptionPipeline& pipe_;
  LogFn log_;
  std::optional<EnuOrigin> origin_;
  HeadingCheck heading_;
  bool heading_warned_ = false;
  std::map<std::string, double> last_stamp_;  // kind -> latest stamp
  std::optional<double> restart_;             // the stamp that jumped back
  std::map<std::string, double> before_;      // last_stamp_ before it
  std::map<std::string, std::chrono::steady_clock::time_point> last_log_;
};

// one object of perception_interface/ObjectArray (deploy/ros2) in the det frame of the scan: gravity aligned,
// yaw with the body, origin on the ground (PerceptionPipeline::to_local)
struct PlannerObject {
  LocalTrack track;
  struct Path {
    double probability = 0.0;
    std::vector<Vec2> xy;                     // at `times`
    std::vector<Mat2> covariance;
  };
  std::vector<double> times;                  // [s] after the stamp
  std::vector<Path> paths;                    // most likely first; none without a prediction
};
std::vector<PlannerObject> planner_objects(const PerceptionPipeline& pipe, const PipelineOutput& out);

}  // namespace lidar3d
