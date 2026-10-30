// The node's input messages in arrival order as plain arrays (msgs.npz, written by
// tools/deploy/cpp_parity.py python --bag): lidar3d_replay --messages runs the online path (OnlineInputs)
// on them without ROS, and the ROS 2 bag replay must give the same frames.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "lidar3d/npz.hpp"
#include "lidar3d/online.hpp"

namespace lidar3d {

class MessageLog {
 public:
  enum Kind { kImu = 0, kOdom = 1, kInspvax = 2, kCloud = 3 };
  explicit MessageLog(const std::string& path);
  size_t size() const { return kind_.size(); }
  Kind kind(size_t i) const { return static_cast<Kind>(kind_[i]); }
  size_t index(size_t i) const { return static_cast<size_t>(index_[i]); }   // among the messages of its kind
  size_t num_clouds() const { return cloud_t_.size(); }

  double imu_t(size_t k) const { return imu_t_[k]; }
  Vec3 imu_w(size_t k) const { return Vec3(imu_w_[3 * k], imu_w_[3 * k + 1], imu_w_[3 * k + 2]); }
  OdomMsg odom(size_t k) const;
  InsSample inspvax(size_t k) const;
  CloudView cloud(size_t k) const;            // points into the log's bytes

  // message i into `in`; a cloud returns its frame (none while no pose covers it, or for a pose message)
  std::optional<PipelineOutput> feed(OnlineInputs& in, size_t i, const Detections* detections = nullptr) const;

 private:
  Npz z_;
  std::vector<int64_t> kind_, index_;
  std::vector<double> imu_t_, imu_w_, odom_t_, odom_p_, odom_q_, odom_v_;
  std::vector<double> ins_t_, ins_lla_, ins_azimuth_, ins_vel_;
  std::vector<int64_t> ins_status_, ins_pos_type_;
  std::vector<double> cloud_t_;
  std::vector<int64_t> cloud_width_, cloud_height_, cloud_step_, cloud_big_, cloud_layout_, cloud_off_;
  std::vector<std::vector<CloudField>> layouts_;
};

}  // namespace lidar3d
