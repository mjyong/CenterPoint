// Recorded sequences as the offline tools store them (lidar_3d_detection/io.py):
//   meta.json, scans/NNNNNN.npz (xyz, intensity, t, stamp), imu.npy (t, wx, wy, wz), odom.npy (t, xyz, quat, vel)
// events() replays them in time order exactly as SequenceReader.events does.
#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lidar3d/geometry.hpp"
#include "lidar3d/preprocess.hpp"

namespace lidar3d {

struct ImuEvent {
  double t;
  Vec3 gyro;
};
struct OdomEvent {
  double t;
  Mat3 R;
  Vec3 p;
  std::optional<Vec3> v;     // none when the recording has no twist (NaN columns)
};
struct ScanEvent {
  size_t index;              // scan file index
  double stamp;              // with the lidar time offset
};
using SequenceEvent = std::variant<ImuEvent, OdomEvent, ScanEvent>;

class SequenceReader {
 public:
  // lidar_time_offset: overrides meta.json's (added to scan stamps and point times)
  explicit SequenceReader(const std::string& dir, std::optional<double> lidar_time_offset = std::nullopt);
  size_t num_scans() const { return scan_files_.size(); }
  LidarScan load_scan(size_t index) const;   // with the lidar time offset applied
  // time-ordered events from 1 s before the first scan to the last; odometry emitted at t + odom_latency
  std::vector<SequenceEvent> events(double odom_latency = 0.0) const;
  double lidar_time_offset() const { return lidar_time_offset_; }

 private:
  std::string dir_;
  double lidar_time_offset_ = 0.0;
  std::vector<std::string> scan_files_;
  std::vector<double> imu_;     // (M, 4)
  std::vector<double> odom_;    // (K, 11)
};

}  // namespace lidar3d
