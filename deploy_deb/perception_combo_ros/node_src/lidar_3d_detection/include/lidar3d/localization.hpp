// kygi665 INSPVAX -> localization pose in a local ENU frame, and the RTK IMU's units (inspvax.py).
//
// Heading only (yaw = azimuth_offset - azimuth, 180 deg with the dual antennas swapped as on 0921, 0 once
// fixed); position latitude / longitude / height converted to ENU about the first good fix and moved from
// the INS point to the body origin by the lever arm. Roll / pitch are not used (auto_level levels the
// det frame on the ground instead).
#pragma once

#include <optional>

#include "lidar3d/geometry.hpp"

namespace lidar3d {

constexpr int kInsSolutionGood = 3;
constexpr int kInsRtkFixed = 56;
constexpr double kKygiGyroScale = kPi / 180.0;    // the RTK IMU publishes deg/s

struct InsSample {           // the INSPVAX fields used
  double stamp = 0.0;
  int ins_status = 0;
  int pos_type = 0;
  double latitude = 0.0, longitude = 0.0, height = 0.0;   // deg, deg, m
  double azimuth = 0.0;                                   // deg
  double east_velocity = 0.0, north_velocity = 0.0, up_velocity = 0.0;
};

class EnuOrigin {
 public:
  EnuOrigin(double lat, double lon, double h);
  Vec3 enu(double lat, double lon, double h) const;

 private:
  Vec3 ecef_;
  Mat3 R_;
};

struct InsPose {
  double t;
  Mat3 R;
  Vec3 p, v;
};

// none when the INS solution is not good
std::optional<InsPose> pose_from_inspvax(const InsSample& s, const EnuOrigin& origin, const Vec3& lever_arm,
                                         double azimuth_offset_deg);
// kygi RTK IMU angular velocity (deg/s, body axes) -> rad/s
inline Vec3 gyro_from_kygi_imu(const Vec3& w_deg) { return w_deg * kKygiGyroScale; }

// the robot should walk along body +y: wrong once >= min_rows samples go backwards and 4x more than forwards
class HeadingCheck {
 public:
  explicit HeadingCheck(double min_speed = 0.5, int min_rows = 300) : min_speed_(min_speed), min_rows_(min_rows) {}
  void add(const Mat3& R, const Vec3& v);
  bool wrong() const { return n_back_ >= min_rows_ && n_back_ > 4 * n_fwd_; }

 private:
  double min_speed_;
  int min_rows_;
  int n_fwd_ = 0, n_back_ = 0;
};

}  // namespace lidar3d
