// Time-indexed body poses for per-point motion compensation (pose_buffer.py).
//
// PoseBuffer holds T_world_body(t) at IMU rate: rotation SLERP'd, translation linear, a little linear
// extrapolation past either end. ImuPropagator builds that stream from low-rate odometry (LIO or INS)
// and high-rate gyro, re-propagating the buffered tail whenever a late odometry anchor arrives.
#pragma once

#include <deque>
#include <optional>
#include <vector>

#include "dogperc/common.hpp"
#include "dogperc/config.hpp"
#include "dogperc/geometry.hpp"

namespace dogperc {

class PoseOutOfRange : public Error {
 public:
  using Error::Error;
};

class PoseBuffer {
 public:
  explicit PoseBuffer(PoseBufferConfig cfg = {}) : cfg_(cfg) {}
  size_t size() const { return t_.size(); }
  double t_min() const { return t_.front(); }
  double t_max() const { return t_.back(); }
  bool add(double t, const Mat3& R, const Vec3& p);   // false for out-of-order samples
  void truncate_from(double t);                      // drop every sample at time >= t
  void clear();
  // T_world_body at each time; throws PoseOutOfRange outside the buffer (+ max_extrapolation)
  void interpolate(const std::vector<double>& times, std::vector<Mat3>& R, std::vector<Vec3>& p) const;
  Mat4 pose_at(double t) const;

 private:
  void prune(double t_old);
  void segments() const;
  void extrapolate(double time, int side, Mat3& R, Vec3& p) const;

  PoseBufferConfig cfg_;
  std::vector<double> t_;
  std::vector<Mat3> R_;
  std::vector<Vec3> p_;
  mutable std::vector<Vec3> seg_;    // so3_log(R_i^T R_i+1), rebuilt after a change
  mutable bool seg_valid_ = false;
};

class ImuPropagator {
 public:
  ImuPropagator(PoseBuffer& buffer, ImuConfig cfg = {}) : buffer_(buffer), cfg_(cfg), bias_(cfg.gyro_bias) {}
  // v_w: world-frame velocity, or none (estimated from consecutive odometry positions)
  void on_odometry(double t, const Mat3& R_wb, const Vec3& p_wb, const std::optional<Vec3>& v_w = std::nullopt);
  void on_imu(double t, const Vec3& gyro);
  void set_gyro_bias(const Vec3& b) { bias_ = b; }
  void reset();                                      // forget the IMU samples and the anchor (not the buffer)

 private:
  void propagate(double t, const Vec3& gyro);

  PoseBuffer& buffer_;
  ImuConfig cfg_;
  Vec3 bias_;
  std::deque<std::pair<double, Vec3>> imu_;
  bool have_anchor_ = false;
  double anchor_t_ = 0.0;
  Mat3 anchor_R_;
  Vec3 anchor_p_, anchor_v_;
  double last_t_ = 0.0;
  Mat3 last_R_;
  std::optional<Vec3> v_est_;
};

}  // namespace dogperc
