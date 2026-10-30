#include "dogperc/pose_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dogperc {

namespace {

size_t search_left(const std::vector<double>& t, double v) {
  return static_cast<size_t>(std::lower_bound(t.begin(), t.end(), v) - t.begin());
}
size_t search_right(const std::vector<double>& t, double v) {
  return static_cast<size_t>(std::upper_bound(t.begin(), t.end(), v) - t.begin());
}

}  // namespace

bool PoseBuffer::add(double t, const Mat3& R, const Vec3& p) {
  if (!t_.empty() && t <= t_.back()) return false;
  t_.push_back(t);
  R_.push_back(R);
  p_.push_back(p);
  prune(t - cfg_.max_age);
  seg_valid_ = false;
  return true;
}

void PoseBuffer::truncate_from(double t) {
  const size_t n = search_left(t_, t);
  if (n < t_.size()) {
    t_.resize(n);
    R_.resize(n);
    p_.resize(n);
    seg_valid_ = false;
  }
}

void PoseBuffer::clear() {
  t_.clear();
  R_.clear();
  p_.clear();
  seg_valid_ = false;
}

void PoseBuffer::prune(double t_old) {
  long n = static_cast<long>(search_left(t_, t_old));
  n = std::min(n, static_cast<long>(t_.size()) - 2);
  if (n > 0) {
    t_.erase(t_.begin(), t_.begin() + n);
    R_.erase(R_.begin(), R_.begin() + n);
    p_.erase(p_.begin(), p_.begin() + n);
  }
}

void PoseBuffer::segments() const {
  if (seg_valid_) return;
  seg_.resize(t_.size() > 1 ? t_.size() - 1 : 0);
  for (size_t i = 0; i + 1 < t_.size(); ++i) seg_[i] = so3_log(R_[i].transpose() * R_[i + 1]);
  seg_valid_ = true;
}

void PoseBuffer::interpolate(const std::vector<double>& times, std::vector<Mat3>& R, std::vector<Vec3>& p) const {
  if (t_.empty()) throw PoseOutOfRange("pose buffer is empty");
  R.resize(times.size());
  p.resize(times.size());
  if (times.empty()) return;
  const double lo = t_.front() - cfg_.max_extrapolation, hi = t_.back() + cfg_.max_extrapolation;
  const auto [mn, mx] = std::minmax_element(times.begin(), times.end());
  if (*mn < lo || *mx > hi) {
    char msg[160];
    std::snprintf(msg, sizeof msg, "query [%.4f, %.4f] outside buffer [%.4f, %.4f]", *mn, *mx, t_.front(), t_.back());
    throw PoseOutOfRange(msg);
  }
  if (t_.size() == 1) {
    std::fill(R.begin(), R.end(), R_[0]);
    std::fill(p.begin(), p.end(), p_[0]);
    return;
  }
  segments();
  const long last = static_cast<long>(t_.size()) - 2;
  for (size_t q = 0; q < times.size(); ++q) {
    const double tq = times[q];
    if (tq > t_.back()) {
      extrapolate(tq, 1, R[q], p[q]);
      continue;
    }
    if (tq < t_.front()) {
      extrapolate(tq, -1, R[q], p[q]);
      continue;
    }
    const long i = std::clamp(static_cast<long>(search_right(t_, tq)) - 1, 0L, last);
    const double frac = (tq - t_[i]) / (t_[i + 1] - t_[i]);
    R[q] = R_[i] * so3_exp(frac * seg_[i]);
    p[q] = p_[i] + frac * (p_[i + 1] - p_[i]);
  }
}

void PoseBuffer::extrapolate(double time, int side, Mat3& R, Vec3& p) const {
  // the rate over at least extrapolation_window; an end segment shorter than 1 ms (two sources stamped
  // at nearly the same time) is a correction, not motion, so the rate is measured from the sample before
  const double w = cfg_.extrapolation_window, min_segment = 1e-3;
  const long n = static_cast<long>(t_.size());
  const long e = side > 0 ? n - 1 : 0;
  long k = e;
  if (n > 2 && std::abs(t_[e] - t_[e - side]) < min_segment) k = e - side;
  long j;
  if (side > 0) j = std::max(std::min(static_cast<long>(search_right(t_, t_[k] - w)) - 1, k - 1), 0L);
  else j = std::min(std::max(static_cast<long>(search_left(t_, t_[k] + w)), k + 1), n - 1);
  const double span = t_[k] - t_[j];
  const Vec3 v = (p_[k] - p_[j]) / span;
  const Vec3 omega = so3_log(R_[j].transpose() * R_[k]) / span;   // body-frame rate
  const double dt = time - t_[e];
  R = R_[e] * so3_exp(dt * omega);
  p = p_[e] + dt * v;
}

Mat4 PoseBuffer::pose_at(double t) const {
  std::vector<Mat3> R;
  std::vector<Vec3> p;
  interpolate({t}, R, p);
  return make_T(R[0], p[0]);
}

void ImuPropagator::reset() {
  bias_ = cfg_.gyro_bias;
  imu_.clear();
  have_anchor_ = false;
  v_est_.reset();
}

void ImuPropagator::on_odometry(double t, const Mat3& R_wb, const Vec3& p_wb, const std::optional<Vec3>& v_w) {
  if (have_anchor_ && t <= anchor_t_) return;     // stale / out-of-order odometry must not roll the anchor back
  Vec3 v;
  if (v_w) {
    v = *v_w;
  } else {
    if (have_anchor_ && t > anchor_t_) {
      const Vec3 v_fd = (p_wb - anchor_p_) / (t - anchor_t_);
      v_est_ = v_est_ ? Vec3(0.5 * *v_est_ + 0.5 * v_fd) : v_fd;
    }
    v = v_est_ ? *v_est_ : Vec3::Zero();
  }
  have_anchor_ = true;
  anchor_t_ = t;
  anchor_R_ = R_wb;
  anchor_p_ = p_wb;
  anchor_v_ = v;
  buffer_.truncate_from(t);
  buffer_.add(t, R_wb, p_wb);
  last_t_ = t;
  last_R_ = R_wb;
  for (const auto& [ti, w] : imu_)
    if (ti > t) propagate(ti, w);
}

void ImuPropagator::on_imu(double t, const Vec3& gyro) {
  imu_.emplace_back(t, gyro);
  while (!imu_.empty() && imu_.front().first < t - cfg_.imu_history) imu_.pop_front();
  if (!have_anchor_ || t <= last_t_) return;
  propagate(t, gyro);
}

void ImuPropagator::propagate(double t, const Vec3& gyro) {
  const Mat3 R = last_R_ * so3_exp((gyro - bias_) * (t - last_t_));
  buffer_.add(t, R, anchor_p_ + anchor_v_ * (t - anchor_t_));
  last_t_ = t;
  last_R_ = R;
}

}  // namespace dogperc
