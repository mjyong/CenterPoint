#include "src/runnable/mission/tracking/crosswalk_yield.h"

#include <algorithm>
#include <cmath>

namespace planning {
namespace mission {

using algorithm::yield::Object;
using algorithm::yield::Vec2;
using algorithm::yield::Mat2;

namespace {

double Wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

Eigen::Matrix2d Rot(double a) {
  Eigen::Matrix2d r;
  r << std::cos(a), -std::sin(a), std::sin(a), std::cos(a);
  return r;
}

// the path from the robot on: the robot, then the global path past its closest segment
std::vector<Vec2> PathAhead(const Vec2 &robot, const std::vector<Vec2> &path, double length) {
  std::vector<Vec2> out{robot};
  if (path.size() < 2) {
    if (path.size() == 1 && (path[0] - robot).norm() > 1e-6) out.push_back(path[0]);
    return out;
  }
  size_t best = 0;
  double best_d = std::numeric_limits<double>::infinity(), best_u = 0.0;
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    const Vec2 a = path[i], ab = path[i + 1] - path[i];
    const double l2 = ab.squaredNorm();
    const double u = l2 > 0 ? std::clamp((robot - a).dot(ab) / l2, 0.0, 1.0) : 0.0;
    const double d = (a + u * ab - robot).norm();
    if (d < best_d) {
      best_d = d;
      best = i;
      best_u = u;
    }
  }
  const Vec2 proj = path[best] + best_u * (path[best + 1] - path[best]);
  if ((proj - robot).norm() > 0.05) out.push_back(proj);
  double s = 0.0;
  for (size_t i = best + 1; i < path.size(); ++i) {
    s += (path[i] - out.back()).norm();
    if ((path[i] - out.back()).norm() > 1e-6) out.push_back(path[i]);
    if (s > length) break;
  }
  return out;
}

// the point `s` metres along a polyline
Vec2 PointAt(const std::vector<Vec2> &p, double s) {
  double acc = 0.0;
  for (size_t i = 0; i + 1 < p.size(); ++i) {
    const double l = (p[i + 1] - p[i]).norm();
    if (acc + l >= s && l > 0) return p[i] + (s - acc) / l * (p[i + 1] - p[i]);
    acc += l;
  }
  return p.back();
}

}  // namespace

const char *ModeName(CrosswalkYieldOutput::Mode m) {
  switch (m) {
    case CrosswalkYieldOutput::kApproach: return "approach";
    case CrosswalkYieldOutput::kAlign: return "align";
    case CrosswalkYieldOutput::kWait: return "wait";
    case CrosswalkYieldOutput::kGo: return "go";
    case CrosswalkYieldOutput::kCross: return "cross";
    default: return "none";
  }
}

void CrosswalkYield::Configure(const CrosswalkYieldConfig &cfg) {
  cfg_ = cfg;
  gate_ = algorithm::yield::YieldGate(cfg_.gate);
  Reset();
}

void CrosswalkYield::Reset() {
  gate_.Reset();
  passed_.clear();
  mode_ = CrosswalkYieldOutput::kNone;
  current_ = -1;
}

void CrosswalkYield::AddOdom(double stamp, const Vec2 &xy, double yaw) {
  if (!odom_.empty() && stamp <= odom_.back().t) {
    if (stamp < odom_.back().t - 1.0) odom_.clear();   // time went back (a new bag): start over
    else return;
  }
  odom_.push_back({stamp, xy, yaw});
  while (odom_.size() > 2 && odom_.back().t - odom_.front().t > 5.0) odom_.pop_front();
}

void CrosswalkYield::SetObjects(double stamp, const std::vector<Object> &objects_det) {
  objects_stamp_ = stamp;
  objects_det_ = objects_det;
}

bool CrosswalkYield::PoseAt(double t, Vec2 *xy, double *yaw) const {
  if (odom_.empty()) return false;
  if (t <= odom_.front().t) {
    if (odom_.front().t - t > cfg_.max_pose_gap) return false;
    *xy = odom_.front().xy;
    *yaw = odom_.front().yaw;
    return true;
  }
  if (t >= odom_.back().t) {
    if (t - odom_.back().t > cfg_.max_pose_gap) return false;
    *xy = odom_.back().xy;
    *yaw = odom_.back().yaw;
    return true;
  }
  for (size_t i = 0; i + 1 < odom_.size(); ++i) {
    const auto &a = odom_[i], &b = odom_[i + 1];
    if (t >= a.t && t <= b.t) {
      const double u = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0.0;
      *xy = a.xy + u * (b.xy - a.xy);
      *yaw = a.yaw + u * Wrap(b.yaw - a.yaw);
      return true;
    }
  }
  return false;
}

bool CrosswalkYield::ObjectsInMap(std::vector<Object> *out) const {
  out->clear();
  if (objects_stamp_ < 0) return false;
  Vec2 xy;
  double yaw;
  if (!PoseAt(objects_stamp_, &xy, &yaw)) return false;
  const double yaw_db = cfg_.det_to_base_yaw_deg * M_PI / 180.0;
  const Eigen::Matrix2d R = Rot(yaw) * Rot(yaw_db);          // det axes -> map axes
  const Vec2 t = xy + Rot(yaw) * Vec2(cfg_.det_to_base_x, cfg_.det_to_base_y);
  for (Object o : objects_det_) {
    o.position = R * o.position + t;
    o.velocity = R * o.velocity;
    o.yaw = Wrap(o.yaw + yaw + yaw_db);
    if (o.has_covariance) o.covariance = R * o.covariance * R.transpose();
    for (auto &m : o.modes) {
      for (auto &p : m.positions) p = R * p + t;
      for (auto &c : m.covariances) c = R * c * R.transpose();
    }
    out->push_back(o);
  }
  return true;
}

CrosswalkYieldOutput CrosswalkYield::Step(double now, const Vec2 &robot_xy, double robot_yaw, double robot_speed,
                                          const std::vector<Vec2> &global_path) {
  CrosswalkYieldOutput out;
  out.mode = CrosswalkYieldOutput::kNone;
  if (!cfg_.enable || crosswalks_.empty()) {
    mode_ = CrosswalkYieldOutput::kNone;
    return out;
  }
  const std::vector<Vec2> ahead = PathAhead(robot_xy, global_path, cfg_.lookahead);
  // the robot on a crosswalk: never stop there
  int inside = -1;
  for (size_t k = 0; k < crosswalks_.size(); ++k) {
    if (algorithm::yield::InsidePolygon(robot_xy, crosswalks_[k])) inside = static_cast<int>(k);
  }
  std::vector<Object> objs;
  const bool have_pose = ObjectsInMap(&objs);
  const double stamp = have_pose ? objects_stamp_ : std::numeric_limits<double>::quiet_NaN();
  for (const auto &o : objs) {
    if (std::find(cfg_.gate.classes.begin(), cfg_.gate.classes.end(), o.label) != cfg_.gate.classes.end()) {
      ++out.objects;
    }
  }
  const bool moving = robot_speed > cfg_.moving_speed;
  if (inside >= 0) {
    if (mode_ != CrosswalkYieldOutput::kCross) passed_.insert(inside);
    mode_ = CrosswalkYieldOutput::kCross;
    current_ = inside;
    out.mode = mode_;
    out.crosswalk = inside;
    algorithm::yield::YieldConfig c = cfg_.gate;
    const auto a = algorithm::yield::Assess(ahead, objs, now, stamp, c);
    out.decision = gate_.Update(a, now, moving, true);
    out.stale = a.stale;
    return out;
  }
  // the next crosswalk along the path, not crossed yet
  int next = -1;
  double s_in = 0.0, s_out = 0.0;
  for (size_t k = 0; k < crosswalks_.size(); ++k) {
    if (passed_.count(static_cast<int>(k))) continue;
    double a = 0.0, b = 0.0;
    if (algorithm::yield::CrosswalkSpan(ahead, crosswalks_[k], &a, &b) && (next < 0 || a < s_in)) {
      next = static_cast<int>(k);
      s_in = a;
      s_out = b;
    }
  }
  if (next < 0) {
    mode_ = CrosswalkYieldOutput::kNone;
    current_ = -1;
    return out;
  }
  if (next != current_) {   // a new crosswalk ahead
    current_ = next;
    mode_ = CrosswalkYieldOutput::kApproach;
  }
  out.crosswalk = next;
  out.s_in = s_in;
  out.s_out = s_out;
  out.s_stop = s_in - (0.5 * cfg_.gate.dog_length + cfg_.stop_margin);
  const Vec2 a_in = PointAt(ahead, s_in), a_out = PointAt(ahead, s_out);
  const double across = std::atan2(a_out.y() - a_in.y(), a_out.x() - a_in.x());
  out.heading_error = Wrap(across - robot_yaw);
  // the gate checks from the robot to the exit plus a margin
  algorithm::yield::YieldConfig check = cfg_.gate;
  check.path_length = s_out + cfg_.check_beyond_exit;
  if (mode_ == CrosswalkYieldOutput::kApproach) {
    if (out.s_stop > cfg_.hold_distance) {
      out.mode = mode_;
      out.speed_scale = std::clamp(out.s_stop / cfg_.slow_distance, cfg_.min_speed_scale, 1.0);
      return out;
    }
    mode_ = CrosswalkYieldOutput::kAlign;   // arrived: a new wait at the curb
    gate_.Reset();
  }
  const auto a = algorithm::yield::Assess(ahead, objs, now, stamp, check);
  out.stale = a.stale;
  if (mode_ == CrosswalkYieldOutput::kAlign) {
    out.decision = gate_.Update(a, now, false, false);   // timers run while turning
    if (std::abs(out.heading_error) > cfg_.align_tolerance_deg * M_PI / 180.0) {
      out.mode = mode_;
      out.hold = true;
      out.hold_omega = std::clamp(cfg_.align_gain * out.heading_error, -cfg_.max_align_omega, cfg_.max_align_omega);
      return out;
    }
    mode_ = CrosswalkYieldOutput::kWait;
    if (out.decision.go) mode_ = CrosswalkYieldOutput::kGo;
  } else {
    out.decision = gate_.Update(a, now, moving && mode_ == CrosswalkYieldOutput::kGo, false);
    mode_ = out.decision.go ? CrosswalkYieldOutput::kGo : CrosswalkYieldOutput::kWait;
  }
  out.mode = mode_;
  if (mode_ == CrosswalkYieldOutput::kWait) {
    out.hold = true;
    out.hold_omega = 0.0;
  }
  return out;
}

}  // namespace mission
}  // namespace planning
