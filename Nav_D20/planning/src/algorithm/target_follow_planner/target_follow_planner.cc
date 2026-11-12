#include "src/algorithm/target_follow_planner/target_follow_planner.h"
#include "src/common/util/debug_datas.h"

#include <limits>

using planning::common::util::DebugDatas;

namespace planning {
namespace algorithm {

TargetFollowPlanner::TargetFollowPlanner(
    const TargetFollowParams &params,
    const TargetPathExtractParams &path_extract_params,
    const planning::config::TebPlannerConf &teb_conf,
    const std::function<double(double, double)> &is_obstacle_func,
    const std::function<double(double, double)> &obstacle_dist_func,
    const std::function<double(double, double)> &composite_cost_func,
    const std::function<bool(double, double, double)> &is_collision_func)
    : params_(params), extract_params_(path_extract_params),
      teb_conf_(teb_conf),
      teb_planner_(teb_conf, is_obstacle_func, obstacle_dist_func,
                   composite_cost_func, is_collision_func),
      is_collision_func_(is_collision_func),
      reference_speed_(teb_conf.dynamic_constraints().reference_velocity()) {
  ResetFilters();
}

void TargetFollowPlanner::Reset() {
  points_.clear();
  consecutive_jumps_ = 0;
  filtered_target_speed_ = 0.0;
  teb_planner_.reset();
  ResetFilters();
}

std::optional<Eigen::Vector3d>
TargetFollowPlanner::Plan(const Point2d &target_pos, const Point2d &target_vel,
                          const Pose2d &robot_pose,
                          const Eigen::Vector3d &robot_vel) {
  filtered_target_speed_ = speed_filter_.Update(target_vel.norm());

  target_path_ = ExtractTargetPath(target_pos, robot_pose.position);
  if (target_path_.empty()) {
    return std::nullopt;
  }

  if (!ComputeFollowGoal(target_path_, robot_pose.position)) {
    return std::nullopt;
  }
  const Point2d &last_point = points_.back();
  follow_goal_.theta = std::atan2(last_point.y() - robot_pose.position.y(),
                                  last_point.x() - robot_pose.position.x());
  DebugDatas::Instance()->emplace("/follow/raw_target/x", last_point.x());
  DebugDatas::Instance()->emplace("/follow/raw_target/y", last_point.y());
  DebugDatas::Instance()->emplace("/follow/target/x",
                                  follow_goal_.position.x());
  DebugDatas::Instance()->emplace("/follow/target/y",
                                  follow_goal_.position.y());
  DebugDatas::Instance()->emplace("/follow/target/theta", follow_goal_.theta);
  DebugDatas::Instance()->emplace("/follow/target_speed",
                                  filtered_target_speed_);

  const double end_dist = (target_path_.back() - robot_pose.position).norm();
  // if (end_dist < params_.min_follow_dist) {
  //   return Eigen::Vector3d::Zero();
  // }

  const Pose2d goal_body = ToBodyFrame(follow_goal_, robot_pose);

  const double goal_dist = goal_body.position.norm();
  const double goal_theta_err =
      std::atan2(std::sin(goal_body.theta), std::cos(goal_body.theta));
  if (goal_dist < params_.arrive_dist_thresh &&
      std::abs(goal_theta_err) < params_.arrive_theta_thresh) {
    return Eigen::Vector3d::Zero();
  }

  const double gap_err = end_dist - follow_gap_;
  const double v_desired =
      std::clamp(filtered_target_speed_ + params_.spacing_gain * gap_err, 0.0,
                 reference_speed_);
  teb_planner_.set_target_speed(v_desired);
  DebugDatas::Instance()->emplace("/follow/desired_speed", v_desired);

  const std::vector<Eigen::Vector3d> waypoints =
      BuildStraightRefPath(goal_body);
  const Eigen::Vector3d final_pose(goal_body.position.x(),
                                   goal_body.position.y(), goal_body.theta);
  planning::Transform3d tf = planning::Transform3d::Identity();
  tf.translate(
      Eigen::Vector3d(robot_pose.position.x(), robot_pose.position.y(), 0.0));
  tf.rotate(Eigen::AngleAxisd(robot_pose.theta, Eigen::Vector3d::UnitZ()));
  bool constant_final_theta =
      !is_collision_func_(final_pose(0), final_pose(1), final_pose(2));
  DebugDatas::Instance()->emplace("/follow/constant_final_theta",
                                  constant_final_theta);
  if (!teb_planner_.solver(waypoints, Eigen::Vector3d::Zero(), robot_vel,
                           final_pose, constant_final_theta, tf,
                           v_desired < 0.1)) {
    AERROR << "follow teb solver failed, output zero cmd.";
    return Eigen::Vector3d::Zero();
  }
  return teb_planner_.getVelocityCmd(teb_conf_.velocity_cmd_time());
}

std::vector<Point2d>
TargetFollowPlanner::ExtractTargetPath(const Point2d &target_pos,
                                       const Point2d &robot_pos) {
  TryAddPoint(target_pos);
  if (points_.empty()) {
    return {};
  }

  TrimToMaxLength();

  std::vector<Point2d> path = points_;
  Smooth(&path);
  return path;
}

void TargetFollowPlanner::ResetFilters() {
  filter_x_ = planning::common::util::MeanFilter(
      static_cast<std::uint_fast8_t>(extract_params_.filter_window));
  filter_y_ = planning::common::util::MeanFilter(
      static_cast<std::uint_fast8_t>(extract_params_.filter_window));
  speed_filter_ = planning::common::util::MeanFilter(
      static_cast<std::uint_fast8_t>(params_.speed_filter_window));
}

void TargetFollowPlanner::TryAddPoint(const Point2d &pt) {
  const Point2d filtered(filter_x_.Update(pt.x()), filter_y_.Update(pt.y()));
  if ((pt - filtered).norm() > extract_params_.jump_reset_thresh) {
    ++consecutive_jumps_;
    if (consecutive_jumps_ >= extract_params_.max_consecutive_jumps) {
      AWARN << "target path extract: consecutive jumps " << consecutive_jumps_
            << ", re-anchor path with new point.";
      ResetFilters();
      consecutive_jumps_ = 0;
    }
    return;
  }
  consecutive_jumps_ = 0;
  if (points_.empty()) {
    points_.push_back(filtered);
    return;
  }

  const double spacing = extract_params_.accept_thresh;
  const Point2d last = points_.back();
  const double dist = (filtered - last).norm();
  if (spacing <= 1e-9) {
    if (dist > 1e-9) {
      points_.push_back(filtered);
    }
    return;
  }
  if (dist <= spacing) {
    return;
  }
  const int n = static_cast<int>(std::floor(dist / spacing));
  const Point2d dir((filtered - last) / dist);
  for (int k = 1; k <= n; ++k) {
    points_.push_back(Point2d((last + k * spacing * dir).eval()));
  }
  const double remain = dist - n * spacing;
  if (remain > 1e-6) {
    points_.push_back(filtered);
  }
}

void TargetFollowPlanner::TrimToMaxLength() {
  if (points_.size() < 2) {
    return;
  }
  double accu_len = 0.0;
  int keep_start = static_cast<int>(points_.size()) - 1;
  for (int i = static_cast<int>(points_.size()) - 1; i > 0; --i) {
    const double seg_len = (points_[i] - points_[i - 1]).norm();
    if (accu_len + seg_len >= extract_params_.max_path_length) {
      keep_start = i - 1;
      const double remain = extract_params_.max_path_length - accu_len;
      if (remain > 1e-6 && seg_len > 1e-6) {
        const double ratio = remain / seg_len;
        const Point2d boundary(
            (points_[i - 1] + ratio * (points_[i] - points_[i - 1])).eval());
        points_.erase(points_.begin(), points_.begin() + keep_start);
        points_.front() = boundary;
      } else {
        points_.erase(points_.begin(), points_.begin() + keep_start);
      }
      return;
    }
    accu_len += seg_len;
  }
}

void TargetFollowPlanner::Smooth(std::vector<Point2d> *pts) const {
  if (pts == nullptr || pts->size() < 3) {
    return;
  }
  const int n = static_cast<int>(pts->size());
  std::vector<double> u(n, 0.0);
  double total_len = 0.0;
  for (int i = 1; i < n; ++i) {
    total_len += ((*pts)[i] - (*pts)[i - 1]).norm();
    u[i] = total_len;
  }
  if (total_len < 2.0 * extract_params_.smooth_resample_interval) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    u[i] /= total_len;
  }
  // 最小二乘逼近: (A^T A)·coef = A^T·y，基函数 [u², u, 1]
  Eigen::Matrix3d ata = Eigen::Matrix3d::Zero();
  Eigen::Vector3d atx = Eigen::Vector3d::Zero();
  Eigen::Vector3d aty = Eigen::Vector3d::Zero();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d basis(u[i] * u[i], u[i], 1.0);
    ata += basis * basis.transpose();
    atx += basis * (*pts)[i].x();
    aty += basis * (*pts)[i].y();
  }
  const Eigen::Vector3d cx = ata.ldlt().solve(atx);
  const Eigen::Vector3d cy = ata.ldlt().solve(aty);
  const int m =
      std::max(4, static_cast<int>(std::round(
                      total_len / extract_params_.smooth_resample_interval)) +
                      1);
  std::vector<Point2d> smoothed;
  smoothed.reserve(m);
  for (int k = 0; k < m; ++k) {
    const double t = static_cast<double>(k) / (m - 1);
    const Eigen::Vector3d basis(t * t, t, 1.0);
    smoothed.emplace_back(Point2d(basis.dot(cx), basis.dot(cy)));
  }
  *pts = std::move(smoothed);
}

bool TargetFollowPlanner::ComputeFollowGoal(const std::vector<Point2d> &path,
                                            const Point2d &robot_pos) {
  if (path.empty()) {
    return false;
  }
  const double heading_to_end = std::atan2(path.back().y() - robot_pos.y(),
                                           path.back().x() - robot_pos.x());

  if (path.size() == 1) {
    follow_goal_ = Pose2d(path.front(), heading_to_end);
    return true;
  }

  const int m = static_cast<int>(path.size());
  std::vector<double> s(m, 0.0);
  for (int i = 1; i < m; ++i) {
    s[i] = s[i - 1] + (path[i] - path[i - 1]).norm();
  }

  // 恒定时距:期望间距 = 滤波后目标速度 × 时距,钳位到 [min, max]
  const double gap =
      std::clamp(filtered_target_speed_ * params_.headway_time,
                 params_.min_follow_dist, params_.max_follow_dist);
  follow_gap_ = gap;

  double s_robot = 0.0;
  double best_dist = std::numeric_limits<double>::max();
  for (int i = 1; i < m; ++i) {
    const Point2d d(path[i] - path[i - 1]);
    const double seg_len_sq = d.squaredNorm();
    double t = 0.0;
    if (seg_len_sq > 1e-12) {
      t = std::clamp((robot_pos - path[i - 1]).dot(d) / seg_len_sq, 0.0, 1.0);
    }
    const Point2d proj(path[i - 1] + t * d);
    const double dist = (robot_pos - proj).norm();
    if (dist < best_dist) {
      best_dist = dist;
      s_robot = s[i - 1] + t * std::sqrt(seg_len_sq);
    }
  }

  // 前向钳位: goal 最小为机器人投影弧长 + min_goal_ahead,
  // 冲过期望间距时 goal 保持在机器人前方(最远不超过轨迹末端),
  // 由 ACC 律减速、目标前移自然恢复间距,避免 goal 落到身后
  const double s_goal =
      std::clamp(std::max(s.back() - gap, s_robot + params_.min_goal_ahead),
                 0.0, s.back());

  if (s_goal <= 1e-9) {
    follow_goal_ = Pose2d(path.front(), heading_to_end);
    return true;
  }
  for (int i = 1; i < m; ++i) {
    if (s_goal <= s[i] + 1e-9) {
      const double seg = s[i] - s[i - 1];
      if (seg < 1e-9) {
        follow_goal_ = Pose2d(path[i], heading_to_end);
        return true;
      }
      const double ratio = (s_goal - s[i - 1]) / seg;
      const Point2d goal_pt(
          (path[i - 1] + ratio * (path[i] - path[i - 1])).eval());
      follow_goal_ = Pose2d(goal_pt, heading_to_end);
      return true;
    }
  }
  follow_goal_ = Pose2d(path.back(), heading_to_end);
  return true;
}

Pose2d TargetFollowPlanner::ToBodyFrame(const Pose2d &global_pose,
                                        const Pose2d &robot_pose) {
  const double dx = global_pose.position.x() - robot_pose.position.x();
  const double dy = global_pose.position.y() - robot_pose.position.y();
  const double c = std::cos(robot_pose.theta);
  const double s = std::sin(robot_pose.theta);
  Pose2d body;
  body.position = Point2d(c * dx + s * dy, -s * dx + c * dy);
  double dtheta = global_pose.theta - robot_pose.theta;
  dtheta = std::atan2(std::sin(dtheta), std::cos(dtheta));
  body.theta = dtheta;
  return body;
}

std::vector<Eigen::Vector3d>
TargetFollowPlanner::BuildStraightRefPath(const Pose2d &goal_body) {
  constexpr double kRefPathSpacing = 0.25;
  const double dist = goal_body.position.norm();
  const double heading =
      std::atan2(goal_body.position.y(), goal_body.position.x());
  const int n =
      std::max(2, static_cast<int>(std::ceil(dist / kRefPathSpacing)));
  std::vector<Eigen::Vector3d> waypoints;
  waypoints.reserve(n + 1);
  waypoints.emplace_back(0.0, 0.0, heading);
  for (int k = 1; k <= n; ++k) {
    const double ratio = static_cast<double>(k) / n;
    waypoints.emplace_back(ratio * goal_body.position.x(),
                           ratio * goal_body.position.y(), heading);
  }
  return waypoints;
}

} // namespace algorithm
} // namespace planning
