#pragma once
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/common/util/math.h"
namespace planning {
struct PathPoint2d {
  Point2d position;
  double s;
  double theta;
  double kappa;
  double dkappa;
  PathPoint2d() {
    position.x() = 0;
    position.y() = 0;
    theta = 0;
    kappa = 0;
    dkappa = 0;
    s = 0;
  }
  PathPoint2d(const Point2d &_position, double _s, double _theta, double _kappa,
              double _dkappa)
      : position(_position), theta(_theta), kappa(_kappa), dkappa(_dkappa),
        s(_s) {}
};

class Path2d {
public:
  Path2d() = default;
  Path2d(const std::vector<Point2d> &points);
  Path2d(const std::vector<Pose2d> &poses);

  Pose2d get_pose(const double path_s) const {
    auto iter = std::lower_bound(s_vec_.begin(), s_vec_.end(), path_s);
    if (iter == s_vec_.begin()) {
      return poses_.front();
    }
    if (iter == s_vec_.end()) {
      return poses_.back();
    }
    int index = std::distance(s_vec_.begin(), iter);
    double weight = (s_vec_[index - 1] - path_s) / (s_vec_[index] - path_s);
    double x = (1. - weight) * poses_[index - 1].position.x() +
               weight * poses_[index].position.x();
    double y = (1. - weight) * poses_[index - 1].position.y() +
               weight * poses_[index].position.y();
    double theta = planning::common::util::slerp(
        poses_[index - 1].theta, s_vec_[index - 1], poses_[index].theta,
        s_vec_[index], path_s);
    return Pose2d(Point2d(x, y), theta);
  }

  Pose2d get_pose(const Point2d &pt) const {
    double min_dist = std::numeric_limits<double>::max();
    Pose2d nearest_pose;
    for (const auto &pose : poses_) {
      double dist = (pose.position - pt).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        nearest_pose = pose;
      }
    }
    return nearest_pose;
  }

  int get_point_idx(const Point2d &pt, const int start_idx) const {
    double min_dist = std::numeric_limits<double>::max();
    int min_idx = start_idx;
    int incrase_cnt{0};
    for (int i = start_idx; i < poses_.size(); ++i) {
      // if (incrase_cnt > 10)
      //   break;
      double dist = (poses_[i].position - pt).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = i;
        // incrase_cnt = 0;
        continue;
      }
      // ++incrase_cnt;
    }
    return min_idx;
  }

  std::vector<Pose2d> pose_subset(const double start_s,
                                  const double end_s) const {
    auto iter = std::lower_bound(s_vec_.begin(), s_vec_.end(), start_s);
    int start_index = std::distance(s_vec_.begin(), iter);
    iter = std::lower_bound(s_vec_.begin(), s_vec_.end(), end_s);
    int end_index = std::distance(s_vec_.begin(), iter);
    return std::vector<Pose2d>(poses_.begin() + start_index,
                               poses_.begin() + end_index);
  }
  std::vector<Pose2d> pose_subset(const Eigen::Vector2d &robot_pos,
                                  const double path_s) const {
    const auto &start_idx = get_point_idx(robot_pos, 0);
    int tmp_start_idx = start_idx - 5 < 0 ? start_idx : start_idx - 5;
    const auto iter = std::lower_bound(s_vec_.begin(), s_vec_.end(),
                                       s_vec_[start_idx] + path_s);
    int end_index = std::distance(s_vec_.begin(), iter);
    std::vector<Pose2d> poses;
    for (size_t i = tmp_start_idx; i < end_index; i++) {
      poses.emplace_back(poses_[i]);
    }
    return poses;
  }

  std::vector<Eigen::Vector2d> subset(const Eigen::Vector2d &robot_pos,
                                      const double path_s, int &start_idx,
                                      double &local_ref_path_start_s,
                                      double &accu_s) const {

    start_idx = get_point_idx(robot_pos, start_idx);
    accu_s = s_vec_[start_idx];
    int tmp_start_idx = start_idx - 5 < 0 ? start_idx : start_idx - 5;

    const auto iter = std::lower_bound(s_vec_.begin(), s_vec_.end(),
                                       s_vec_[start_idx] + path_s);
    int end_index = std::distance(s_vec_.begin(), iter);

    local_ref_path_start_s = s_vec_[tmp_start_idx];
    AINFO << "start index: " << tmp_start_idx << " end idx: " << end_index
          << " local_ref_path_start_s: " << local_ref_path_start_s;
    std::vector<Eigen::Vector2d> points;
    for (size_t i = tmp_start_idx; i < end_index; i++) {
      points.emplace_back(poses_[i].position);
    }
    return points;
  }

  double length() const {
    if (this->s_vec_.empty())
      return 0;
    return this->s_vec_.back();
  }
  void clear() {
    this->points_.clear();
    this->poses_.clear();
    this->s_vec_.clear();
  }
  const std::vector<Pose2d> &poses() const { return poses_; }
  std::vector<Pose2d> *poses_ptr() { return &poses_; }

private:
  bool computePathProfile(const std::vector<Point2d> &xy_points,
                          std::vector<double> *headings,
                          std::vector<double> *s_vec,
                          std::vector<double> *kappas,
                          std::vector<double> *dkappas);

  std::vector<PathPoint2d> points_;
  std::vector<Pose2d> poses_;
  std::vector<double> s_vec_;
};

} // namespace planning
