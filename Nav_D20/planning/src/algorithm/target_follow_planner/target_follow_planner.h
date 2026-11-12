#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <vector>

#include "src/algorithm/teb_planner/teb_planner.h"
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/common/util/mean_filter.h"

namespace planning {
namespace algorithm {

struct TargetFollowParams {
  double max_follow_dist = 3.0;
  double min_follow_dist = 0.5;
  double headway_time = 2.0;
  double spacing_gain = 0.5;
  int speed_filter_window = 10;
  double min_goal_ahead = 0.3;
  double arrive_dist_thresh = 0.25;
  double arrive_theta_thresh = 0.5;
};

struct TargetPathExtractParams {
  int filter_window = 30;
  double jump_reset_thresh = 0.5;
  int max_consecutive_jumps = 5;
  double accept_thresh = 0.25;
  double max_path_length = 10.0;
  double smooth_resample_interval = 0.25;
};

class TargetFollowPlanner {
public:
  TargetFollowPlanner(
      const TargetFollowParams &params,
      const TargetPathExtractParams &path_extract_params,
      const planning::config::TebPlannerConf &teb_conf,
      const std::function<double(double, double)> &is_obstacle_func,
      const std::function<double(double, double)> &obstacle_dist_func,
      const std::function<double(double, double)> &composite_cost_func,
      const std::function<bool(double, double, double)> &is_collision_func);
  ~TargetFollowPlanner() = default;

  void set_params(const TargetFollowParams &params) { params_ = params; }
  const TargetFollowParams &params() const { return params_; }

  std::optional<Eigen::Vector3d> Plan(const Point2d &target_pos,
                                      const Point2d &target_vel,
                                      const Pose2d &robot_pose,
                                      const Eigen::Vector3d &robot_vel);

  void Reset();

  const std::vector<Point2d> &target_path() const { return target_path_; }
  const std::vector<Point2d> &raw_target_points() const { return points_; }
  const Pose2d &follow_goal() const { return follow_goal_; }
  const std::vector<Eigen::Vector3d> &teb_optimize_path() const {
    return teb_planner_.getOptimizePath();
  }

private:
  std::vector<Point2d> ExtractTargetPath(const Point2d &target_pos,
                                         const Point2d &robot_pos);
  void ResetFilters();
  void TryAddPoint(const Point2d &pt);
  void TrimToMaxLength();
  void Smooth(std::vector<Point2d> *pts) const;
  bool ComputeFollowGoal(const std::vector<Point2d> &path,
                         const Point2d &robot_pos);
  static Pose2d ToBodyFrame(const Pose2d &global_pose,
                            const Pose2d &robot_pose);
  static std::vector<Eigen::Vector3d>
  BuildStraightRefPath(const Pose2d &goal_body);

private:
  TargetFollowParams params_;
  TargetPathExtractParams extract_params_;
  planning::config::TebPlannerConf teb_conf_;
  teb_planner::TebPlanner teb_planner_;
  double reference_speed_ = 1.0;

  std::vector<Point2d> points_;
  int consecutive_jumps_{0};
  planning::common::util::MeanFilter filter_x_;
  planning::common::util::MeanFilter filter_y_;

  planning::common::util::MeanFilter speed_filter_;
  double filtered_target_speed_ = 0.0;
  double follow_gap_ = 0.0;

  // 输出/调试
  std::vector<Point2d> target_path_;
  Pose2d follow_goal_;
  std::function<bool(double, double, double)> is_collision_func_;
};

} // namespace algorithm
} // namespace planning
