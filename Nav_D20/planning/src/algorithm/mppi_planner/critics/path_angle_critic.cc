// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/path_angle_critic.h"

#include <cmath>

#include "src/common/util/log.h"
#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void PathAngleCritic::initialize() {
  const auto &c = ctx_.conf->path_angle_critic();
  enabled_ = c.enabled();
  offset_from_furthest_ = static_cast<size_t>(c.offset_from_furthest());
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  threshold_to_consider_ = static_cast<float>(c.threshold_to_consider());
  max_angle_to_furthest_ = static_cast<float>(c.max_angle_to_furthest());

  const float vx_min = static_cast<float>(ctx_.conf->dynamic_constraints().min_vx());
  if (fabs(vx_min) < 1e-6f) { // zero
    reversing_allowed_ = false;
  } else if (vx_min < 0.0f) { // reversing possible
    reversing_allowed_ = true;
  }

  mode_ = static_cast<PathAngleMode>(c.mode());
  if (!reversing_allowed_ && mode_ == PathAngleMode::NO_DIRECTIONAL_PREFERENCE) {
    mode_ = PathAngleMode::FORWARD_PREFERENCE;
    AWARN << "Path angle mode set to no directional preference, but settings "
             "don't allow for reversing! Setting mode to forward preference.";
  }

  AINFO << "PathAngleCritic initialized with mode: " << modeToStr(mode_);
}

void PathAngleCritic::score(CriticData &data) {
  if (!enabled_ ||
      utils::withinPositionGoalTolerance(
          threshold_to_consider_, data.state.pose, data.goal)) {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  auto offsetted_idx = std::min(
      *data.furthest_reached_path_point + offset_from_furthest_,
      static_cast<size_t>(data.path.x.size()) - 1);

  const float goal_x = data.path.x(offsetted_idx);
  const float goal_y = data.path.y(offsetted_idx);
  const float goal_yaw = data.path.yaws(offsetted_idx);
  const models::Pose &pose = data.state.pose;

  switch (mode_) {
  case PathAngleMode::FORWARD_PREFERENCE:
    if (utils::posePointAngle(pose, goal_x, goal_y, true) < max_angle_to_furthest_) {
      return;
    }
    break;
  case PathAngleMode::NO_DIRECTIONAL_PREFERENCE:
    if (utils::posePointAngle(pose, goal_x, goal_y, false) < max_angle_to_furthest_) {
      return;
    }
    break;
  case PathAngleMode::CONSIDER_FEASIBLE_PATH_ORIENTATIONS:
    if (utils::posePointAngle(pose, goal_x, goal_y, goal_yaw) <
        max_angle_to_furthest_) {
      return;
    }
    break;
  }

  int last_idx = data.trajectories.y.cols() - 1;
  auto diff_y = goal_y - data.trajectories.y.col(last_idx);
  auto diff_x = goal_x - data.trajectories.x.col(last_idx);
  auto yaws_between_points = diff_y
                                 .binaryExpr(diff_x, [&](const float &y, const float &x) {
                                   return atan2f(y, x);
                                 })
                                 .eval();

  switch (mode_) {
  case PathAngleMode::FORWARD_PREFERENCE: {
    auto last_yaws = data.trajectories.yaws.col(last_idx);
    auto yaws = utils::shortest_angular_distance(last_yaws, yaws_between_points).abs();
    if (power_ > 1u) {
      data.costs += (yaws * weight_).pow(power_);
    } else {
      data.costs += yaws * weight_;
    }
    return;
  }
  case PathAngleMode::NO_DIRECTIONAL_PREFERENCE: {
    auto last_yaws = data.trajectories.yaws.col(last_idx);
    auto yaws_between_points_corrected =
        utils::normalize_yaws_between_points(last_yaws, yaws_between_points);
    auto corrected_yaws =
        utils::shortest_angular_distance(last_yaws, yaws_between_points_corrected).abs();
    if (power_ > 1u) {
      data.costs += (corrected_yaws * weight_).pow(power_);
    } else {
      data.costs += corrected_yaws * weight_;
    }
    return;
  }
  case PathAngleMode::CONSIDER_FEASIBLE_PATH_ORIENTATIONS: {
    auto last_yaws = data.trajectories.yaws.col(last_idx);
    auto yaws_between_points_corrected =
        utils::normalize_yaws_between_points(goal_yaw, yaws_between_points);
    auto corrected_yaws =
        utils::shortest_angular_distance(last_yaws, yaws_between_points_corrected).abs();
    if (power_ > 1u) {
      data.costs += (corrected_yaws * weight_).pow(power_);
    } else {
      data.costs += corrected_yaws * weight_;
    }
    return;
  }
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("PathAngleCritic", mppi::critics::PathAngleCritic)
