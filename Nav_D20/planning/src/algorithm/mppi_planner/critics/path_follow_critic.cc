// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/path_follow_critic.h"

#include <Eigen/Dense>

#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void PathFollowCritic::initialize() {
  const auto &c = ctx_.conf->path_follow_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  threshold_to_consider_ = static_cast<float>(c.threshold_to_consider());
  offset_from_furthest_ = static_cast<size_t>(c.offset_from_furthest());
}

void PathFollowCritic::score(CriticData &data) {
  if (!enabled_ || data.path.x.size() < 2 ||
      utils::withinPositionGoalTolerance(
          threshold_to_consider_, data.state.pose, data.goal)) {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  utils::setPathCostsIfNotSet(data);
  const size_t path_size = data.path.x.size() - 1;

  auto offsetted_idx = std::min(
      *data.furthest_reached_path_point + offset_from_furthest_, path_size);

  // Drive to the first valid path point, in case of dynamic obstacles on path
  // we want to drive past it, not through it
  bool valid = false;
  while (!valid && offsetted_idx < path_size - 1) {
    valid = (*data.path_pts_valid)[offsetted_idx];
    if (!valid) {
      offsetted_idx++;
    }
  }

  const auto path_x = data.path.x(offsetted_idx);
  const auto path_y = data.path.y(offsetted_idx);

  const int &&rightmost_idx = data.trajectories.x.cols() - 1;
  const auto last_x = data.trajectories.x.col(rightmost_idx);
  const auto last_y = data.trajectories.y.col(rightmost_idx);

  const auto delta_x = last_x - path_x;
  const auto delta_y = last_y - path_y;
  if (power_ > 1u) {
    data.costs += (((delta_x.square() + delta_y.square()).sqrt()) * weight_).pow(power_);
  } else {
    data.costs += ((delta_x.square() + delta_y.square()).sqrt()) * weight_;
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("PathFollowCritic", mppi::critics::PathFollowCritic)
