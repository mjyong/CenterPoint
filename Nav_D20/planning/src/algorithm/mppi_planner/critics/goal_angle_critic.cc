// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/goal_angle_critic.h"

#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void GoalAngleCritic::initialize() {
  const auto &c = ctx_.conf->goal_angle_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  threshold_to_consider_ = static_cast<float>(c.threshold_to_consider());
}

void GoalAngleCritic::score(CriticData &data) {
  if (!enabled_ ||
      !utils::withinPositionGoalTolerance(
          threshold_to_consider_, data.state.pose, data.goal)) {
    return;
  }

  const auto goal_idx = data.path.x.size() - 1;
  const float goal_yaw = data.path.yaws(goal_idx);

  if (power_ > 1u) {
    data.costs +=
        ((utils::shortest_angular_distance(data.trajectories.yaws, goal_yaw).abs())
             .rowwise()
             .mean() *
         weight_)
            .pow(power_)
            .eval();
  } else {
    data.costs +=
        ((utils::shortest_angular_distance(data.trajectories.yaws, goal_yaw).abs())
             .rowwise()
             .mean() *
         weight_)
            .eval();
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("GoalAngleCritic", mppi::critics::GoalAngleCritic)
