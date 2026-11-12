// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/goal_critic.h"

#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void GoalCritic::initialize() {
  const auto &c = ctx_.conf->goal_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  threshold_to_consider_ = static_cast<float>(c.threshold_to_consider());
}

void GoalCritic::score(CriticData &data) {
  if (!enabled_ ||
      !utils::withinPositionGoalTolerance(
          threshold_to_consider_, data.state.pose, data.goal)) {
    return;
  }

  const auto goal_x = data.goal.x;
  const auto goal_y = data.goal.y;

  const auto delta_x = data.trajectories.x - goal_x;
  const auto delta_y = data.trajectories.y - goal_y;

  if (power_ > 1u) {
    data.costs += (((delta_x.square() + delta_y.square()).sqrt()).rowwise().mean() *
                   weight_)
                      .pow(power_);
  } else {
    data.costs += (((delta_x.square() + delta_y.square()).sqrt()).rowwise().mean() *
                   weight_)
                      .eval();
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("GoalCritic", mppi::critics::GoalCritic)
