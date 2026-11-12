// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/prefer_forward_critic.h"

#include <Eigen/Dense>

#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void PreferForwardCritic::initialize() {
  const auto &c = ctx_.conf->prefer_forward_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  threshold_to_consider_ = static_cast<float>(c.threshold_to_consider());
}

void PreferForwardCritic::score(CriticData &data) {
  if (!enabled_ ||
      utils::withinPositionGoalTolerance(
          threshold_to_consider_, data.state.pose, data.goal)) {
    return;
  }

  if (power_ > 1u) {
    data.costs += ((data.state.vx.unaryExpr(
                        [&](const float &x) { return std::max(-x, 0.0f); }) *
                    data.model_dt)
                       .rowwise()
                       .sum() *
                   weight_)
                      .pow(power_);
  } else {
    data.costs += ((data.state.vx.unaryExpr(
                        [&](const float &x) { return std::max(-x, 0.0f); }) *
                    data.model_dt)
                       .rowwise()
                       .sum() *
                   weight_);
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("PreferForwardCritic", mppi::critics::PreferForwardCritic)
