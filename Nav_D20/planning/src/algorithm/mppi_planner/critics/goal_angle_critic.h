// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#pragma once

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::GoalAngleCritic
 * @brief Critic objective function for driving towards goal orientation
 */
class GoalAngleCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  float threshold_to_consider_{0};
  unsigned int power_{0};
  float weight_{0};
};

} // namespace critics
} // namespace mppi
