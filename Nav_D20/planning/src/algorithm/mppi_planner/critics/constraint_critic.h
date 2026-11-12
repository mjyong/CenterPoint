// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#pragma once

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::ConstraintCritic
 * @brief Critic objective function for enforcing feasible constraints
 */
class ConstraintCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

  float getMaxVelConstraint() { return max_vel_; }
  float getMinVelConstraint() { return min_vel_; }

protected:
  unsigned int power_{0};
  float weight_{0};
  float min_vel_{0};
  float max_vel_{0};
};

} // namespace critics
} // namespace mppi
