// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#pragma once

#include <array>

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::VelocityDeadbandCritic
 * @brief Critic objective function for penalizing velocities below the deadband
 */
class VelocityDeadbandCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  std::array<float, 3> deadband_velocities_{{0.0f, 0.0f, 0.0f}};
  unsigned int power_{0};
  float weight_{0};
};

} // namespace critics
} // namespace mppi
