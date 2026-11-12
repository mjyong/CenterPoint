// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#pragma once

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::PathAlignCritic
 * @brief Critic objective function for aligning to the path
 */
class PathAlignCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  size_t offset_from_furthest_{0};
  int trajectory_point_step_{0};
  float threshold_to_consider_{0};
  float max_path_occupancy_ratio_{0};
  bool use_path_orientations_{false};
  unsigned int power_{0};
  float weight_{0};
};

} // namespace critics
} // namespace mppi
