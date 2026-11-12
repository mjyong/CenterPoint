// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#pragma once

#include <string>

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @brief Enum type for different modes of operation
 */
enum class PathAngleMode {
  FORWARD_PREFERENCE = 0,
  NO_DIRECTIONAL_PREFERENCE = 1,
  CONSIDER_FEASIBLE_PATH_ORIENTATIONS = 2
};

/**
 * @brief Method to convert mode enum to string for printing
 */
inline std::string modeToStr(const PathAngleMode &mode) {
  if (mode == PathAngleMode::FORWARD_PREFERENCE) {
    return "Forward Preference";
  } else if (mode == PathAngleMode::CONSIDER_FEASIBLE_PATH_ORIENTATIONS) {
    return "Consider Feasible Path Orientations";
  } else if (mode == PathAngleMode::NO_DIRECTIONAL_PREFERENCE) {
    return "No Directional Preference";
  }
  return "Invalid mode!";
}

/**
 * @class mppi::critics::PathAngleCritic
 * @brief Critic objective function for aligning to path in cases of extreme
 * misalignment or turning
 */
class PathAngleCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  float max_angle_to_furthest_{0};
  float threshold_to_consider_{0};

  size_t offset_from_furthest_{0};
  bool reversing_allowed_{true};
  PathAngleMode mode_{0};

  unsigned int power_{0};
  float weight_{0};
};

} // namespace critics
} // namespace mppi
