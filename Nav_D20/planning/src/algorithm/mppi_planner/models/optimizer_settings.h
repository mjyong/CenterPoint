// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov)

#pragma once

#include "src/algorithm/mppi_planner/models/constraints.h"

namespace mppi {
namespace models {

struct OptimizerSettings {
  models::ControlConstraints base_constraints{
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  models::ControlConstraints constraints{
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  models::SamplingStd sampling_std{0.0f, 0.0f, 0.0f};
  float model_dt{0.0f};
  float temperature{0.0f};
  float gamma{0.0f};
  unsigned int batch_size{0u};
  unsigned int time_steps{0u};
  unsigned int iteration_count{0u};
  bool shift_control_sequence{false};
  size_t retry_attempt_limit{0};
};

} // namespace models
} // namespace mppi
