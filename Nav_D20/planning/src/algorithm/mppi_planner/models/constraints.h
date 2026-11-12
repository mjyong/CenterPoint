// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov)

#pragma once

namespace mppi {
namespace models {

struct ControlConstraints {
  float vx_max;
  float vx_min;
  float vy;
  float wz;
  float ax_max;
  float ax_min;
  float ay_max;
  float az_max;
};

struct SamplingStd {
  float vx;
  float vy;
  float wz;
};

} // namespace models
} // namespace mppi
