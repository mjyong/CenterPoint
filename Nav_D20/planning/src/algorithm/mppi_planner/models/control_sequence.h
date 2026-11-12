// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov)

#pragma once

#include <Eigen/Dense>

namespace mppi {
namespace models {

struct Control {
  float vx, vy, wz;
};

struct ControlSequence {
  Eigen::ArrayXf vx;
  Eigen::ArrayXf vy;
  Eigen::ArrayXf wz;

  void reset(unsigned int time_steps) {
    vx.setZero(time_steps);
    vy.setZero(time_steps);
    wz.setZero(time_steps);
  }
};

} // namespace models
} // namespace mppi
