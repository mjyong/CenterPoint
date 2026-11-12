// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov)

#pragma once

#include <Eigen/Dense>

namespace mppi {
namespace models {

struct Trajectories {
  Eigen::ArrayXXf x;
  Eigen::ArrayXXf y;
  Eigen::ArrayXXf yaws;

  void reset(unsigned int batch_size, unsigned int time_steps) {
    x.setZero(batch_size, time_steps);
    y.setZero(batch_size, time_steps);
    yaws.setZero(batch_size, time_steps);
  }
};

} // namespace models
} // namespace mppi
