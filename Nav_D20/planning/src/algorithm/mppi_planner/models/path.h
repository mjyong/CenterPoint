// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov)

#pragma once

#include <Eigen/Dense>

namespace mppi {
namespace models {

struct Path {
  Eigen::ArrayXf x;
  Eigen::ArrayXf y;
  Eigen::ArrayXf yaws;

  void reset(unsigned int size) {
    x.setZero(size);
    y.setZero(size);
    yaws.setZero(size);
  }
};

} // namespace models
} // namespace mppi
