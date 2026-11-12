// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). geometry_msgs types are replaced by POD.

#pragma once

#include <Eigen/Dense>

namespace mppi {
namespace models {

// Robot pose in the working (robot/base_link) frame.
struct Pose {
  float x{0.0f};
  float y{0.0f};
  float yaw{0.0f};
};

// Robot twist in the working (robot/base_link) frame.
struct Velocity {
  float vx{0.0f};
  float vy{0.0f};
  float wz{0.0f};
};

struct State {
  Eigen::ArrayXXf vx;
  Eigen::ArrayXXf vy;
  Eigen::ArrayXXf wz;

  Eigen::ArrayXXf cvx;
  Eigen::ArrayXXf cvy;
  Eigen::ArrayXXf cwz;

  Pose pose;
  Velocity speed;

  void reset(unsigned int batch_size, unsigned int time_steps) {
    vx.setZero(batch_size, time_steps);
    vy.setZero(batch_size, time_steps);
    wz.setZero(batch_size, time_steps);

    cvx.setZero(batch_size, time_steps);
    cvy.setZero(batch_size, time_steps);
    cwz.setZero(batch_size, time_steps);
  }
};

} // namespace models
} // namespace mppi
