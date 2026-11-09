#pragma once

#include "src/common/util/macros.h"
#include <Eigen/Core>
#include <chrono>
#include <string>
#include <vector>

namespace planning {
namespace common {
namespace util {

class VelocityCalibration {
public:
  bool Loaded() const { return loaded_; }

  Eigen::Vector3d ComputeCommand(const Eigen::Vector3d &desired,
                                 const Eigen::Vector3d &actual);
  void Reset();
  const double *IntegralState() const { return integral_state_; }

private:
  VelocityCalibration();

  struct Sample {
    double cmd;
    double real;
  };

  double InterpolateInverse(const std::vector<Sample> &table,
                            double desired) const;
  double ComputeIntegral(int idx, double desired, double actual, double dt,
                         double ff_cmd, const std::vector<Sample> *table);

  std::vector<Sample> table_vx_;
  std::vector<Sample> table_vy_;
  std::vector<Sample> table_w_;
  bool loaded_ = false;

  // 积分补偿参数，0/1/2 对应 vx/vy/omega
  struct IntegralParams {
    bool enabled = false;
    double ki = 0.0;
    double i_max = 0.0;
    double error_deadband = 0.0;
  };
  IntegralParams integral_params_[3];
  double integral_state_[3] = {0.0, 0.0, 0.0};
  double leak_rate_ = 0.0;

  std::chrono::steady_clock::time_point last_call_time_;
  bool has_last_call_time_ = false;

  BASE_DECLARE_SINGLETON(VelocityCalibration)
};

} // namespace util
} // namespace common
} // namespace planning
