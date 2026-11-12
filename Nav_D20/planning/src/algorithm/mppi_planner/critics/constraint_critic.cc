// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/constraint_critic.h"

#include "src/algorithm/mppi_planner/motion_models.h"

namespace mppi {
namespace critics {

void ConstraintCritic::initialize() {
  const auto &c = ctx_.conf->constraint_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());

  const auto &dc = ctx_.conf->dynamic_constraints();
  const float vx_max = static_cast<float>(dc.max_vx());
  const float vy_max = static_cast<float>(dc.max_vy());
  const float vx_min = static_cast<float>(dc.min_vx());
  const float min_sgn = vx_min > 0.0f ? 1.0f : -1.0f;
  max_vel_ = sqrtf(vx_max * vx_max + vy_max * vy_max);
  min_vel_ = min_sgn * sqrtf(vx_min * vx_min + vy_max * vy_max);
}

void ConstraintCritic::score(CriticData &data) {
  if (!enabled_) {
    return;
  }

  // Differential motion model
  auto diff = dynamic_cast<DiffDriveMotionModel *>(data.motion_model.get());
  if (diff != nullptr) {
    if (power_ > 1u) {
      data.costs += (((((data.state.vx - max_vel_).max(0.0f) +
                        (min_vel_ - data.state.vx).max(0.0f)) *
                       data.model_dt)
                          .rowwise()
                          .sum()
                          .eval()) *
                     weight_)
                        .pow(power_)
                        .eval();
    } else {
      data.costs += (((((data.state.vx - max_vel_).max(0.0f) +
                        (min_vel_ - data.state.vx).max(0.0f)) *
                       data.model_dt)
                          .rowwise()
                          .sum()
                          .eval()) *
                     weight_)
                        .eval();
    }
    return;
  }

  // Omnidirectional motion model
  auto omni = dynamic_cast<OmniMotionModel *>(data.motion_model.get());
  if (omni != nullptr) {
    auto &vx = data.state.vx;
    unsigned int n_rows = data.state.vx.rows();
    unsigned int n_cols = data.state.vx.cols();
    Eigen::ArrayXXf sgn(n_rows, n_cols);
    sgn = vx.unaryExpr([](const float x) { return copysignf(1.0f, x); });

    auto vel_total = sgn * (data.state.vx.square() + data.state.vy.square()).sqrt();
    if (power_ > 1u) {
      data.costs += ((((vel_total - max_vel_).max(0.0f) +
                       (min_vel_ - vel_total).max(0.0f)) *
                      data.model_dt)
                         .rowwise()
                         .sum()
                         .eval() *
                     weight_)
                        .pow(power_)
                        .eval();
    } else {
      data.costs += ((((vel_total - max_vel_).max(0.0f) +
                       (min_vel_ - vel_total).max(0.0f)) *
                      data.model_dt)
                         .rowwise()
                         .sum()
                         .eval() *
                     weight_)
                        .eval();
    }
    return;
  }

  // Ackermann motion model
  auto acker = dynamic_cast<AckermannMotionModel *>(data.motion_model.get());
  if (acker != nullptr) {
    auto &vx = data.state.vx;
    auto &wz = data.state.wz;
    float min_turning_rad = acker->getMinTurningRadius();
    auto out_of_turning_rad_motion =
        (min_turning_rad - (vx.abs() / wz.abs())).max(0.0f);
    if (power_ > 1u) {
      data.costs += ((((vx - max_vel_).max(0.0f) + (min_vel_ - vx).max(0.0f) +
                       out_of_turning_rad_motion) *
                      data.model_dt)
                         .rowwise()
                         .sum()
                         .eval() *
                     weight_)
                        .pow(power_)
                        .eval();
    } else {
      data.costs += ((((vx - max_vel_).max(0.0f) + (min_vel_ - vx).max(0.0f) +
                       out_of_turning_rad_motion) *
                      data.model_dt)
                         .rowwise()
                         .sum()
                         .eval() *
                     weight_)
                        .eval();
    }
    return;
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("ConstraintCritic", mppi::critics::ConstraintCritic)
