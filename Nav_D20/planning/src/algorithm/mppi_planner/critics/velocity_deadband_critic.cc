// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/velocity_deadband_critic.h"

namespace mppi {
namespace critics {

void VelocityDeadbandCritic::initialize() {
  const auto &c = ctx_.conf->velocity_deadband_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  deadband_velocities_ = {static_cast<float>(c.vx_deadband()),
                          static_cast<float>(c.vy_deadband()),
                          static_cast<float>(c.wz_deadband())};
}

void VelocityDeadbandCritic::score(CriticData &data) {
  if (!enabled_) {
    return;
  }

  if (data.motion_model->isHolonomic()) {
    if (power_ > 1u) {
      data.costs += ((((fabs(deadband_velocities_[0]) - data.state.vx.abs()).max(0.0f) +
                       (fabs(deadband_velocities_[1]) - data.state.vy.abs()).max(0.0f) +
                       (fabs(deadband_velocities_[2]) - data.state.wz.abs()).max(0.0f)) *
                      data.model_dt)
                         .rowwise()
                         .sum() *
                     weight_)
                        .pow(power_)
                        .eval();
    } else {
      data.costs += ((((fabs(deadband_velocities_[0]) - data.state.vx.abs()).max(0.0f) +
                       (fabs(deadband_velocities_[1]) - data.state.vy.abs()).max(0.0f) +
                       (fabs(deadband_velocities_[2]) - data.state.wz.abs()).max(0.0f)) *
                      data.model_dt)
                         .rowwise()
                         .sum() *
                     weight_)
                        .eval();
    }
    return;
  }

  if (power_ > 1u) {
    data.costs += ((((fabs(deadband_velocities_[0]) - data.state.vx.abs()).max(0.0f) +
                     (fabs(deadband_velocities_[2]) - data.state.wz.abs()).max(0.0f)) *
                    data.model_dt)
                       .rowwise()
                       .sum() *
                   weight_)
                      .pow(power_)
                      .eval();
  } else {
    data.costs += ((((fabs(deadband_velocities_[0]) - data.state.vx.abs()).max(0.0f) +
                     (fabs(deadband_velocities_[2]) - data.state.wz.abs()).max(0.0f)) *
                    data.model_dt)
                       .rowwise()
                       .sum() *
                   weight_)
                      .eval();
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("VelocityDeadbandCritic", mppi::critics::VelocityDeadbandCritic)
