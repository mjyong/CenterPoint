// Cruise speed tracking critic: pulls the rollout speed toward a unified
// target profile min(v_ref, sqrt(2 * a * dist_to_goal)) — bilateral tracking
// of the reference velocity far from the goal, decelerating envelope near it.

#pragma once

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::ReferenceVelocityCritic
 * @brief Critic objective function tracking the commanded speed profile:
 *        cruise at the reference velocity, then follow the physically
 *        brakable envelope sqrt(2 * a * dist) down to zero at the goal
 */
class ReferenceVelocityCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  float reference_velocity_{0};
  // Distance at which deceleration toward the goal starts; the envelope
  // deceleration is derived as v_ref^2 / (2 * decel_start_dist)
  float decel_start_dist_{5};
  float weight_{0};
  unsigned int power_{1};
  // Near-start suppression gate (see proto comment): disable the critic while
  // the robot is at the path start with a large heading error, so reorienting
  // in place is not billed as a speed deficit.
  bool near_start_disable_{false};
  int near_start_max_segments_{5};
  float near_start_heading_error_{1.57f};
};

} // namespace critics
} // namespace mppi
