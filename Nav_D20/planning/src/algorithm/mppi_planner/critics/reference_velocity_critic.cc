#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/reference_velocity_critic.h"

#include "src/algorithm/mppi_planner/tools/utils.h"

namespace mppi {
namespace critics {

void ReferenceVelocityCritic::initialize() {
  const auto &c = ctx_.conf->reference_velocity_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  weight_ = static_cast<float>(c.weight());
  decel_start_dist_ = static_cast<float>(c.decel_start_dist());
  reference_velocity_ = static_cast<float>(c.reference_velocity());
  near_start_disable_ = c.near_start_disable();
  near_start_max_segments_ = c.near_start_max_segments();
  near_start_heading_error_ =
      static_cast<float>(c.near_start_heading_error());
}

void ReferenceVelocityCritic::score(CriticData &data) {
  if (!enabled_) {
    return;
  }

  // Near-start suppression: at the path start with a large heading error the
  // robot must reorient first; the alignment payoff lies beyond the horizon,
  // so the speed deficit billing would only block the rotation. Gate off
  // until the robot has advanced past near_start_max_segments segments OR
  // roughly faces the path start orientation.
  if (near_start_disable_ && data.path.x.size() >= 2) {
    utils::setPathFurthestPointIfNotSet(data);
    if (static_cast<int>(*data.furthest_reached_path_point) <
        near_start_max_segments_) {
      const float heading_err = fabs(utils::shortest_angular_distance(
          data.path.yaws(0), data.state.pose.yaw));
      if (heading_err > near_start_heading_error_) {
        return;
      }
    }
  }

  // Externally injected reference speed (via MppiPlanner::set_target_speed)
  // takes precedence over the configured value when positive
  float reference_velocity = reference_velocity_;
  if (data.ctx != nullptr &&
      data.ctx->runtime_reference_velocity > 0.0) {
    reference_velocity =
        static_cast<float>(data.ctx->runtime_reference_velocity);
  }

  // Clamp the effective reference to the reachable speed range: an oversized
  // externally injected max_speed flattens the decel envelope (it only bends
  // below vx_max very close to the goal), degenerating the critic into a
  // constant-speed tracker with no terminal deceleration.
  const float vx_max = static_cast<float>(
      ctx_.conf->dynamic_constraints().max_vx());
  if (vx_max > 0.0f && reference_velocity > vx_max) {
    reference_velocity = vx_max;
  }

  // Feature is disabled when no reference velocity is available
  if (reference_velocity <= 0.0f) {
    return;
  }

  // Planar speed: |vx| for non-holonomic, hypot(vx, vy) for omni
  Eigen::ArrayXXf speed = data.state.vx.abs();
  if (data.motion_model->isHolonomic()) {
    speed = (data.state.vx.square() + data.state.vy.square()).sqrt();
  }

  // Remaining distance to the goal per rollout step, reduced to its running
  // minimum along each trajectory so the envelope is monotonically
  // non-increasing. With the raw distance, a trajectory that overshoots the
  // goal sees dist_rem grow again past the crossing point (a "donut"
  // profile), so the critic rewards driving away and keeps re-accelerating.
  // With the running min the target stays ~0 beyond the closest approach,
  // so any move away from the goal is penalized instead.
  Eigen::ArrayXXf dist_env =
      ((data.goal.x - data.trajectories.x).square() +
       (data.goal.y - data.trajectories.y).square())
          .sqrt()
          .eval();
  for (Eigen::Index t = 1; t < dist_env.cols(); ++t) {
    dist_env.col(t) = dist_env.col(t).min(dist_env.col(t - 1));
  }

  // Unified target profile: cruise at the reference velocity, then follow
  // the physically brakable envelope down to zero. The envelope deceleration
  // is derived as a = v_ref^2 / (2 * decel_start_dist) so the envelope
  // equals the reference exactly at decel_start_dist — deceleration starts
  // there and the profile is seamless without any threshold branching:
  //   target = v_ref * sqrt(min(dist_env / decel_start_dist, 1))
  //          = min(v_ref, sqrt(2 * a * dist_env))
  // NOTE: `target` must be a concrete array, not `auto`: assigning to a
  // plain ArrayXXf evaluates the expression immediately. An `auto` binding
  // like `v_ref * (...).sqrt().eval()` keeps a dangling reference to the
  // eval() temporary (CwiseUnaryOp stores nested expressions by reference),
  // which crashed with std::bad_alloc downstream.
  Eigen::ArrayXXf target =
      reference_velocity * (dist_env / decel_start_dist_).min(1.0f).sqrt();

  // Profile tracking, integrated over the horizon like ConstraintCritic so
  // the weight scale is comparable. Inside the decel zone (dist_env below
  // decel_start_dist) the penalty is one-sided: only speeds above the
  // envelope are billed, so braking harder than the envelope is free — the
  // optimizer can always pick a firmer stop, fixing the structural braking
  // deficit near the goal. While cruising (dist_env at/above the decel
  // start) the tracking stays bilateral so the cruise speed converges to
  // the reference instead of drifting to the constraint cap or to zero.
  const Eigen::ArrayXXf dev = (speed - target).eval();
  const Eigen::ArrayXXf penalty =
      (dist_env < decel_start_dist_).select(dev.cwiseMax(0.0f), dev.abs());
  auto deficit = (penalty * data.model_dt)
                     .rowwise()
                     .sum()
                     .eval();
  if (power_ > 1u) {
    data.costs += (deficit * weight_).pow(power_).eval();
  } else {
    data.costs += (deficit * weight_).eval();
  }
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("ReferenceVelocityCritic", mppi::critics::ReferenceVelocityCritic)
