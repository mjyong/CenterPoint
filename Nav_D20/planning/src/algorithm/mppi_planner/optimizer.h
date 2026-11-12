// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). ROS types are replaced by the POD
// mppi::models types; fallback throws are replaced by a bool return value.

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <string>

#include "mppi_planner_conf.pb.h"
#include "src/algorithm/mppi_planner/critic_data.h"
#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/models/control_sequence.h"
#include "src/algorithm/mppi_planner/models/optimizer_settings.h"
#include "src/algorithm/mppi_planner/models/path.h"
#include "src/algorithm/mppi_planner/models/state.h"
#include "src/algorithm/mppi_planner/models/trajectories.h"
#include "src/algorithm/mppi_planner/motion_models.h"
#include "src/algorithm/mppi_planner/tools/noise_generator.h"
#include "src/algorithm/mppi_cuda_planner/mppi_cuda_backend.h"

namespace mppi {

/**
 * @class mppi::Optimizer
 * @brief Main algorithm optimizer of the MPPI Controller
 */
class Optimizer {
public:
  Optimizer() = default;
  ~Optimizer() { shutdown(); }

  /**
   * @brief Initializes optimizer on startup
   * @param conf Planner configuration (must outlive the optimizer)
   * @param ctx Shared critic context (costmap callbacks)
   */
  void initialize(const planning::config::MppiPlannerConf &conf,
                  const CriticContext &ctx);

  /**
   * @brief Shutdown for optimizer at process end
   */
  void shutdown();

  /**
   * @brief Compute control using MPPI algorithm
   * @param pose Pose of the robot at given time (robot frame)
   * @param speed Speed of the robot at given time (robot frame)
   * @param plan Path plan to track (robot frame)
   * @param goal Goal pose to reach (robot frame)
   * @return false if the optimizer failed to find a collision free solution
   */
  bool evalControl(
      const models::Pose &pose, const models::Velocity &speed,
      const models::Path &plan, const models::Pose &goal);

  /**
   * @brief First control of the (post-shift) optimal control sequence
   */
  models::Control getControlFromSequence() const;

  /**
   * @brief Access the optimal control sequence after evalControl (already
   * shifted if enabled, so index 0 corresponds to the next cycle)
   */
  const models::ControlSequence &controlSequence() const {
    return control_sequence_;
  }

  /**
   * @brief Get the optimal trajectory for a cycle for visualization
   */
  Eigen::ArrayXXf getOptimizedTrajectory();

  /**
   * @brief Reset the optimization problem to initial conditions
   */
  void reset();

  /**
   * @brief Light per-cycle reset: restore configured constraints and clear
   * costs, but keep the control sequence (warm start) and the regenerated
   * noises. The full reset() is reserved for fallback / re-initialization.
   */
  void softReset();

  /**
   * @brief Enable/disable control sequence shifting (e.g. when the control
   * period does not match model_dt)
   */
  void setShiftControlSequence(bool enable) {
    settings_.shift_control_sequence = enable;
  }

  /**
   * @brief Inject an externally commanded reference speed for the
   * ReferenceVelocityCritic (0 = fall back to the configured value)
   */
  void setRuntimeReferenceVelocity(double v) {
    ctx_.runtime_reference_velocity = v;
  }

  /**
   * @brief Attach a CUDA backend that offloads rollout propagation and
   * critic scoring (may be nullptr for the pure CPU path). Must be called
   * after initialize(). On the first backend failure the optimizer falls
   * back to the CPU path permanently.
   */
  void setCudaBackend(cuda::MppiCudaBackend *backend) {
    cuda_backend_ = backend;
  }

  /**
   * @brief Provide the device-side obstacle distance field (with its grid
   * geometry) consumed by the CUDA backend each cycle. Returns an empty field
   * when the costmap CUDA pipeline is unavailable, which makes evalStep fail
   * and the optimizer fall back to the CPU path.
   */
  void setCudaMapFieldProvider(
      std::function<cuda::MppiCudaMapField()> provider) {
    cuda_map_field_provider_ = std::move(provider);
  }

  const models::OptimizerSettings &settings() const { return settings_; }

protected:
  /**
   * @brief Main function to generate, score, and return trajectories
   */
  void optimize();

  /**
   * @brief Run one rollout + scoring step on the GPU backend. Uploads the
   * CPU-noised controls, downloads trajectories, state and costs. Returns
   * false when no backend is attached or the GPU step failed (caller runs
   * the CPU path instead).
   */
  bool runCudaStep();

  /**
   * @brief Prepare state information on new request for trajectory rollouts
   */
  void prepare(
      const models::Pose &pose, const models::Velocity &speed,
      const models::Path &plan, const models::Pose &goal);

  /**
   * @brief Set the motion model of the vehicle platform
   */
  void setMotionModel();

  /**
   * @brief Shift the optimal control sequence after processing for
   * next iterations initial conditions after execution
   */
  void shiftControlSequence();

  /**
   * @brief updates generated trajectories with noised trajectories
   * from the last cycle's optimal control
   */
  void generateNoisedTrajectories();

  /**
   * @brief Apply hard vehicle constraints on control sequence
   */
  void applyControlSequenceConstraints();

  /**
   * @brief Update velocities in state
   */
  void updateStateVelocities(models::State &state) const;

  /**
   * @brief Update initial velocity in state
   */
  void updateInitialStateVelocities(models::State &state) const;

  /**
   * @brief predict velocities in state using model
   */
  void propagateStateVelocitiesFromInitials(models::State &state) const;

  /**
   * @brief Rollout velocities in state to poses (batched)
   */
  void integrateStateVelocities(
      models::Trajectories &trajectories, const models::State &state) const;

  /**
   * @brief Update control sequence with state controls weighted by costs
   * using softmax function
   */
  void updateControlSequence();

  /**
   * @brief Whether the motion model is holonomic
   */
  bool isHolonomic() const;

  /**
   * @brief Perform fallback behavior to try to recover from a set of
   * trajectories in collision
   * @param fail Whether the system failed to recover from
   * @return true if another attempt should be made
   */
  bool fallback(bool fail);

  /**
   * @brief Roll out the final (filtered and shifted) control sequence as a
   * single trajectory, score it with every critic, log the per-critic costs
   * and record them via DebugDatas for PlotJuggler visualization
   */
  void logFinalTrajectoryCriticCosts();

protected:
  const planning::config::MppiPlannerConf *conf_{nullptr};
  CriticContext ctx_;
  std::shared_ptr<MotionModel> motion_model_;

  CriticManager critic_manager_;
  NoiseGenerator noise_generator_;

  models::OptimizerSettings settings_;

  models::State state_;
  models::ControlSequence control_sequence_;
  std::array<mppi::models::Control, 4> control_history_;
  models::Trajectories generated_trajectories_;
  models::Path path_;
  models::Pose goal_;
  Eigen::ArrayXf costs_;
  size_t fallback_counter_{0};

  CriticData critics_data_ = {
      state_, generated_trajectories_, path_, goal_,
      costs_, settings_.model_dt, false, nullptr, std::nullopt, std::nullopt,
      nullptr}; /// Caution, keep references

  // Optional CUDA hot-path backend (owned by the MppiPlanner facade).
  cuda::MppiCudaBackend *cuda_backend_{nullptr};
  // Per-cycle device distance field supplied by the preprocess CUDA pipeline.
  std::function<cuda::MppiCudaMapField()> cuda_map_field_provider_;
  // Sticky: set on the first GPU step failure; all later cycles use the CPU
  // path so behavior never oscillates between backends.
  bool cuda_failed_{false};
  // MPPI_CUDA_DEBUG_COMPARE env: run the CPU rollout + critics every cycle
  // and log the divergence against the GPU result.
  bool cuda_debug_compare_{false};
};

} // namespace mppi
