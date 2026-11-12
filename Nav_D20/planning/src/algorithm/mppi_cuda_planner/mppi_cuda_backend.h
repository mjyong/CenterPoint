// Host-side orchestration of the CUDA MPPI hot path: noised-control rollout
// propagation (Optimizer::generateNoisedTrajectories) and fused critic
// scoring (CriticManager::evalTrajectoriesScores) run on the GPU. Softmax,
// control-sequence update, Savitzky-Golay filtering and shifting stay on the
// CPU, shared with the CPU path.
//
// Layout convention: all batch x time_steps matrices are Eigen column-major
// and the device buffers mirror that layout, so host<->device transfers are
// plain contiguous copies and kernels index (j, t) as t * batch + j.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mppi_planner_conf.pb.h"
#include "src/algorithm/mppi_planner/critic_data.h"
#include "src/algorithm/mppi_planner/models/optimizer_settings.h"

namespace mppi {
namespace cuda {

struct MppiDeviceParams;
struct MppiCycleParams;

// Obstacle distance field (device memory owned by the cost-map CUDA
// pipeline). Column-major linearization mirroring the grid_map layer matrix:
// linear = j * rows + i, with i the x-buffer index (x decreases as i grows)
// and j the y-buffer index. Geometry values are doubles because the CPU
// sampling path (GridMap::atPosition INTER_LINEAR) computes in double.
struct MppiCudaMapField {
  const float *distance_field{nullptr};
  int rows{0};
  int cols{0};
  double resolution{0.0};
  double map_position_x{0.0};
  double map_position_y{0.0};
  double length_x{0.0};
  double length_y{0.0};
};

#ifdef PLANNING_ENABLE_CUDA

class MppiCudaBackend {
public:
  MppiCudaBackend() = default;
  ~MppiCudaBackend();

  MppiCudaBackend(const MppiCudaBackend &) = delete;
  MppiCudaBackend &operator=(const MppiCudaBackend &) = delete;

  /**
   * @brief Allocate device memory and upload static parameters.
   * @return false when CUDA is unavailable, the motion model has no GPU
   *         implementation (Ackermann), any enabled critic is not supported
   *         on the GPU, or a device allocation fails. The backend stays
   *         unusable and the planner uses the CPU path.
   */
  bool init(const planning::config::MppiPlannerConf &conf,
            const models::OptimizerSettings &settings,
            const CriticContext &ctx);

  void release();

  /**
   * @brief Execute one optimization step on the GPU.
   *
   * Uploads the CPU-overlaid noised controls (state.cvx/cvy/cwz) and the
   * per-cycle inputs, runs the rollout and fused critic kernels, then
   * downloads the propagated velocities, clamped controls, integrated
   * trajectories and per-trajectory costs. On success:
   *   - state.vx/vy/wz hold the propagated rollout velocities,
   *   - state.cvx/cvy/cwz hold the acceleration-clamped controls (matching
   *     MotionModel::predict write-back semantics),
   *   - trajectories.x/y/yaws hold the integrated poses,
   *   - costs accumulates the critic scores (+=, like the CPU critics),
   *   - fail_flag is set when every rollout is in collision.
   *
   * @return false on any error; the caller must fall back to the CPU path
   *         for this and all subsequent cycles (sticky fallback).
   */
  bool evalStep(models::State &state, models::Trajectories &trajectories,
                const models::Path &path, const models::Pose &goal,
                const CriticContext &ctx, const MppiCudaMapField &map,
                Eigen::ArrayXf &costs, bool &fail_flag);

  bool ready() const { return ready_; }

  /**
   * @brief Debug helper: download the raw per-critic cost components from the
   * last evalStep (11 * batch floats, slot-major). Used by the
   * MPPI_CUDA_DEBUG_COMPARE diagnostics; returns false on a transfer error.
   */
  bool downloadDebugComponents(std::vector<float> &out);

  /**
   * @brief Debug helper: configured weight of a critic slot (DeviceCriticId
   * order), for the MPPI_CUDA_DEBUG_COMPARE diagnostics.
   */
  float criticWeight(int slot) const { return critic_weight_[slot]; }

private:
  bool checkCriticSupport(const planning::config::MppiPlannerConf &conf);

  bool ready_{false};

  // Problem dimensions (fixed at init).
  int batch_size_{0};
  int time_steps_{0};
  bool holonomic_{false};

  // Host-side mirrors needed for the per-cycle gating logic (critic enabled
  // flags in DeviceCriticId order, the reversing-adjusted path-angle mode and
  // the configured reference velocity). Array size = kNumDeviceCritics (11:
  // nine original slots + Twirling + VelocityDeadband).
  bool critic_enabled_[11] = {false, false, false, false, false, false,
                              false, false, false, false, false};
  int path_angle_mode_{0};
  float reference_velocity_config_{0.0f};
  // Configured weight per critic slot, mirrored at init for the
  // MPPI_CUDA_DEBUG_COMPARE per-critic divergence check.
  float critic_weight_[11] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Device buffers (nullptr until init succeeds).
  MppiDeviceParams *d_params_{nullptr}; // static parameters (uploaded once)
  MppiCycleParams *d_cycle_{nullptr};   // per-step parameters
  float *d_cvx_{nullptr};        // batch x T noised controls (in, clamped out)
  float *d_cvy_{nullptr};
  float *d_cwz_{nullptr};
  float *d_vx_{nullptr};         // batch x T propagated velocities (out)
  float *d_vy_{nullptr};
  float *d_wz_{nullptr};
  float *d_x_{nullptr};          // batch x T integrated poses (out)
  float *d_y_{nullptr};
  float *d_yaws_{nullptr};
  float *d_costs_{nullptr};      // batch costs (in/out, += semantics)
  unsigned char *d_collision_{nullptr};  // batch collision flags (out)
  float *d_path_{nullptr};       // 4 * kMaxPathPoints: x, y, yaw, arclen
  unsigned char *d_path_valid_{nullptr}; // kMaxPathPoints validity flags
  float *d_components_{nullptr}; // 11 * batch raw critic components
  float *d_repulsive_{nullptr};  // batch repulsive costs (obstacles critic)
  float *d_min_rep_{nullptr};    // batch min of the repulsive cost
  int *d_all_collide_{nullptr};  // 1 when every rollout collides
  int *d_fail_{nullptr};         // fail flag written by the finalize kernel
};

#else // PLANNING_ENABLE_CUDA

// CPU-only build stub: init always fails so the planner transparently uses
// the CPU path, even if enable_cuda is mistakenly left on in the config.
class MppiCudaBackend {
public:
  MppiCudaBackend() = default;
  ~MppiCudaBackend() = default;

  bool init(const planning::config::MppiPlannerConf &,
            const models::OptimizerSettings &, const CriticContext &) {
    return false;
  }
  void release() {}

  bool evalStep(models::State &, models::Trajectories &,
                const models::Path &, const models::Pose &,
                const CriticContext &, const MppiCudaMapField &,
                Eigen::ArrayXf &, bool &) {
    return false;
  }

  bool ready() const { return false; }

  bool downloadDebugComponents(std::vector<float> &) { return false; }

  float criticWeight(int) const { return 0.0f; }
};

#endif // PLANNING_ENABLE_CUDA

} // namespace cuda
} // namespace mppi
