// Kernel declarations for the CUDA MPPI backend. Launch wrappers are host
// functions defined in mppi_kernels.cu and called from mppi_cuda_backend.cu.

#pragma once

#include "src/algorithm/mppi_cuda_planner/device_params.cuh"

namespace mppi {
namespace cuda {

// Rollout propagation: one thread per trajectory, serial over the horizon.
// Replicates Optimizer::updateStateVelocities (initial velocities +
// MotionModel::predict, including the Omni lateral-coupling joint box and the
// clamped-control write-back) followed by
// Optimizer::integrateStateVelocities (yaw prefix sum, position integration
// with the previous step's cos/sin).
// All matrices are batch x time_steps column-major: index (j, t) -> t*batch+j.
// batch_size is the host-side dimension used for the launch grid.
void launchRolloutKernel(const MppiDeviceParams *d_params,
                         const MppiCycleParams *d_cycle, int batch_size,
                         float *cvx, float *cvy, float *cwz, float *vx,
                         float *vy, float *wz, float *x, float *y,
                         float *yaws);

// Fused critic scoring: one thread per trajectory, serial over the horizon.
// Computes the raw per-critic cost components (weights and powers are applied
// later by the finalize kernel) for the nine GPU-supported critics, exactly
// mirroring the CPU CriticFunction::score math including the
// threshold/gating bits evaluated on the host (MppiCycleParams::critic_active).
// Outputs:
//   d_components[k * batch + j]  raw cost of critic k for trajectory j
//   d_repulsive[j], d_collision[j]  ObstaclesCritic extras (the repulsive
//       term still needs a batch-wide min reduction, the collision flag the
//       batch-wide AND).
// d_path is the host-precomputed path table: rows x / y / yaw / arclength,
// each kMaxPathPoints long (only [0, segments-1] is meaningful);
// d_path_valid holds the path-point validity flags (path_count - 1 entries).
void launchCostKernel(const MppiDeviceParams *d_params,
                      const MppiCycleParams *d_cycle, int batch_size,
                      const float *vx, const float *vy, const float *wz,
                      const float *x, const float *y, const float *yaws,
                      const float *d_path, const unsigned char *d_path_valid,
                      float *d_components, float *d_repulsive,
                      unsigned char *d_collision);

// Batch reductions for ObstaclesCritic: minimum of the repulsive cost over
// all trajectories and whether every trajectory collides.
void launchReduceKernel(int batch_size, const float *d_repulsive,
                        const unsigned char *d_collision, float *d_min_rep,
                        int *d_all_collide);

// Final accumulation: one thread per trajectory (grid-stride), applies the
// critic weights/powers in the configured critic-list order, mirrors the
// CriticManager fail_flag break via the post-obstacles mask, and adds the
// result to the incoming costs in place.
void launchFinalizeKernel(const MppiDeviceParams *d_params,
                          const MppiCycleParams *d_cycle, int batch_size,
                          float *d_costs, const float *d_components,
                          const float *d_repulsive, const float *d_min_rep,
                          const int *d_all_collide, int *d_fail_flag);

} // namespace cuda
} // namespace mppi
