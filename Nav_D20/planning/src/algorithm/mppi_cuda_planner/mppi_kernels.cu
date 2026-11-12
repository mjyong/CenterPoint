// CUDA MPPI kernels. The rollout kernel replicates the CPU hot path of
// optimizer.cc / motion_models.h step by step (same clamp order, same
// integration formulas, accurate cosf/sinf) so GPU and CPU results are
// numerically equivalent up to float rounding.

#include "src/algorithm/mppi_cuda_planner/mppi_kernels.cuh"

#include <cuda_runtime.h>

#include <cfloat>

#include "src/common/util/log.h"

// Mirrors tools/utils.h (kept local to avoid pulling Eigen into nvcc).
#define M_PIF 3.141592653589793238462643383279502884e+00F
#define M_PIF_2 1.5707963267948966e+00F

namespace mppi {
namespace cuda {

namespace {

// Path table row offsets (d_path is 4 x kMaxPathPoints).
constexpr int kPathRowX = 0;
constexpr int kPathRowY = 1;
constexpr int kPathRowYaw = 2;
constexpr int kPathRowArc = 3;

// utils::clamp(lower, upper, input) = min(upper, max(input, lower))
__device__ __forceinline__ float clampf(float lo, float hi, float v) {
  return fminf(hi, fmaxf(v, lo));
}

// utils::normalize_angle / shortest_angular_distance (float fmod semantics).
__device__ __forceinline__ float normalizeAngle(float angle) {
  float remainder = fmodf(angle + M_PIF, 2.0f * M_PIF);
  if (remainder < 0.0f) {
    remainder += 2.0f * M_PIF;
  }
  return remainder - M_PIF;
}

__device__ __forceinline__ float shortestAngularDistance(float from,
                                                         float to) {
  return normalizeAngle(to - from);
}

// OmniMotionModel::lateralVyCap
__device__ __forceinline__ float lateralVyCap(const MppiDeviceParams *p,
                                              float abs_vx) {
  if (abs_vx <= p->lateral_blend_vx_start) {
    return p->vy_max;
  }
  if (abs_vx >= p->lateral_blend_vx_end) {
    return 0.0f;
  }
  const float t =
      (abs_vx - p->lateral_blend_vx_start) /
      (p->lateral_blend_vx_end - p->lateral_blend_vx_start);
  return p->vy_max * (1.0f - t);
}

// Obstacle distance sampling, a step-by-step replica of
// CostMapProduction::GetObstacleDistance -> grid_map::GridMap::atPosition(
// layer, pos, INTER_LINEAR): out-of-map queries return 1e6, in-map queries
// bilinearly interpolate the column-major field (linear = j * rows + i, x
// decreases as the buffer index i grows); when any of the four neighbors
// falls outside the buffer (grid_map checks the size_t linear index against
// [0, rows*cols] inclusive, negative indices wrap to huge values) it falls
// back to INTER_NEAREST. All geometry math runs in double like the CPU.
__device__ float sampleObstacleDistance(const MppiCycleParams *c, double px,
                                        double py) {
  // GridMap::isInside: transformed = 0.5*len - (p - map_position) must lie in
  // [0, len) component-wise (asymmetric bounds preserved).
  const double tx = 0.5 * c->map_length_x - (px - c->map_position_x);
  const double ty = 0.5 * c->map_length_y - (py - c->map_position_y);
  if (!(tx >= 0.0 && ty >= 0.0 && tx < c->map_length_x &&
        ty < c->map_length_y)) {
    return 1000000.0f;
  }

  const double res = c->map_resolution;
  // Continuous buffer indices; the cell index is the truncated value (Eigen
  // Vector2d -> Array2i cast truncates towards zero, matching the CPU's
  // transformMapFrameToBufferOrder on the negated index vector).
  const double exf = (0.5 * c->map_length_x + c->map_position_x - px) / res;
  const double eyf = (0.5 * c->map_length_y + c->map_position_y - py) / res;
  const int i0 = static_cast<int>(exf);
  const int j0 = static_cast<int>(eyf);
  // Cell center of indices[0] (GridMap::getPositionFromIndex in doubles).
  const double cx0 =
      c->map_position_x + (0.5 * c->map_length_x - 0.5 * res) -
      static_cast<double>(i0) * res;
  const double cy0 =
      c->map_position_y + (0.5 * c->map_length_y - 0.5 * res) -
      static_cast<double>(j0) * res;
  const int i1 = (px >= cx0) ? i0 - 1 : i0 + 1;
  const int j1 = (py >= cy0) ? j0 - 1 : j0 + 1;

  // The four neighbors (i0/j0), (i1/j0), (i0/j1), (i1/j1) and the idxShift
  // remap from atPositionLinearInterpolated.
  const int ii[4] = {i0, i1, i0, i1};
  const int jj[4] = {j0, j0, j1, j1};
  int shift[4];
  if (px >= cx0) {
    if (py >= cy0) {
      shift[0] = 0; shift[1] = 1; shift[2] = 2; shift[3] = 3;
    } else {
      shift[0] = 2; shift[1] = 3; shift[2] = 0; shift[3] = 1;
    }
  } else {
    if (py >= cy0) {
      shift[0] = 1; shift[1] = 0; shift[2] = 3; shift[3] = 2;
    } else {
      shift[0] = 3; shift[1] = 2; shift[2] = 1; shift[3] = 0;
    }
  }

  const int rows = c->map_rows;
  const long long total = static_cast<long long>(rows) * c->map_cols;
  const float *field = c->distance_field;
  float f[4];
  for (int k = 0; k < 4; ++k) {
    const int ci = ii[shift[k]];
    const int cj = jj[shift[k]];
    // size_t linear index: negatives wrap to huge values and fail the range
    // check; linear == total passes on the CPU (an inclusive end bound that
    // reads one element past the buffer, undefined there). Anything at or
    // past total falls back to the nearest cell here, which is guaranteed
    // in range by the isInside check above.
    const long long lin = static_cast<long long>(cj) * rows + ci;
    if (ci < 0 || cj < 0 || lin >= total) {
      return field[static_cast<long long>(j0) * rows + i0];
    }
    f[k] = field[lin];
  }

  // Bilinear weights relative to the idxShift[0] corner center.
  const double p00x =
      c->map_position_x + (0.5 * c->map_length_x - 0.5 * res) -
      static_cast<double>(ii[shift[0]]) * res;
  const double p00y =
      c->map_position_y + (0.5 * c->map_length_y - 0.5 * res) -
      static_cast<double>(jj[shift[0]]) * res;
  const double red_x = (px - p00x) / res;
  const double red_y = (py - p00y) / res;
  const double val = static_cast<double>(f[0]) * (1.0 - red_x) * (1.0 - red_y) +
                     static_cast<double>(f[1]) * red_x * (1.0 - red_y) +
                     static_cast<double>(f[2]) * (1.0 - red_x) * red_y +
                     static_cast<double>(f[3]) * red_x * red_y;
  return static_cast<float>(val);
}

// utils::findClosestPathPt: linear scan over the arclength prefix table,
// starting from the previous (monotonic) path point.
__device__ __forceinline__ unsigned int findClosestPathPtDev(
    const float *vec, int size, float dist, unsigned int init) {
  float distim1 = init != 0u ? vec[init] : 0.0f;
  float disti = 0.0f;
  const unsigned int usize = static_cast<unsigned int>(size);
  for (unsigned int i = init + 1; i != usize; i++) {
    disti = vec[i];
    if (disti > dist) {
      if (i > 0 && dist - distim1 < disti - dist) {
        return i - 1;
      }
      return i;
    }
    distim1 = disti;
  }
  return usize - 1;
}

__global__ void rolloutKernel(const MppiDeviceParams *p,
                              const MppiCycleParams *c, float *cvx,
                              float *cvy, float *cwz, float *vx, float *vy,
                              float *wz, float *x, float *y, float *yaws) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int B = p->batch_size;
  if (j >= B) {
    return;
  }
  const int T = p->time_steps;
  const float dt = p->model_dt;
  const bool holo = p->motion_model == kMotionOmni;
  const bool coupled = holo && p->lateral_coupling != 0;

  // Optimizer::updateInitialStateVelocities
  vx[0 * B + j] = c->speed_vx;
  wz[0 * B + j] = c->speed_wz;
  if (holo) {
    vy[0 * B + j] = c->speed_vy;
  }

  // MotionModel::predict: acceleration boxes; the clamped control is written
  // back into cvx/cvy/cwz and the actual velocity approaches it with the
  // per-axis first-order response factor dt/(tau+dt) (CPU semantics; 1.0
  // when tau = 0, i.e. instant response).
  const float max_dvx = dt * p->ax_max;
  const float min_dvx = dt * p->ax_min;
  const float max_dvy = dt * p->ay_max;
  const float max_dwz = dt * p->az_max;
  const float alpha_vx = dt / (p->response_tau_vx + dt);
  const float alpha_vy = dt / (p->response_tau_vy + dt);
  const float alpha_wz = dt / (p->response_tau_wz + dt);

  for (int t = 1; t < T; ++t) {
    float vx_last = vx[(t - 1) * B + j];
    float cvx_curr = clampf(vx_last + min_dvx, vx_last + max_dvx,
                            cvx[(t - 1) * B + j]);
    cvx[(t - 1) * B + j] = cvx_curr;
    vx[t * B + j] = vx_last + alpha_vx * (cvx_curr - vx_last);

    float wz_last = wz[(t - 1) * B + j];
    float cwz_curr = clampf(wz_last - max_dwz, wz_last + max_dwz,
                            cwz[(t - 1) * B + j]);
    cwz[(t - 1) * B + j] = cwz_curr;
    wz[t * B + j] = wz_last + alpha_wz * (cwz_curr - wz_last);

    if (holo) {
      float vy_last = vy[(t - 1) * B + j];
      float cvy_curr = cvy[(t - 1) * B + j];
      if (coupled) {
        // OmniMotionModel::predict joint box: acceleration limit AND the
        // lateral coupling cap evaluated at the new longitudinal speed.
        const float vy_cap = lateralVyCap(p, fabsf(vx[t * B + j]));
        const float vy_hi = clampf(0.0f, vy_cap, vy_last + max_dvy);
        const float vy_lo = clampf(-vy_cap, 0.0f, vy_last - max_dvy);
        cvy_curr = clampf(vy_lo, vy_hi, cvy_curr);
      } else {
        cvy_curr = clampf(vy_last - max_dvy, vy_last + max_dvy, cvy_curr);
      }
      cvy[(t - 1) * B + j] = cvy_curr;
      vy[t * B + j] = vy_last + alpha_vy * (cvy_curr - vy_last);
    }
  }

  // Optimizer::integrateStateVelocities: yaws(t) = yaws(t-1) + wz(t)*dt with
  // yaws(-1) = initial yaw; positions integrate vx/vy on the *previous*
  // step's yaw (shifted cos/sin on the CPU).
  float yaw_last = c->pose_yaw;
  float px = c->pose_x;
  float py = c->pose_y;
  for (int t = 0; t < T; ++t) {
    const float cs = cosf(yaw_last);
    const float sn = sinf(yaw_last);
    const float v = vx[t * B + j];
    float dx = v * cs;
    float dy = v * sn;
    if (holo) {
      const float v_lat = vy[t * B + j];
      dx -= v_lat * sn;
      dy += v_lat * cs;
    }
    px += dx * dt;
    py += dy * dt;
    x[t * B + j] = px;
    y[t * B + j] = py;

    yaw_last = yaw_last + wz[t * B + j] * dt;
    yaws[t * B + j] = yaw_last;
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Fused critic scoring
// ---------------------------------------------------------------------------

namespace {

// One thread per trajectory. Computes every active critic's raw cost
// component (weights/powers are applied in finalizeKernel). Each critic block
// replicates the corresponding CriticFunction::score math step by step; the
// threshold/gating decisions were already made on the host and arrive as
// MppiCycleParams::critic_active bits.
__global__ void costKernel(const MppiDeviceParams *p,
                           const MppiCycleParams *c, const float *vx,
                           const float *vy, const float *wz, const float *x,
                           const float *y, const float *yaws,
                           const float *path, const unsigned char *path_valid,
                           float *components, float *repulsive,
                           unsigned char *collision) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int B = p->batch_size;
  if (j >= B) {
    return;
  }
  const int T = p->time_steps;
  const float dt = p->model_dt;
  const bool holo = p->motion_model == kMotionOmni;
  const unsigned active = c->critic_active;

  // --- ConstraintCritic: acceleration-box excess integrated over dt ---
  if (active & (1u << kCriticConstraint)) {
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      const float v = vx[t * B + j];
      float excess;
      if (holo) {
        const float v_lat = vy[t * B + j];
        const float sgn = copysignf(1.0f, v);
        const float vel_total = sgn * sqrtf(v * v + v_lat * v_lat);
        excess = fmaxf(vel_total - p->max_vel, 0.0f) +
                 fmaxf(p->min_vel - vel_total, 0.0f);
      } else {
        excess = fmaxf(v - p->max_vel, 0.0f) +
                 fmaxf(p->min_vel - v, 0.0f);
      }
      sum += excess * dt;
    }
    components[kCriticConstraint * B + j] = sum;
  }

  // --- GoalCritic: mean distance of every rollout point to the goal ---
  if (active & (1u << kCriticGoal)) {
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      const float dx = x[t * B + j] - c->goal_x;
      const float dy = y[t * B + j] - c->goal_y;
      sum += sqrtf(dx * dx + dy * dy);
    }
    components[kCriticGoal * B + j] = sum / static_cast<float>(T);
  }

  // --- GoalAngleCritic: mean |shortest angular distance| to the path yaw ---
  if (active & (1u << kCriticGoalAngle)) {
    const float goal_yaw = c->goal_yaw_from_path;
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      sum += fabsf(shortestAngularDistance(yaws[t * B + j], goal_yaw));
    }
    components[kCriticGoalAngle * B + j] = sum / static_cast<float>(T);
  }

  // --- ObstaclesCritic: multi-circle footprint clearance walk ---
  // Per-point: minimum (field distance at circle center - circle radius);
  // early-break on collision; margin and repulsion accumulation. The final
  // normalized repulsive term needs the batch-wide min (reduceKernel) and is
  // assembled in finalizeKernel.
  if (active & (1u << kCriticObstacles)) {
    const bool near_goal = c->obstacles_near_goal != 0;
    float raw = 0.0f;
    float rep = 0.0f;
    bool collided = false;
    for (int t = 0; t < T; ++t) {
      const float px = x[t * B + j];
      const float py = y[t * B + j];
      const float th = yaws[t * B + j];
      const float cos_t = cosf(th);
      const float sin_t = sinf(th);
      float min_clearance = FLT_MAX;
      for (int ci = 0; ci < p->fp_count; ++ci) {
        const float cx = px + cos_t * p->fp_x[ci] - sin_t * p->fp_y[ci];
        const float cy = py + sin_t * p->fp_x[ci] + cos_t * p->fp_y[ci];
        const float clearance =
            sampleObstacleDistance(c, static_cast<double>(cx),
                                   static_cast<double>(cy)) -
            p->fp_r[ci];
        if (clearance < min_clearance) {
          min_clearance = clearance;
        }
      }
      if (min_clearance < 0.0f) {
        collided = true;
        break;
      }
      if (min_clearance > p->repulsion_radius &&
          min_clearance > p->collision_margin_distance) {
        continue;
      }
      if (min_clearance < p->collision_margin_distance) {
        raw += (p->collision_margin_distance - min_clearance);
      }
      if (!near_goal) {
        rep += (p->repulsion_radius - min_clearance);
      }
    }
    components[kCriticObstacles * B + j] =
        collided ? p->collision_cost : raw;
    repulsive[j] = rep;
    collision[j] = collided ? 1u : 0u;
  }

  // --- PathAlignCritic: strided sampled arclength alignment ---
  if (active & (1u << kCriticPathAlign)) {
    const int segs = c->path_align_segments;
    const int step = p->trajectory_point_step;
    const int strided_cols = (T - 1) / step + 1;
    const float *path_x = path + kPathRowX * kMaxPathPoints;
    const float *path_y = path + kPathRowY * kMaxPathPoints;
    const float *path_yaw = path + kPathRowYaw * kMaxPathPoints;
    const float *arclen = path + kPathRowArc * kMaxPathPoints;

    float summed = 0.0f;
    unsigned int num_samples = 0u;
    unsigned int path_pt = 0u;
    float traj_dist = 0.0f;
    float prev_x = x[0 * B + j];
    float prev_y = y[0 * B + j];
    for (int q = 1; q < strided_cols; ++q) {
      const int t = q * step;
      const float cur_x = x[t * B + j];
      const float cur_y = y[t * B + j];
      const float dx = cur_x - prev_x;
      const float dy = cur_y - prev_y;
      prev_x = cur_x;
      prev_y = cur_y;
      traj_dist += sqrtf(dx * dx + dy * dy);
      path_pt = findClosestPathPtDev(arclen, segs, traj_dist, path_pt);
      if (path_valid[path_pt]) {
        const float ddx = path_x[path_pt] - cur_x;
        const float ddy = path_y[path_pt] - cur_y;
        ++num_samples;
        if (p->use_path_orientations) {
          const float dyaw =
              shortestAngularDistance(path_yaw[path_pt], yaws[t * B + j]);
          summed += sqrtf(ddx * ddx + ddy * ddy + dyaw * dyaw);
        } else {
          summed += sqrtf(ddx * ddx + ddy * ddy);
        }
      }
    }
    components[kCriticPathAlign * B + j] =
        num_samples > 0u ? summed / static_cast<float>(num_samples) : 0.0f;
  }

  // --- PathAngleCritic: terminal heading vs the path target point ---
  if (active & (1u << kCriticPathAngle)) {
    const int last = T - 1;
    const float last_x = x[last * B + j];
    const float last_y = y[last * B + j];
    const float last_yaw = yaws[last * B + j];
    const float gx = c->path_angle_goal_x;
    const float gy = c->path_angle_goal_y;
    const float gyaw = c->path_angle_goal_yaw;
    const float yaw_between = atan2f(gy - last_y, gx - last_x);
    float val = 0.0f;
    if (p->path_angle_mode == 0) { // FORWARD_PREFERENCE
      val = fabsf(shortestAngularDistance(last_yaw, yaw_between));
    } else if (p->path_angle_mode == 1) { // NO_DIRECTIONAL_PREFERENCE
      // normalize_yaws_between_points(last_yaws, yaw_between)
      const float dist =
          fabsf(shortestAngularDistance(last_yaw, yaw_between));
      const float corrected = dist < M_PIF_2
                                  ? yaw_between
                                  : normalizeAngle(yaw_between + M_PIF);
      val = fabsf(shortestAngularDistance(last_yaw, corrected));
    } else { // CONSIDER_FEASIBLE_PATH_ORIENTATIONS
      // normalize_yaws_between_points(goal_yaw, yaw_between)
      const float corrected =
          fabsf(normalizeAngle(yaw_between - gyaw)) < M_PIF_2
              ? yaw_between
              : normalizeAngle(yaw_between + M_PIF);
      val = fabsf(shortestAngularDistance(last_yaw, corrected));
    }
    components[kCriticPathAngle * B + j] = val;
  }

  // --- PathFollowCritic: terminal distance to the follow target point ---
  if (active & (1u << kCriticPathFollow)) {
    const int last = T - 1;
    const float dx = x[last * B + j] - c->path_follow_goal_x;
    const float dy = y[last * B + j] - c->path_follow_goal_y;
    components[kCriticPathFollow * B + j] = sqrtf(dx * dx + dy * dy);
  }

  // --- PreferForwardCritic: backward motion penalty ---
  if (active & (1u << kCriticPreferForward)) {
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      sum += fmaxf(-vx[t * B + j], 0.0f) * dt;
    }
    components[kCriticPreferForward * B + j] = sum;
  }

  // --- ReferenceVelocityCritic: speed profile tracking. The envelope uses
  // a running minimum of the remaining distance along the trajectory
  // (monotone non-increasing; a raw distance would reward overshooting the
  // goal with a growing "donut" target), and the penalty is one-sided inside
  // the decel zone so braking harder than the envelope is free, bilateral
  // while cruising so the speed converges to the reference. Mirrors the CPU
  // critic exactly. ---
  if (active & (1u << kCriticReferenceVelocity)) {
    const float v_ref = c->v_ref;
    const float decel = p->decel_start_dist;
    float sum = 0.0f;
    float dist_min = FLT_MAX;
    for (int t = 0; t < T; ++t) {
      const float v = vx[t * B + j];
      float speed;
      if (holo) {
        const float v_lat = vy[t * B + j];
        speed = sqrtf(v * v + v_lat * v_lat);
      } else {
        speed = fabsf(v);
      }
      const float dx = c->goal_x - x[t * B + j];
      const float dy = c->goal_y - y[t * B + j];
      dist_min = fminf(dist_min, sqrtf(dx * dx + dy * dy));
      const float target =
          v_ref * sqrtf(fminf(dist_min / decel, 1.0f));
      if (dist_min < decel) {
        // Decel zone: one-sided, only speeds above the envelope are billed
        sum += fmaxf(speed - target, 0.0f) * dt;
      } else {
        sum += fabsf(speed - target) * dt;
      }
    }
    components[kCriticReferenceVelocity * B + j] = sum;
  }

  // --- TwirlingCritic: mean |wz| over the horizon (spinning penalty). The
  // CPU critic only scores within the goal tolerance; that gate is evaluated
  // on the host (active bit), so no extra condition here.
  if (active & (1u << kCriticTwirling)) {
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      sum += fabsf(wz[t * B + j]);
    }
    components[kCriticTwirling * B + j] = sum / static_cast<float>(T);
  }

  // --- VelocityDeadbandCritic: per-axis shortfall below the deadband
  // (compared against fabs of the configured deadband, like the CPU),
  // integrated over dt (holonomic adds the vy axis).
  if (active & (1u << kCriticVelocityDeadband)) {
    const float db_vx = fabsf(p->deadband_vx);
    const float db_vy = fabsf(p->deadband_vy);
    const float db_wz = fabsf(p->deadband_wz);
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      float axis =
          fmaxf(db_vx - fabsf(vx[t * B + j]), 0.0f) +
          fmaxf(db_wz - fabsf(wz[t * B + j]), 0.0f);
      if (holo) {
        axis += fmaxf(db_vy - fabsf(vy[t * B + j]), 0.0f);
      }
      sum += axis * dt;
    }
    components[kCriticVelocityDeadband * B + j] = sum;
  }
}

// Single-block batch reductions: min of the repulsive cost (Eigen minCoeff)
// and the collision AND (all_trajectories_collide).
__global__ void reduceKernel(int batch_size, const float *repulsive,
                             const unsigned char *collision, float *min_rep,
                             int *all_collide) {
  __shared__ float s_min[256];
  __shared__ int s_all[256];
  const int tid = threadIdx.x;
  float local_min = FLT_MAX;
  int local_all = 1;
  for (int j = tid; j < batch_size; j += blockDim.x) {
    local_min = fminf(local_min, repulsive[j]);
    local_all = local_all & (collision[j] ? 1 : 0);
  }
  s_min[tid] = local_min;
  s_all[tid] = local_all;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
      s_all[tid] = s_all[tid] & s_all[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *min_rep = s_min[0];
    *all_collide = s_all[0];
  }
}

// Final accumulation. Adds every active critic's weighted (and powered)
// contribution to the incoming costs, in the configured critic-list order —
// exactly the sequence of `data.costs += ...` on the CPU. When every
// trajectory collides, critics listed after ObstaclesCritic are dropped
// (CriticManager::evalTrajectoriesScores breaks on data.fail_flag).
__global__ void finalizeKernel(const MppiDeviceParams *p,
                               const MppiCycleParams *c, float *costs,
                               const float *components,
                               const float *repulsive, const float *min_rep,
                               const int *all_collide, int *fail_flag) {
  const int B = p->batch_size;
  const int T = p->time_steps;
  const unsigned active = c->critic_active;
  // The fail-flag break only matters when ObstaclesCritic actually ran (it is
  // the only critic setting data.fail_flag); d_all_collide is unspecified
  // otherwise, so gate the mask on the active bit as well.
  const unsigned skip_mask =
      ((*all_collide != 0) && (active & (1u << kCriticObstacles)))
          ? p->post_obstacles_mask
          : 0u;

  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < B;
       j += blockDim.x * gridDim.x) {
    float sum = costs[j];
    for (int k = 0; k < p->critic_order_count; ++k) {
      const int id = p->critic_order[k];
      if (!(active & (1u << id)) || (skip_mask & (1u << id))) {
        continue;
      }
      const float w = p->critic_weight[id];
      const int power = p->critic_power[id];
      if (id == kCriticObstacles) {
        const float rep_norm = (repulsive[j] - *min_rep) /
                               static_cast<float>(T);
        const float v = p->critical_weight * components[id * B + j] +
                        w * rep_norm; // w == repulsion_weight
        // Eigen .pow(power) semantics for unsigned int exponents (powf
        // returns exactly v for 1 and 1 for 0).
        sum += power == 1 ? v : powf(v, static_cast<float>(power));
      } else {
        const float v = components[id * B + j] * w;
        sum += power == 1 ? v : powf(v, static_cast<float>(power));
      }
    }
    costs[j] = sum;
  }

  // data.fail_flag = all_trajectories_collide (only when the obstacles
  // critic actually ran).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *fail_flag = (active & (1u << kCriticObstacles)) ? *all_collide : 0;
  }
}

} // namespace

void launchCostKernel(const MppiDeviceParams *d_params,
                      const MppiCycleParams *d_cycle, int batch_size,
                      const float *vx, const float *vy, const float *wz,
                      const float *x, const float *y, const float *yaws,
                      const float *d_path, const unsigned char *d_path_valid,
                      float *d_components, float *d_repulsive,
                      unsigned char *d_collision) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  costKernel<<<blocks, threads>>>(d_params, d_cycle, vx, vy, wz, x, y, yaws,
                                  d_path, d_path_valid, d_components,
                                  d_repulsive, d_collision);
}

void launchReduceKernel(int batch_size, const float *d_repulsive,
                        const unsigned char *d_collision, float *d_min_rep,
                        int *d_all_collide) {
  reduceKernel<<<1, 256>>>(batch_size, d_repulsive, d_collision, d_min_rep,
                           d_all_collide);
}

void launchFinalizeKernel(const MppiDeviceParams *d_params,
                          const MppiCycleParams *d_cycle, int batch_size,
                          float *d_costs, const float *d_components,
                          const float *d_repulsive, const float *d_min_rep,
                          const int *d_all_collide, int *d_fail_flag) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  finalizeKernel<<<blocks, threads>>>(d_params, d_cycle, d_costs,
                                      d_components, d_repulsive, d_min_rep,
                                      d_all_collide, d_fail_flag);
}

void launchRolloutKernel(const MppiDeviceParams *d_params,
                         const MppiCycleParams *d_cycle, int batch_size,
                         float *cvx, float *cvy, float *cwz, float *vx,
                         float *vy, float *wz, float *x, float *y,
                         float *yaws) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  rolloutKernel<<<blocks, threads>>>(d_params, d_cycle, cvx, cvy, cwz, vx, vy,
                                     wz, x, y, yaws);
}

} // namespace cuda
} // namespace mppi
