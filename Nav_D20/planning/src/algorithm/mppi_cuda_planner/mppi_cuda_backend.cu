// Host-side implementation of the CUDA MPPI backend: rollout propagation and
// fused critic scoring on the GPU, threshold/gating decisions and the shared
// path bookkeeping (furthest reached point, path validity, arclength table)
// on the host. Softmax, control update, Savitzky-Golay and shifting stay on
// the CPU, shared with the CPU path.

#include "src/algorithm/mppi_cuda_planner/mppi_cuda_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "src/algorithm/mppi_cuda_planner/device_params.cuh"
#include "src/algorithm/mppi_cuda_planner/mppi_kernels.cuh"
#include "src/algorithm/mppi_planner/tools/utils.h"
#include "src/common/util/log.h"

namespace mppi {
namespace cuda {

namespace {

// Map a configured critic name to its DeviceCriticId slot; -1 when the critic
// has no GPU implementation (or is unknown).
int deviceCriticSlot(const std::string &name) {
  if (name == "ConstraintCritic") return kCriticConstraint;
  if (name == "GoalCritic") return kCriticGoal;
  if (name == "GoalAngleCritic") return kCriticGoalAngle;
  if (name == "ObstaclesCritic") return kCriticObstacles;
  if (name == "PathAlignCritic") return kCriticPathAlign;
  if (name == "PathAngleCritic") return kCriticPathAngle;
  if (name == "PathFollowCritic") return kCriticPathFollow;
  if (name == "PreferForwardCritic") return kCriticPreferForward;
  if (name == "ReferenceVelocityCritic") return kCriticReferenceVelocity;
  if (name == "TwirlingCritic") return kCriticTwirling;
  if (name == "VelocityDeadbandCritic") return kCriticVelocityDeadband;
  return -1;
}

// Critics with a GPU implementation in the fused cost kernel. Any other
// enabled critic forces the CPU path at init time.
const std::vector<std::string> &gpuSupportedCritics() {
  static const std::vector<std::string> kCritics = {
      "ConstraintCritic",     "GoalCritic",
      "GoalAngleCritic",      "ObstaclesCritic",
      "PathAlignCritic",      "PathAngleCritic",
      "PathFollowCritic",     "PreferForwardCritic",
      "ReferenceVelocityCritic", "TwirlingCritic",
      "VelocityDeadbandCritic"};
  return kCritics;
}

// Effective critic list, mirroring CriticManager::on_configure.
std::vector<std::string> effectiveCriticList(
    const planning::config::MppiPlannerConf &conf) {
  if (conf.critics().empty()) {
    return {"PathAlignCritic",     "PathFollowCritic",   "PreferForwardCritic",
            "GoalCritic",          "GoalAngleCritic",    "PathAngleCritic",
            "TwirlingCritic",      "ConstraintCritic",
            "VelocityDeadbandCritic", "ObstaclesCritic",
            "ReferenceVelocityCritic"};
  }
  return std::vector<std::string>(conf.critics().begin(), conf.critics().end());
}

// Whether the named critic is enabled in the proto sub-configuration.
bool criticEnabled(const planning::config::MppiPlannerConf &conf,
                   const std::string &name) {
  if (name == "ConstraintCritic") {
    return conf.has_constraint_critic() ? conf.constraint_critic().enabled()
                                        : true;
  }
  if (name == "GoalCritic") {
    return conf.has_goal_critic() ? conf.goal_critic().enabled() : true;
  }
  if (name == "GoalAngleCritic") {
    return conf.has_goal_angle_critic() ? conf.goal_angle_critic().enabled()
                                        : true;
  }
  if (name == "ObstaclesCritic") {
    return conf.has_obstacles_critic() ? conf.obstacles_critic().enabled() : true;
  }
  if (name == "PathAlignCritic") {
    return conf.has_path_align_critic() ? conf.path_align_critic().enabled()
                                        : true;
  }
  if (name == "PathAngleCritic") {
    return conf.has_path_angle_critic() ? conf.path_angle_critic().enabled()
                                        : true;
  }
  if (name == "PathFollowCritic") {
    return conf.has_path_follow_critic() ? conf.path_follow_critic().enabled()
                                         : true;
  }
  if (name == "PreferForwardCritic") {
    return conf.has_prefer_forward_critic()
               ? conf.prefer_forward_critic().enabled()
               : true;
  }
  if (name == "TwirlingCritic") {
    return conf.has_twirling_critic() ? conf.twirling_critic().enabled() : true;
  }
  if (name == "VelocityDeadbandCritic") {
    return conf.has_velocity_deadband_critic()
               ? conf.velocity_deadband_critic().enabled()
               : true;
  }
  if (name == "ReferenceVelocityCritic") {
    return conf.has_reference_velocity_critic()
               ? conf.reference_velocity_critic().enabled()
               : true;
  }
  return false;
}

#define CUDA_CHECK(expr, action)                                               \
  do {                                                                         \
    cudaError_t err_ = (expr);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      AERROR << "MPPI CUDA error at " << __FILE__ << ":" << __LINE__ << ": "   \
             << cudaGetErrorString(err_);                                      \
      action;                                                                  \
    }                                                                          \
  } while (0)

} // namespace

MppiCudaBackend::~MppiCudaBackend() { release(); }

bool MppiCudaBackend::init(
    const planning::config::MppiPlannerConf &conf,
    const models::OptimizerSettings &settings, const CriticContext & /*ctx*/) {
  if (ready_) {
    return true;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count), return false);
  if (device_count <= 0) {
    AWARN << "MPPI CUDA backend: no CUDA device, using CPU path";
    return false;
  }

  switch (conf.motion_model()) {
  case planning::config::MppiPlannerConf::DiffDrive:
  case planning::config::MppiPlannerConf::Omni:
    break;
  case planning::config::MppiPlannerConf::Ackermann:
  default:
    AWARN << "MPPI CUDA backend: motion model "
          << conf.motion_model() << " has no GPU implementation yet";
    return false;
  }

  if (!checkCriticSupport(conf)) {
    return false;
  }

  batch_size_ = static_cast<int>(settings.batch_size);
  time_steps_ = static_cast<int>(settings.time_steps);
  holonomic_ = conf.motion_model() ==
               planning::config::MppiPlannerConf::Omni;
  if (batch_size_ <= 0 || time_steps_ <= 0) {
    AWARN << "MPPI CUDA backend: invalid dimensions batch=" << batch_size_
          << " time_steps=" << time_steps_;
    return false;
  }

  const size_t mat_bytes =
      static_cast<size_t>(batch_size_) * time_steps_ * sizeof(float);
  const size_t vec_bytes = static_cast<size_t>(batch_size_) * sizeof(float);

  // Host mirrors for the per-cycle gating (critic enabled flags, adjusted
  // path-angle mode, configured reference velocity).
  {
    const auto &names = effectiveCriticList(conf);
    for (const auto &name : names) {
      const int slot = deviceCriticSlot(name);
      if (slot >= 0 && criticEnabled(conf, name)) {
        critic_enabled_[slot] = true;
      }
    }
    reference_velocity_config_ =
        conf.has_reference_velocity_critic()
            ? static_cast<float>(
                  conf.reference_velocity_critic().reference_velocity())
            : 0.0f;
  }

  CUDA_CHECK(cudaMalloc(&d_params_, sizeof(MppiDeviceParams)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_cycle_, sizeof(MppiCycleParams)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_cvx_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_cvy_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_cwz_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_vx_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_vy_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_wz_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_x_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_y_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_yaws_, mat_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_costs_, vec_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_collision_, batch_size_ * sizeof(unsigned char)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_path_, 4 * kMaxPathPoints * sizeof(float)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_path_valid_, kMaxPathPoints * sizeof(unsigned char)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_components_,
                        kNumDeviceCritics * static_cast<size_t>(batch_size_) *
                            sizeof(float)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_repulsive_, vec_bytes), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_min_rep_, sizeof(float)), goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_all_collide_, sizeof(int)),
             goto cleanup);
  CUDA_CHECK(cudaMalloc(&d_fail_, sizeof(int)), goto cleanup);

  {
    MppiDeviceParams params = MppiDeviceParams::fromSettings(conf, settings);

    // Critic execution order and the fail-flag blocking mask, mirroring
    // CriticManager::evalTrajectoriesScores: critics run in the configured
    // list order, and everything after ObstaclesCritic is skipped once the
    // fail flag is set (every rollout in collision).
    params.critic_order_count = 0;
    params.post_obstacles_mask = 0u;
    int obstacles_position = -1;
    int position = 0;
    for (const auto &name : effectiveCriticList(conf)) {
      const int slot = deviceCriticSlot(name);
      if (slot < 0 || !criticEnabled(conf, name)) {
        continue;
      }
      params.critic_order[params.critic_order_count++] = slot;
      if (slot == kCriticObstacles) {
        obstacles_position = position;
      }
      ++position;
    }
    if (obstacles_position >= 0) {
      // Re-walk the list to mark the critics after ObstaclesCritic.
      int pos = 0;
      for (const auto &name : effectiveCriticList(conf)) {
        const int slot = deviceCriticSlot(name);
        if (slot < 0 || !criticEnabled(conf, name)) {
          continue;
        }
        if (pos > obstacles_position) {
          params.post_obstacles_mask |= (1u << slot);
        }
        ++pos;
      }
    }

    // Host mirror of the reversing-adjusted path-angle mode (used by the
    // per-cycle posePointAngle gate).
    path_angle_mode_ = params.path_angle_mode;
    // Host mirror of the critic weights (debug per-critic divergence check).
    for (int i = 0; i < kNumDeviceCritics; ++i) {
      critic_weight_[i] = params.critic_weight[i];
    }

    CUDA_CHECK(cudaMemcpy(d_params_, &params, sizeof(params),
                          cudaMemcpyHostToDevice),
               goto cleanup);
  }

  ready_ = true;
  AINFO << "MPPI CUDA backend initialized (batch=" << batch_size_
        << ", time_steps=" << time_steps_ << ")";
  return true;

cleanup:
  release();
  return false;
}

bool MppiCudaBackend::checkCriticSupport(
    const planning::config::MppiPlannerConf &conf) {
  const auto &supported = gpuSupportedCritics();
  for (const auto &name : effectiveCriticList(conf)) {
    if (!criticEnabled(conf, name)) {
      continue;
    }
    if (std::find(supported.begin(), supported.end(), name) ==
        supported.end()) {
      AWARN << "MPPI CUDA backend: critic " << name
            << " has no GPU implementation, using CPU path";
      return false;
    }
  }
  return true;
}

void MppiCudaBackend::release() {
  ready_ = false;
  if (d_params_ != nullptr) {
    cudaFree(d_params_);
    d_params_ = nullptr;
  }
  if (d_cycle_ != nullptr) {
    cudaFree(d_cycle_);
    d_cycle_ = nullptr;
  }
  float **buffers[] = {&d_cvx_, &d_cvy_, &d_cwz_, &d_vx_,  &d_vy_,
                       &d_wz_,  &d_x_,   &d_y_,   &d_yaws_};
  for (auto **buf : buffers) {
    if (*buf != nullptr) {
      cudaFree(*buf);
      *buf = nullptr;
    }
  }
  if (d_costs_ != nullptr) {
    cudaFree(d_costs_);
    d_costs_ = nullptr;
  }
  if (d_collision_ != nullptr) {
    cudaFree(d_collision_);
    d_collision_ = nullptr;
  }
  if (d_path_ != nullptr) {
    cudaFree(d_path_);
    d_path_ = nullptr;
  }
  if (d_path_valid_ != nullptr) {
    cudaFree(d_path_valid_);
    d_path_valid_ = nullptr;
  }
  if (d_components_ != nullptr) {
    cudaFree(d_components_);
    d_components_ = nullptr;
  }
  if (d_repulsive_ != nullptr) {
    cudaFree(d_repulsive_);
    d_repulsive_ = nullptr;
  }
  if (d_min_rep_ != nullptr) {
    cudaFree(d_min_rep_);
    d_min_rep_ = nullptr;
  }
  if (d_all_collide_ != nullptr) {
    cudaFree(d_all_collide_);
    d_all_collide_ = nullptr;
  }
  if (d_fail_ != nullptr) {
    cudaFree(d_fail_);
    d_fail_ = nullptr;
  }
}

bool MppiCudaBackend::downloadDebugComponents(std::vector<float> &out) {
  if (!ready_ || d_components_ == nullptr) {
    return false;
  }
  out.resize(static_cast<size_t>(kNumDeviceCritics) * batch_size_);
  cudaError_t err = cudaMemcpy(out.data(), d_components_,
                                out.size() * sizeof(float),
                                cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    AERROR << "MPPI CUDA component download failed: "
           << cudaGetErrorString(err);
    return false;
  }
  return true;
}

bool MppiCudaBackend::evalStep(
    models::State &state, models::Trajectories &trajectories,
    const models::Path &path, const models::Pose &goal,
    const CriticContext &ctx, const MppiCudaMapField &map,
    Eigen::ArrayXf &costs, bool &fail_flag) {
  if (!ready_) {
    return false;
  }

  const int path_count = static_cast<int>(path.x.size());
  if (path_count <= 0 || path_count > kMaxPathPoints) {
    AERROR << "MPPI CUDA: path has " << path.x.size() << " points, device "
           << "capacity is " << kMaxPathPoints;
    return false;
  }
  const auto &conf = *ctx.conf;

  const bool fail_in = fail_flag; // CriticManager would break immediately
  const bool obstacles_on = critic_enabled_[kCriticObstacles];
  if (obstacles_on && map.distance_field == nullptr) {
    // No device distance field wired in: obstacle scoring impossible.
    return false;
  }

  const size_t mat_elems =
      static_cast<size_t>(batch_size_) * time_steps_;
  const size_t mat_bytes = mat_elems * sizeof(float);
  const size_t vec_bytes = static_cast<size_t>(batch_size_) * sizeof(float);

  // 1) Upload per-cycle parameters and the CPU-overlaid noised controls.
  MppiCycleParams cyc{};
  cyc.pose_x = state.pose.x;
  cyc.pose_y = state.pose.y;
  cyc.pose_yaw = state.pose.yaw;
  cyc.speed_vx = state.speed.vx;
  cyc.speed_vy = state.speed.vy;
  cyc.speed_wz = state.speed.wz;
  cyc.goal_x = goal.x;
  cyc.goal_y = goal.y;
  cyc.goal_yaw = goal.yaw;
  cyc.runtime_reference_velocity =
      static_cast<float>(ctx.runtime_reference_velocity);
  cyc.path_count = path_count;
  cyc.distance_field = map.distance_field;
  cyc.map_rows = map.rows;
  cyc.map_cols = map.cols;
  cyc.map_resolution = map.resolution;
  cyc.map_position_x = map.map_position_x;
  cyc.map_position_y = map.map_position_y;
  cyc.map_length_x = map.length_x;
  cyc.map_length_y = map.length_y;

  CUDA_CHECK(cudaMemcpy(d_cycle_, &cyc, sizeof(cyc), cudaMemcpyHostToDevice),
             return false);
  CUDA_CHECK(cudaMemcpy(d_cvx_, state.cvx.data(), mat_bytes,
                        cudaMemcpyHostToDevice),
             return false);
  CUDA_CHECK(cudaMemcpy(d_cvy_, state.cvy.data(), mat_bytes,
                        cudaMemcpyHostToDevice),
             return false);
  CUDA_CHECK(cudaMemcpy(d_cwz_, state.cwz.data(), mat_bytes,
                        cudaMemcpyHostToDevice),
             return false);

  // 2) Rollout propagation on the GPU.
  launchRolloutKernel(d_params_, d_cycle_, batch_size_, d_cvx_, d_cvy_,
                      d_cwz_, d_vx_, d_vy_, d_wz_, d_x_, d_y_, d_yaws_);
  CUDA_CHECK(cudaGetLastError(), return false);

  // 3) Download the propagated velocities, clamped controls (written back by
  // the kernel, matching MotionModel::predict semantics) and trajectories.
  CUDA_CHECK(cudaMemcpy(state.vx.data(), d_vx_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(state.wz.data(), d_wz_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(state.cvx.data(), d_cvx_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(state.cwz.data(), d_cwz_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(trajectories.x.data(), d_x_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(trajectories.y.data(), d_y_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  CUDA_CHECK(cudaMemcpy(trajectories.yaws.data(), d_yaws_, mat_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  if (holonomic_) {
    CUDA_CHECK(cudaMemcpy(state.vy.data(), d_vy_, mat_bytes,
                          cudaMemcpyDeviceToHost),
               return false);
    CUDA_CHECK(cudaMemcpy(state.cvy.data(), d_cvy_, mat_bytes,
                          cudaMemcpyDeviceToHost),
               return false);
  }

  // 4) With the fail flag already set, CriticManager::evalTrajectoriesScores
  // breaks before any critic runs: only the rollout happened, the costs and
  // the flag stay untouched.
  if (fail_in) {
    return true;
  }

  // 5) Host-side gating: mirror every CriticFunction::score early-return
  // decision (scalar comparisons) and the shared path bookkeeping.
  const models::Pose &pose = state.pose;
  unsigned active = 0u;

  if (critic_enabled_[kCriticConstraint]) {
    active |= 1u << kCriticConstraint;
  }
  if (critic_enabled_[kCriticGoal] &&
      utils::withinPositionGoalTolerance(
          static_cast<float>(conf.goal_critic().threshold_to_consider()),
          pose, goal)) {
    active |= 1u << kCriticGoal;
  }
  if (critic_enabled_[kCriticGoalAngle] &&
      utils::withinPositionGoalTolerance(
          static_cast<float>(
              conf.goal_angle_critic().threshold_to_consider()),
          pose, goal)) {
    active |= 1u << kCriticGoalAngle;
  }
  // PreferForwardCritic: CPU early-returns within the goal tolerance, so it
  // only scores while AWAY from the goal.
  if (critic_enabled_[kCriticPreferForward] &&
      !utils::withinPositionGoalTolerance(
          static_cast<float>(
              conf.prefer_forward_critic().threshold_to_consider()),
          pose, goal)) {
    active |= 1u << kCriticPreferForward;
  }

  // ReferenceVelocityCritic: runtime injection wins when positive. The active
  // bit is set below, after the shared path bookkeeping: the near-start
  // suppression needs the furthest reached path point.
  float v_ref = reference_velocity_config_;
  if (ctx.runtime_reference_velocity > 0.0) {
    v_ref = static_cast<float>(ctx.runtime_reference_velocity);
  }

  // Clamp the effective reference to the reachable speed range, mirroring
  // the CPU critic: an oversized externally injected max_speed would flatten
  // the decel envelope (it only bends below vx_max very close to the goal),
  // degenerating the critic into a constant-speed tracker.
  const float vx_max =
      static_cast<float>(conf.dynamic_constraints().max_vx());
  if (vx_max > 0.0f && v_ref > vx_max) {
    v_ref = vx_max;
  }

  // TwirlingCritic: only scores WITHIN the goal tolerance (terminal heading
  // damping), like the CPU critic.
  if (critic_enabled_[kCriticTwirling] &&
      utils::withinPositionGoalTolerance(
          static_cast<float>(conf.twirling_critic().threshold_to_consider()),
          pose, goal)) {
    active |= 1u << kCriticTwirling;
  }

  // VelocityDeadbandCritic: no tolerance gate on the CPU.
  if (critic_enabled_[kCriticVelocityDeadband]) {
    active |= 1u << kCriticVelocityDeadband;
  }

  // ObstaclesCritic: no tolerance gate, but the near-goal flag suppresses
  // the repulsive term.
  int near_goal = 0;
  if (obstacles_on) {
    active |= 1u << kCriticObstacles;
    near_goal = utils::withinPositionGoalTolerance(
                    static_cast<float>(
                        conf.obstacles_critic().near_goal_distance()),
                    pose, goal)
                    ? 1
                    : 0;
  }

  // Shared path bookkeeping through the very same utils functions the CPU
  // critics use (furthest reached point, path-point validity).
  size_t furthest = 0;
  std::vector<bool> path_valid;
  {
    float model_dt = static_cast<float>(conf.optimizer_conf().model_dt());
    CriticData cdata{state,        trajectories, path, goal, costs, model_dt,
                     /*fail_flag=*/false, /*motion_model=*/nullptr,
                     /*path_pts_valid=*/std::nullopt,
                     /*furthest_reached_path_point=*/std::nullopt, &ctx};
    utils::setPathFurthestPointIfNotSet(cdata);
    furthest = *cdata.furthest_reached_path_point;
    utils::setPathCostsIfNotSet(cdata);
    path_valid = *cdata.path_pts_valid;
  }
  const int segments = static_cast<int>(furthest);

  // ReferenceVelocityCritic active bit (moved here for the near-start gate):
  // runtime v_ref wins when positive; the near-start suppression mirrors the
  // CPU critic — disabled while the robot has not advanced past
  // near_start_max_segments segments AND its heading deviates from the path
  // start orientation by more than near_start_heading_error.
  if (critic_enabled_[kCriticReferenceVelocity] && v_ref > 0.0f) {
    bool suppress = false;
    const auto &refvel_conf = conf.reference_velocity_critic();
    if (refvel_conf.near_start_disable() && path_count >= 2 &&
        furthest < static_cast<size_t>(refvel_conf.near_start_max_segments())) {
      const float heading_err = fabsf(utils::shortest_angular_distance(
          path.yaws(0), pose.yaw));
      suppress =
          heading_err > static_cast<float>(refvel_conf.near_start_heading_error());
    }
    if (!suppress) {
      active |= 1u << kCriticReferenceVelocity;
    }
  }

  // PathAlignCritic gates: CPU early-returns within the goal tolerance;
  // minimum segment count and the occupancy early-return.
  if (critic_enabled_[kCriticPathAlign] &&
      !utils::withinPositionGoalTolerance(
          static_cast<float>(
              conf.path_align_critic().threshold_to_consider()),
          pose, goal) &&
      segments > 0 &&
      static_cast<size_t>(segments) >=
          static_cast<size_t>(
              conf.path_align_critic().offset_from_furthest())) {
    const float ratio =
        static_cast<float>(
            conf.path_align_critic().max_path_occupancy_ratio());
    float invalid_ctr = 0.0f;
    const float seg_flt = static_cast<float>(segments);
    bool occupied = false;
    for (int i = 0; i < segments; ++i) {
      if (!path_valid[i]) {
        invalid_ctr += 1.0f;
      }
      if (invalid_ctr / seg_flt > ratio && invalid_ctr > 2.0f) {
        occupied = true;
        break;
      }
    }
    if (!occupied) {
      active |= 1u << kCriticPathAlign;
    }
  }

  // PathFollowCritic gate (CPU early-returns within the goal tolerance) +
  // follow target point (first valid path point at furthest + offset, walking
  // forward while blocked).
  float follow_goal_x = 0.0f, follow_goal_y = 0.0f;
  if (critic_enabled_[kCriticPathFollow] && path_count >= 2 &&
      !utils::withinPositionGoalTolerance(
          static_cast<float>(
              conf.path_follow_critic().threshold_to_consider()),
          pose, goal)) {
    const size_t path_size = static_cast<size_t>(path_count) - 1;
    size_t offsetted = std::min(
        furthest + static_cast<size_t>(
                       conf.path_follow_critic().offset_from_furthest()),
        path_size);
    bool valid = false;
    while (!valid && offsetted < path_size - 1) {
      valid = path_valid[offsetted];
      if (!valid) {
        offsetted++;
      }
    }
    follow_goal_x = path.x(offsetted);
    follow_goal_y = path.y(offsetted);
    active |= 1u << kCriticPathFollow;
  }

  // PathAngleCritic gate (CPU early-returns within the goal tolerance) +
  // target point (furthest + offset, capped at the last path point) and the
  // posePointAngle early-return.
  float angle_goal_x = 0.0f, angle_goal_y = 0.0f, angle_goal_yaw = 0.0f;
  if (critic_enabled_[kCriticPathAngle] &&
      !utils::withinPositionGoalTolerance(
          static_cast<float>(
              conf.path_angle_critic().threshold_to_consider()),
          pose, goal)) {
    const size_t offsetted = std::min(
        furthest + static_cast<size_t>(
                       conf.path_angle_critic().offset_from_furthest()),
        static_cast<size_t>(path_count) - 1);
    const float gx = path.x(offsetted);
    const float gy = path.y(offsetted);
    const float gyaw = path.yaws(offsetted);
    const float max_angle =
        static_cast<float>(
            conf.path_angle_critic().max_angle_to_furthest());
    bool gate_passed = false;
    switch (path_angle_mode_) {
    case 0: // FORWARD_PREFERENCE
      gate_passed =
          utils::posePointAngle(pose, gx, gy, true) >= max_angle;
      break;
    case 1: // NO_DIRECTIONAL_PREFERENCE
      gate_passed =
          utils::posePointAngle(pose, gx, gy, false) >= max_angle;
      break;
    default: // CONSIDER_FEASIBLE_PATH_ORIENTATIONS
      gate_passed =
          utils::posePointAngle(pose, gx, gy, gyaw) >= max_angle;
      break;
    }
    if (gate_passed) {
      angle_goal_x = gx;
      angle_goal_y = gy;
      angle_goal_yaw = gyaw;
      active |= 1u << kCriticPathAngle;
    }
  }

  cyc.critic_active = active;
  cyc.obstacles_near_goal = near_goal;
  // Start in collision: same rule as ObstaclesCritic (critic_data.h); the kernel measures the start clearance in
  // the device field it scores with. Here only the tolerance, and the host field's view for the log.
  cyc.obstacles_escape_tolerance = kStartEscapeTolerance;
  {
    float start_clearance = std::numeric_limits<float>::max();
    if (obstacles_on && ctx.obstacle_dist_func) {
      try {
        for (const auto &circle : planning::common::util::ConfigHelper::GetBoundingCircles(
                 state.pose.x, state.pose.y, state.pose.yaw)) {
          start_clearance = std::min(
              start_clearance,
              static_cast<float>(ctx.obstacle_dist_func(circle.center_x, circle.center_y) - circle.radius));
        }
      } catch (const std::exception &) {
        start_clearance = std::numeric_limits<float>::max();
      }
    }
    if (start_clearance < 0.0f) {
      AWARN_EVERY(20) << "MPPI CUDA: robot footprint in collision (clearance " << start_clearance
                      << " m), only trajectories moving away are allowed";
    }
  }
  cyc.v_ref = v_ref;
  cyc.goal_yaw_from_path = path.yaws(path_count - 1);
  cyc.path_angle_goal_x = angle_goal_x;
  cyc.path_angle_goal_y = angle_goal_y;
  cyc.path_angle_goal_yaw = angle_goal_yaw;
  cyc.path_follow_goal_x = follow_goal_x;
  cyc.path_follow_goal_y = follow_goal_y;
  cyc.path_align_segments = segments;

  // 6) Path table for PathAlignCritic: poses + arclength prefix sums,
  // truncated at the furthest reached point (utils replica).
  if (active & (1u << kCriticPathAlign)) {
    std::vector<float> table(4 * kMaxPathPoints, 0.0f);
    float *tx = table.data();
    float *ty = tx + kMaxPathPoints;
    float *tyaw = ty + kMaxPathPoints;
    float *tarc = tyaw + kMaxPathPoints;
    for (int i = 1; i < segments; ++i) {
      tx[i - 1] = path.x(i - 1);
      ty[i - 1] = path.y(i - 1);
      tyaw[i - 1] = path.yaws(i - 1);
      const float dx = path.x(i) - tx[i - 1];
      const float dy = path.y(i) - ty[i - 1];
      tarc[i] = tarc[i - 1] + sqrtf(dx * dx + dy * dy);
    }
    tx[segments - 1] = path.x(segments - 1);
    ty[segments - 1] = path.y(segments - 1);
    tyaw[segments - 1] = path.yaws(segments - 1);

    std::vector<unsigned char> valid_uc(path_count - 1);
    for (int i = 0; i < path_count - 1; ++i) {
      valid_uc[i] = path_valid[i] ? 1u : 0u;
    }
    CUDA_CHECK(cudaMemcpy(d_path_, table.data(),
                          table.size() * sizeof(float), cudaMemcpyHostToDevice),
               return false);
    CUDA_CHECK(cudaMemcpy(d_path_valid_, valid_uc.data(),
                          valid_uc.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice),
               return false);
  }

  CUDA_CHECK(cudaMemcpy(d_cycle_, &cyc, sizeof(cyc), cudaMemcpyHostToDevice),
             return false);

  // 7) Fused critic scoring. Components are zeroed first so slots of
  // critics gated off this cycle read 0 in the debug dump instead of stale
  // values from a previous cycle.
  CUDA_CHECK(cudaMemset(d_components_, 0,
                        static_cast<size_t>(kNumDeviceCritics) * batch_size_ *
                            sizeof(float)),
             return false);
  CUDA_CHECK(cudaMemcpy(d_costs_, costs.data(), vec_bytes,
                        cudaMemcpyHostToDevice),
             return false);
  launchCostKernel(d_params_, d_cycle_, batch_size_, d_vx_, d_vy_, d_wz_,
                   d_x_, d_y_, d_yaws_, d_path_, d_path_valid_,
                   d_components_, d_repulsive_, d_collision_);
  CUDA_CHECK(cudaGetLastError(), return false);
  if (active & (1u << kCriticObstacles)) {
    launchReduceKernel(batch_size_, d_repulsive_, d_collision_, d_min_rep_,
                       d_all_collide_);
    CUDA_CHECK(cudaGetLastError(), return false);
  }
  launchFinalizeKernel(d_params_, d_cycle_, batch_size_, d_costs_,
                       d_components_, d_repulsive_, d_min_rep_,
                       d_all_collide_, d_fail_);
  CUDA_CHECK(cudaGetLastError(), return false);

  // 8) Download the accumulated costs and the fail flag.
  CUDA_CHECK(cudaMemcpy(costs.data(), d_costs_, vec_bytes,
                        cudaMemcpyDeviceToHost),
             return false);
  int fail_int = 0;
  CUDA_CHECK(cudaMemcpy(&fail_int, d_fail_, sizeof(int),
                        cudaMemcpyDeviceToHost),
             return false);
  fail_flag = fail_int != 0;
  return true;
}

} // namespace cuda
} // namespace mppi
