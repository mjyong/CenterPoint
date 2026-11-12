// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/optimizer.h"

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <vector>

#include "src/algorithm/mppi_planner/tools/utils.h"
#include "src/common/util/debug_datas.h"
#include "src/common/util/log.h"

namespace mppi {

void Optimizer::initialize(
    const planning::config::MppiPlannerConf &conf, const CriticContext &ctx) {
  conf_ = &conf;
  ctx_ = ctx;

  auto &s = settings_;
  const auto &oc = conf.optimizer_conf();
  s.model_dt = static_cast<float>(oc.model_dt());
  s.time_steps = static_cast<unsigned int>(oc.time_steps());
  s.batch_size = static_cast<unsigned int>(oc.batch_size());
  s.iteration_count = static_cast<unsigned int>(oc.iteration_count());
  s.temperature = static_cast<float>(oc.temperature());
  s.gamma = static_cast<float>(oc.gamma());
  s.retry_attempt_limit = static_cast<size_t>(oc.retry_attempt_limit());

  const auto &dc = conf.dynamic_constraints();
  s.base_constraints.vx_max = static_cast<float>(dc.max_vx());
  s.base_constraints.vy = static_cast<float>(dc.max_vy());
  s.base_constraints.vx_min = static_cast<float>(dc.min_vx());
  s.base_constraints.wz = static_cast<float>(dc.max_omega());
  s.base_constraints.ax_max = static_cast<float>(dc.max_accel_vx());
  s.base_constraints.ax_min = static_cast<float>(dc.min_accel_vx());
  s.base_constraints.ay_max = static_cast<float>(dc.max_accel_vy());
  s.base_constraints.az_max = static_cast<float>(dc.max_accel_omega());

  const auto &std = conf.sampling_std();
  s.sampling_std.vx = static_cast<float>(std.vx());
  s.sampling_std.vy = static_cast<float>(std.vy());
  s.sampling_std.wz = static_cast<float>(std.wz());

  s.base_constraints.ax_max = fabs(s.base_constraints.ax_max);
  if (s.base_constraints.ax_min > 0.0f) {
    s.base_constraints.ax_min = -1.0f * s.base_constraints.ax_min;
    AWARN << "Sign of min_accel_vx is incorrect, consider setting it negative.";
  }

  s.constraints = s.base_constraints;
  setMotionModel();

  // Control sequence shifting assumes model_dt equals the control period; the
  // facade verifies this at runtime and may disable it.
  s.shift_control_sequence = true;

  critic_manager_.on_configure(ctx);
  noise_generator_.initialize(settings_, isHolonomic(), conf.regenerate_noises());

  cuda_debug_compare_ = (std::getenv("MPPI_CUDA_DEBUG_COMPARE") != nullptr);

  reset();
}

void Optimizer::shutdown() { noise_generator_.shutdown(); }

void Optimizer::setMotionModel() {
  switch (conf_->motion_model()) {
  case planning::config::MppiPlannerConf::DiffDrive:
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
    break;
  case planning::config::MppiPlannerConf::Omni:
    if (conf_->dynamic_constraints().lateral_coupling_enabled()) {
      motion_model_ = std::make_shared<OmniMotionModel>(
          static_cast<float>(conf_->dynamic_constraints().lateral_blend_vx_start()),
          static_cast<float>(conf_->dynamic_constraints().lateral_blend_vx_end()));
    } else {
      motion_model_ = std::make_shared<OmniMotionModel>();
    }
    break;
  case planning::config::MppiPlannerConf::Ackermann:
    motion_model_ = std::make_shared<AckermannMotionModel>(
        static_cast<float>(conf_->min_turning_r()));
    break;
  default:
    AERROR << "Unknown MPPI motion model, falling back to DiffDrive";
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
    break;
  }
  motion_model_->initialize(
      settings_.constraints, settings_.model_dt,
      static_cast<float>(conf_->dynamic_constraints().response_tau_vx()),
      static_cast<float>(conf_->dynamic_constraints().response_tau_vy()),
      static_cast<float>(conf_->dynamic_constraints().response_tau_wz()));
}

void Optimizer::reset() {
  state_.reset(settings_.batch_size, settings_.time_steps);
  control_sequence_.reset(settings_.time_steps);
  control_history_[0] = {0.0f, 0.0f, 0.0f};
  control_history_[1] = {0.0f, 0.0f, 0.0f};
  control_history_[2] = {0.0f, 0.0f, 0.0f};
  control_history_[3] = {0.0f, 0.0f, 0.0f};

  settings_.constraints = settings_.base_constraints;

  costs_.setZero(settings_.batch_size);
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());
  AINFO << "MPPI optimizer reset";
}

void Optimizer::softReset() {
  settings_.constraints = settings_.base_constraints;
  costs_.setZero(settings_.batch_size);
}

bool Optimizer::isHolonomic() const { return motion_model_->isHolonomic(); }

bool Optimizer::evalControl(
    const models::Pose &pose, const models::Velocity &speed,
    const models::Path &plan, const models::Pose &goal) {
  prepare(pose, speed, plan, goal);

  do {
    optimize();
  } while (fallback(critics_data_.fail_flag));

  if (critics_data_.fail_flag) {
    return false;
  }

  utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);

  if (settings_.shift_control_sequence) {
    shiftControlSequence();
  }

  logFinalTrajectoryCriticCosts();

  return true;
}

void Optimizer::optimize() {
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    if (!runCudaStep()) {
      generateNoisedTrajectories();
      critic_manager_.evalTrajectoriesScores(critics_data_);
    }
    updateControlSequence();
  }
}

bool Optimizer::runCudaStep() {
  if (cuda_backend_ == nullptr || cuda_failed_) {
    return false;
  }

  // CPU noise + control-sequence overlay, identical to the CPU path: the
  // GPU consumes the overlaid noised controls.
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();

  // Debug mode: CPU reference rollout computed from the very same overlaid
  // noised controls, compared against the GPU result after evalStep.
  models::State dbg_state;
  models::Trajectories dbg_traj;
  const bool dbg = cuda_debug_compare_;
  if (dbg) {
    dbg_state = state_;
    dbg_traj = generated_trajectories_;
    updateStateVelocities(dbg_state);
    integrateStateVelocities(dbg_traj, dbg_state);
  }

  // Device distance field from the preprocess CUDA pipeline; empty when the
  // costmap runs on CPU (evalStep then fails and we fall back for good).
  const cuda::MppiCudaMapField map =
      cuda_map_field_provider_ ? cuda_map_field_provider_()
                                : cuda::MppiCudaMapField{};
  if (!cuda_backend_->evalStep(state_, generated_trajectories_, path_, goal_,
                               ctx_, map, costs_, critics_data_.fail_flag)) {
    cuda_failed_ = true;
    AWARN << "MPPI CUDA step failed, permanently falling back to the CPU path";
    return false;
  }

  if (dbg) {
    const float dx = (generated_trajectories_.x - dbg_traj.x).abs().maxCoeff();
    const float dy = (generated_trajectories_.y - dbg_traj.y).abs().maxCoeff();
    const float dyaw =
        (generated_trajectories_.yaws - dbg_traj.yaws).abs().maxCoeff();
    const float dvx = (state_.vx - dbg_state.vx).abs().maxCoeff();
    const float dwz = (state_.wz - dbg_state.wz).abs().maxCoeff();

    // CPU critic scoring on the CPU reference trajectories, starting from the
    // same zeroed costs the GPU path started from.
    Eigen::ArrayXf cpu_costs =
        Eigen::ArrayXf::Zero(static_cast<Eigen::Index>(settings_.batch_size));
    bool cpu_fail = false;
    std::vector<std::pair<std::string, Eigen::ArrayXf>> per_critic;
    {
      CriticData cdata{dbg_state,     dbg_traj,      path_, goal_, cpu_costs,
                       settings_.model_dt, cpu_fail, motion_model_,
                       std::nullopt,  std::nullopt,  &ctx_};
      critic_manager_.evalTrajectoriesScores(cdata, &per_critic);
      cpu_fail = cdata.fail_flag;
    }
    const float dcosts = (costs_ - cpu_costs).abs().maxCoeff();
    Eigen::Index gpu_arg = 0, cpu_arg = 0;
    const float gpu_min = costs_.minCoeff(&gpu_arg);
    const float cpu_min = cpu_costs.minCoeff(&cpu_arg);
    AWARN << "MPPI CUDA debug: rollout maxdiff x=" << dx << " y=" << dy
          << " yaw=" << dyaw << " vx=" << dvx << " wz=" << dwz
          << " | costs maxdiff=" << dcosts << " min gpu=" << gpu_min << " (j="
          << gpu_arg << ") cpu=" << cpu_min << " (j=" << cpu_arg << ")"
          << " | fail gpu=" << critics_data_.fail_flag
          << " cpu=" << cpu_fail;

    // Per-critic breakdown: max over j of |cpu_contribution_j -
    // weight * gpu_raw_j| (the conclusive per-critic equivalence check),
    // plus the min values. Obstacles uses a composite formula in the
    // finalize kernel, so only the totals comparison covers it.
    static const char *kCriticSlotNames[] = {
        "ConstraintCritic",    "GoalCritic",
        "GoalAngleCritic",     "ObstaclesCritic",
        "PathAlignCritic",     "PathAngleCritic",
        "PathFollowCritic",    "PreferForwardCritic",
        "ReferenceVelocityCritic", "TwirlingCritic",
        "VelocityDeadbandCritic"};
    constexpr int kNumSlots = 11;
    std::vector<float> gpu_components;
    if (cuda_backend_->downloadDebugComponents(gpu_components)) {
      const int B = static_cast<int>(settings_.batch_size);
      for (const auto &entry : per_critic) {
        const std::string &critic_name = entry.first;
        const Eigen::ArrayXf &cpu_contrib = entry.second;
        const float cpu_min_c =
            cpu_contrib.rows() > 0 ? cpu_contrib.minCoeff() : 0.0f;
        int slot = -1;
        for (int s = 0; s < kNumSlots; ++s) {
          if (critic_name == kCriticSlotNames[s]) {
            slot = s;
            break;
          }
        }
        if (slot < 0) {
          AWARN << "  critic " << critic_name
                << ": cpu_min=" << cpu_min_c << " (no GPU slot)";
          continue;
        }
        if (slot == 3) { // ObstaclesCritic
          AWARN << "  critic " << critic_name
                << ": cpu_min=" << cpu_min_c
                << " (composite formula, see totals maxdiff)";
          continue;
        }
        const float w = cuda_backend_->criticWeight(slot);
        float gpu_raw_min = FLT_MAX;
        float md = 0.0f;
        int jm = 0;
        for (int j = 0; j < B; ++j) {
          const float g = gpu_components[slot * B + j];
          gpu_raw_min = std::min(gpu_raw_min, g);
          const float d = std::fabs(cpu_contrib(j) - w * g);
          if (d > md) {
            md = d;
            jm = j;
          }
        }
        AWARN << "  critic " << critic_name << ": maxdiff=" << md
              << " cpu_min=" << cpu_min_c << " gpu_raw_min=" << gpu_raw_min;
        if (md > 1e-2f) {
          // Divergent critic: dump the worst j plus the inputs that could
          // explain a systematic difference, and cross-check every other
          // slot's components against this CPU contribution (slot-confusion
          // detection).
          AWARN << "    DIVERGES at j=" << jm << ": cpu=" << cpu_contrib(jm)
                << " w=" << w << " g=" << gpu_components[slot * B + jm]
                << " cpu/w=" << (w != 0.0f ? cpu_contrib(jm) / w : 0.0f)
                << " | at gpu_arg j=" << gpu_arg
                << ": cpu=" << cpu_contrib(gpu_arg)
                << " g=" << gpu_components[slot * B + gpu_arg]
                << " | comps=" << gpu_components.size() << " B=" << B;
          if (slot == 8) { // ReferenceVelocityCritic inputs
            AWARN << "    v_ref runtime=" << ctx_.runtime_reference_velocity
                  << " cfg="
                  << ctx_.conf->reference_velocity_critic().reference_velocity()
                  << " decel="
                  << ctx_.conf->reference_velocity_critic().decel_start_dist()
                  << " goal=(" << goal_.x << "," << goal_.y << ")"
                  << " pose=(" << state_.pose.x << "," << state_.pose.y
                  << ")";
            // Decisive check: recompute the RefVel component on the CPU from
            // the downloaded GPU rollout (x/y/vx/vy already verified equal to
            // the CPU reference above), using the exact kernel formula. Three-
            // way comparison against the downloaded device value g and the
            // recorded CPU contribution identifies whether the kernel, the
            // download, or the CPU recording is wrong.
            const float v_ref_dbg = [&] {
              const float rt = static_cast<float>(ctx_.runtime_reference_velocity);
              return rt > 0.0f ? rt
                               : static_cast<float>(
                                     ctx_.conf->reference_velocity_critic()
                                         .reference_velocity());
            }();
            const float decel_dbg = static_cast<float>(
                ctx_.conf->reference_velocity_critic().decel_start_dist());
            const int T_dbg = static_cast<int>(settings_.time_steps);
            const float dt_dbg = static_cast<float>(settings_.model_dt);
            // Replica of the critic's exact Eigen code on the dbg (CPU) data:
            // if this disagrees with the recorded delta, the divergence is
            // inside the critic; if it agrees but disagrees with the GPU, the
            // dbg inputs differ from the GPU rollout at this j.
            Eigen::ArrayXXf speed_dbg = dbg_state.vx.abs();
            if (isHolonomic()) {
              speed_dbg = (dbg_state.vx.square() + dbg_state.vy.square()).sqrt();
            }
            Eigen::ArrayXXf dist_dbg =
                ((goal_.x - dbg_traj.x).square() +
                 (goal_.y - dbg_traj.y).square())
                    .sqrt();
            Eigen::ArrayXXf target_dbg =
                v_ref_dbg * (dist_dbg / decel_dbg).min(1.0f).sqrt();
            Eigen::ArrayXf deficit_dbg =
                ((speed_dbg - target_dbg).abs() * dt_dbg).rowwise().sum();
            const float dvy_roll =
                (state_.vy - dbg_state.vy).abs().maxCoeff();
            AWARN << "    deficit_replica(jm)=" << deficit_dbg(jm)
                  << " recorded=" << cpu_contrib(jm) / w
                  << " | replica maxdiff_vs_recorded="
                  << (deficit_dbg - cpu_contrib / w).abs().maxCoeff()
                  << " | rollout dvy=" << dvy_roll;
            for (int jj : {jm, static_cast<int>(gpu_arg)}) {
              float recomp = 0.0f;
              for (int t = 0; t < T_dbg; ++t) {
                const float svx = state_.vx.coeff(t * B + jj);
                const float svy = state_.vy.coeff(t * B + jj);
                const float speed = std::sqrt(svx * svx + svy * svy);
                const float dx = goal_.x - generated_trajectories_.x.coeff(t * B + jj);
                const float dy = goal_.y - generated_trajectories_.y.coeff(t * B + jj);
                const float dist = std::sqrt(dx * dx + dy * dy);
                const float target =
                    v_ref_dbg * std::sqrt(std::min(dist / decel_dbg, 1.0f));
                recomp += std::fabs(speed - target) * dt_dbg;
              }
              AWARN << "    recompute j=" << jj << ": cpu_formula=" << recomp
                    << " device_g=" << gpu_components[8 * B + jj]
                    << " cpu_recorded/w="
                    << (w != 0.0f ? cpu_contrib(jj) / w : 0.0f);
            }
          }
          for (int s = 0; s < kNumSlots; ++s) {
            if (s == slot) {
              continue;
            }
            const float ws = cuda_backend_->criticWeight(s);
            if (ws == 0.0f) {
              continue;
            }
            float mds = 0.0f;
            for (int j = 0; j < B; ++j) {
              mds = std::max(mds, std::fabs(cpu_contrib(j) -
                                            ws * gpu_components[s * B + j]));
            }
            if (mds < 1e-2f) {
              AWARN << "    -> matches slot " << s << " ("
                    << kCriticSlotNames[s] << ") md=" << mds;
            }
          }
        }
      }
    }
  }
  return true;
}

bool Optimizer::fallback(bool fail) {
  if (!fail) {
    fallback_counter_ = 0;
    return false;
  }

  reset();

  if (++fallback_counter_ > settings_.retry_attempt_limit) {
    fallback_counter_ = 0;
    AERROR << "MPPI optimizer failed to compute a collision free path";
    return false;
  }

  return true;
}

void Optimizer::prepare(
    const models::Pose &pose, const models::Velocity &speed,
    const models::Path &plan, const models::Pose &goal) {
  state_.pose = pose;
  state_.speed = speed;
  path_ = plan;
  costs_.setZero();
  goal_ = goal;

  critics_data_.fail_flag = false;
  critics_data_.motion_model = motion_model_;
  critics_data_.ctx = &ctx_;
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
}

void Optimizer::shiftControlSequence() {
  auto size = control_sequence_.vx.size();
  utils::shiftColumnsByOnePlace(control_sequence_.vx, -1);
  utils::shiftColumnsByOnePlace(control_sequence_.wz, -1);
  control_sequence_.vx(size - 1) = control_sequence_.vx(size - 2);
  control_sequence_.wz(size - 1) = control_sequence_.wz(size - 2);

  if (isHolonomic()) {
    utils::shiftColumnsByOnePlace(control_sequence_.vy, -1);
    control_sequence_.vy(size - 1) = control_sequence_.vy(size - 2);
  }
}

void Optimizer::generateNoisedTrajectories() {
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
}

void Optimizer::applyControlSequenceConstraints() {
  motion_model_->constrainCommandSequence(control_sequence_, state_.speed);
}

void Optimizer::updateStateVelocities(models::State &state) const {
  updateInitialStateVelocities(state);
  propagateStateVelocitiesFromInitials(state);
}

void Optimizer::updateInitialStateVelocities(models::State &state) const {
  state.vx.col(0) = state.speed.vx;
  state.wz.col(0) = state.speed.wz;

  if (isHolonomic()) {
    state.vy.col(0) = state.speed.vy;
  }
}

void Optimizer::propagateStateVelocitiesFromInitials(models::State &state) const {
  motion_model_->predict(state);
}

void Optimizer::integrateStateVelocities(
    models::Trajectories &trajectories, const models::State &state) const {
  const float initial_yaw = state.pose.yaw;
  const unsigned int n_cols = trajectories.yaws.cols();

  trajectories.yaws.col(0) = state.wz.col(0) * settings_.model_dt + initial_yaw;
  for (unsigned int i = 1; i != n_cols; i++) {
    trajectories.yaws.col(i) =
        trajectories.yaws.col(i - 1) + state.wz.col(i) * settings_.model_dt;
  }

  Eigen::ArrayXXf yaw_cos = trajectories.yaws.cos();
  Eigen::ArrayXXf yaw_sin = trajectories.yaws.sin();
  utils::shiftColumnsByOnePlace(yaw_cos, 1);
  utils::shiftColumnsByOnePlace(yaw_sin, 1);
  yaw_cos.col(0) = cosf(initial_yaw);
  yaw_sin.col(0) = sinf(initial_yaw);

  auto dx = (state.vx * yaw_cos).eval();
  auto dy = (state.vx * yaw_sin).eval();

  if (isHolonomic()) {
    dx = dx - state.vy * yaw_sin;
    dy = dy + state.vy * yaw_cos;
  }

  trajectories.x.col(0) = dx.col(0) * settings_.model_dt + state.pose.x;
  trajectories.y.col(0) = dy.col(0) * settings_.model_dt + state.pose.y;
  for (unsigned int i = 1; i != n_cols; i++) {
    trajectories.x.col(i) =
        trajectories.x.col(i - 1) + dx.col(i) * settings_.model_dt;
    trajectories.y.col(i) =
        trajectories.y.col(i - 1) + dy.col(i) * settings_.model_dt;
  }
}

Eigen::ArrayXXf Optimizer::getOptimizedTrajectory() {
  const bool is_holo = isHolonomic();
  models::State state;
  state.reset(1, settings_.time_steps);
  state.pose = state_.pose;
  state.speed = state_.speed;
  state.cvx.row(0) = control_sequence_.vx.transpose();
  state.cwz.row(0) = control_sequence_.wz.transpose();
  if (is_holo) {
    state.cvy.row(0) = control_sequence_.vy.transpose();
  }
  updateStateVelocities(state);

  models::Trajectories trajectories;
  trajectories.reset(1, settings_.time_steps);
  integrateStateVelocities(trajectories, state);

  Eigen::ArrayXXf out(settings_.time_steps, 3);
  out.col(0) = trajectories.x.row(0).transpose();
  out.col(1) = trajectories.y.row(0).transpose();
  out.col(2) = trajectories.yaws.row(0).transpose();
  return out;
}

void Optimizer::logFinalTrajectoryCriticCosts() {
  // Batch-1 rollout of the final control sequence (post Savitzky-Golay
  // filtering and shifting), i.e. the trajectory that will be executed.
  const bool is_holo = isHolonomic();
  models::State state;
  state.reset(1, settings_.time_steps);
  state.pose = state_.pose;
  state.speed = state_.speed;
  state.cvx.row(0) = control_sequence_.vx.transpose();
  state.cwz.row(0) = control_sequence_.wz.transpose();
  if (is_holo) {
    state.cvy.row(0) = control_sequence_.vy.transpose();
  }
  updateStateVelocities(state);

  models::Trajectories trajectories;
  trajectories.reset(1, settings_.time_steps);
  integrateStateVelocities(trajectories, state);

  Eigen::ArrayXf costs = Eigen::ArrayXf::Zero(1);
  bool fail_flag = false;
  std::vector<std::pair<std::string, Eigen::ArrayXf>> per_critic;
  CriticData data{state, trajectories, path_, goal_, costs,
                  settings_.model_dt, fail_flag, motion_model_,
                  std::nullopt, std::nullopt, &ctx_};
  critic_manager_.evalTrajectoriesScores(data, &per_critic);

  float total = 0.0f;
  for (const auto &entry : per_critic) {
    const float cost = entry.second(0);
    total += cost;
    AINFO << "MPPI final trajectory critic: " << entry.first << " = " << cost;
    planning::common::util::DebugDatas::Instance()->emplace(
        "mppi/critic/" + entry.first, cost);
  }
  planning::common::util::DebugDatas::Instance()->emplace(
      "mppi/critic/total", total);
}

void Optimizer::updateControlSequence() {
  const bool is_holo = isHolonomic();
  auto &s = settings_;
  auto bounded_noises_vx = state_.cvx.rowwise() - control_sequence_.vx.transpose();
  auto bounded_noises_wz = state_.cwz.rowwise() - control_sequence_.wz.transpose();
  costs_ += (s.gamma / powf(s.sampling_std.vx, 2) *
             (bounded_noises_vx.rowwise() * control_sequence_.vx.transpose())
                 .rowwise()
                 .sum())
                .eval();
  costs_ += (s.gamma / powf(s.sampling_std.wz, 2) *
             (bounded_noises_wz.rowwise() * control_sequence_.wz.transpose())
                 .rowwise()
                 .sum())
                .eval();
  if (is_holo) {
    auto bounded_noises_vy = state_.cvy.rowwise() - control_sequence_.vy.transpose();
    costs_ += (s.gamma / powf(s.sampling_std.vy, 2) *
               (bounded_noises_vy.rowwise() * control_sequence_.vy.transpose())
                   .rowwise()
                   .sum())
                  .eval();
  }

  auto costs_normalized = costs_ - costs_.minCoeff();
  auto exponents = ((-1 / settings_.temperature * costs_normalized).exp()).eval();
  auto softmaxes = (exponents / exponents.sum()).eval();

  control_sequence_.vx = (state_.cvx.colwise() * softmaxes).colwise().sum();
  control_sequence_.wz = (state_.cwz.colwise() * softmaxes).colwise().sum();
  if (is_holo) {
    control_sequence_.vy = (state_.cvy.colwise() * softmaxes).colwise().sum();
  }

  applyControlSequenceConstraints();
}

models::Control Optimizer::getControlFromSequence() const {
  return {
      control_sequence_.vx(0), control_sequence_.vy(0),
      control_sequence_.wz(0)};
}

} // namespace mppi
