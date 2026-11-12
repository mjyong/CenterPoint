#include "src/algorithm/mppi_planner/mppi_planner.h"

#include "src/common/util/debug_datas.h"
#include <algorithm>
#include <cmath>

namespace planning {
namespace algorithm {

using planning::common::util::normalize_angle;

MppiPlanner::MppiPlanner(
    const planning::config::MppiPlannerConf &conf,
    std::function<bool(const double, const double)> is_obstacle_func,
    std::function<double(const double, const double)> obstacle_dist_func,
    std::function<mppi::cuda::MppiCudaMapField()> cuda_map_field_func)
    : conf_(conf) {
  mppi::CriticContext ctx;
  ctx.conf = &conf_;
  ctx.is_obstacle_func = std::move(is_obstacle_func);
  ctx.obstacle_dist_func = std::move(obstacle_dist_func);
  target_speed_ = conf_.reference_velocity_critic().reference_velocity();

  optimizer_ = std::make_unique<mppi::Optimizer>();
  optimizer_->initialize(conf_, ctx);
  optimizer_->setCudaMapFieldProvider(std::move(cuda_map_field_func));

  if (conf_.enable_cuda()) {
    cuda_backend_ = std::make_unique<mppi::cuda::MppiCudaBackend>();
    if (cuda_backend_->init(conf_, optimizer_->settings(), ctx)) {
      optimizer_->setCudaBackend(cuda_backend_.get());
      AINFO << "MPPI CUDA backend attached";
    } else {
      AWARN << "MPPI CUDA backend unavailable, using the CPU path";
      cuda_backend_.reset();
    }
  }
}

MppiPlanner::~MppiPlanner() = default;

void MppiPlanner::reset() {
  optimizer_->reset();
  optimize_path_.clear();
  velocity_output_.clear();
  has_last_solver_time_ = false;
  control_period_checked_ = false;
  vx_filter_ = planning::common::util::MeanFilter();
  vy_filter_ = planning::common::util::MeanFilter();
  wz_filter_ = planning::common::util::MeanFilter();
  filter_initialized_ = false;
}

void MppiPlanner::checkControlPeriod() {
  const auto now = std::chrono::steady_clock::now();
  if (has_last_solver_time_) {
    const double elapsed =
        std::chrono::duration<double>(now - last_solver_time_).count();
    const double model_dt =
        static_cast<double>(optimizer_->settings().model_dt);
    // Shifting the control sequence is only sound if one sequence step is
    // consumed per control period.
    if (std::abs(elapsed - model_dt) > 0.2 * model_dt) {
      if (optimizer_->settings().shift_control_sequence) {
        optimizer_->setShiftControlSequence(false);
        AERROR << "MPPI solver period (" << elapsed
               << "s) differs from model_dt (" << model_dt
               << "s), control sequence shifting disabled";
      }
    }
    control_period_checked_ = true;
  }
  last_solver_time_ = now;
  has_last_solver_time_ = true;
}

bool MppiPlanner::solver(std::vector<Eigen::Vector3d> waypoints,
                         const Eigen::Vector3d &initial_pose,
                         const Eigen::Vector3d &initial_velocity) {
  if (waypoints.size() < 2) {
    // A single point carries no direction information; duplicate it so the
    // goal critics can still drive the robot to it.
    waypoints.push_back(waypoints.empty() ? Eigen::Vector3d::Zero()
                                          : waypoints.back());
  }

  mppi::models::Path path;
  path.reset(static_cast<unsigned int>(waypoints.size()));
  for (size_t i = 0; i < waypoints.size(); ++i) {
    path.x(i) = static_cast<float>(waypoints[i].x());
    path.y(i) = static_cast<float>(waypoints[i].y());
    path.yaws(i) = static_cast<float>(waypoints[i].z());
  }

  mppi::models::Pose pose;
  pose.x = static_cast<float>(initial_pose.x());
  pose.y = static_cast<float>(initial_pose.y());
  pose.yaw = static_cast<float>(initial_pose.z());

  mppi::models::Velocity speed;
  speed.vx = static_cast<float>(initial_velocity.x());
  speed.vy = static_cast<float>(initial_velocity.y());
  speed.wz = static_cast<float>(initial_velocity.z());

  {
    constexpr std::uint_fast8_t kWindowN = 5;
    if (!filter_initialized_) {
      vx_filter_ = planning::common::util::MeanFilter(kWindowN);
      vy_filter_ = planning::common::util::MeanFilter(kWindowN);
      wz_filter_ = planning::common::util::MeanFilter(kWindowN);
      filter_initialized_ = true;
    }
    filtered_velocity_.x() = vx_filter_.Update(initial_velocity.x());
    filtered_velocity_.y() = vy_filter_.Update(initial_velocity.y());
    filtered_velocity_.z() = wz_filter_.Update(initial_velocity.z());
    speed.vx = static_cast<float>(filtered_velocity_.x());
    speed.vy = static_cast<float>(filtered_velocity_.y());
    speed.wz = static_cast<float>(filtered_velocity_.z());

    auto *dbg = planning::common::util::DebugDatas::Instance();
    dbg->emplace("mppi/velocity/raw_vx", initial_velocity.x());
    dbg->emplace("mppi/velocity/raw_vy", initial_velocity.y());
    dbg->emplace("mppi/velocity/raw_wz", initial_velocity.z());
    dbg->emplace("mppi/velocity/filtered_vx", filtered_velocity_.x());
    dbg->emplace("mppi/velocity/filtered_vy", filtered_velocity_.y());
    dbg->emplace("mppi/velocity/filtered_wz", filtered_velocity_.z());
  }

  mppi::models::Pose goal;
  goal.x = static_cast<float>(waypoints.back().x());
  goal.y = static_cast<float>(waypoints.back().y());
  goal.yaw = static_cast<float>(waypoints.back().z());

  optimizer_->softReset();
  optimizer_->setRuntimeReferenceVelocity(target_speed_);

  checkControlPeriod();

  if (!optimizer_->evalControl(pose, speed, path, goal)) {
    reset();
    return false;
  }

  // Build the timed velocity output from the (post-shift) optimal control
  // sequence: index 0 is the command for the next cycle.
  const auto &seq = optimizer_->controlSequence();
  const double model_dt = static_cast<double>(optimizer_->settings().model_dt);
  velocity_output_.clear();
  velocity_output_.reserve(static_cast<size_t>(seq.vx.size()));
  for (Eigen::Index k = 0; k < seq.vx.size(); ++k) {
    velocity_output_.emplace_back(
        std::make_pair(model_dt * static_cast<double>(k),
                       Eigen::Vector3d(static_cast<double>(seq.vx(k)),
                                       static_cast<double>(seq.vy(k)),
                                       static_cast<double>(seq.wz(k)))));
  }

  // Optimized trajectory for visualization (robot frame).
  const auto trajectory = optimizer_->getOptimizedTrajectory();
  optimize_path_.clear();
  optimize_path_.reserve(static_cast<size_t>(trajectory.rows()));
  for (Eigen::Index k = 0; k < trajectory.rows(); ++k) {
    optimize_path_.emplace_back(
        Eigen::Vector3d(static_cast<double>(trajectory(k, 0)),
                        static_cast<double>(trajectory(k, 1)),
                        static_cast<double>(trajectory(k, 2))));
  }

  return true;
}

Eigen::Vector3d MppiPlanner::getVelocityCmd(double t) {
  const auto &dc = conf_.dynamic_constraints();
  if (velocity_output_.empty()) {
    return Eigen::Vector3d::Zero();
  }
  auto func_comp = [](const std::pair<double, Eigen::Vector3d> &point,
                      const double t) { return point.first < t; };

  auto it = std::lower_bound(velocity_output_.begin(), velocity_output_.end(),
                             t, func_comp);
  Eigen::Vector3d cmd;
  if (it == velocity_output_.begin()) {
    cmd = it->second;
  } else if (it == velocity_output_.end()) {
    cmd = velocity_output_.back().second;
  } else {
    auto prev = it - 1;
    double ratio = (t - prev->first) / (it->first - prev->first);
    cmd = prev->second + ratio * (it->second - prev->second);
  }

  cmd[0] = std::max(-dc.max_vx(), std::min(dc.max_vx(), cmd[0]));
  cmd[1] = std::max(-dc.max_vy(), std::min(dc.max_vy(), cmd[1]));
  cmd[2] = std::max(-dc.max_omega(), std::min(dc.max_omega(), cmd[2]));

  return cmd;
}

} // namespace algorithm
} // namespace planning
