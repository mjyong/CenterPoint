#include "src/algorithm/teb_planner/teb_planner.h"
#include "src/algorithm/teb_planner/coarse_path_search.h"

namespace teb_planner {

TebPlanner::TebPlanner(
    const planning::config::TebPlannerConf &conf,
    std::function<double(const double, const double)> is_obstacle_func,
    std::function<double(const double, const double)> obstacle_dist_func,
    std::function<double(const double, const double)> composite_cost_func,
    std::function<bool(const double, const double, const double)>
        is_collision_func)
    : conf_(conf), is_obstacle_func_(is_obstacle_func),
      obstacle_dist_func_(obstacle_dist_func),
      composite_cost_func_(composite_cost_func),
      is_collision_func_(is_collision_func),
      coarse_path_search_(is_collision_func, composite_cost_func,
                          conf.coarse_search_conf()),
      filter_vx_(0.02, 10.0), filter_vy_(0.02, 10.0), filter_omega_(0.02, 10.0),
      filter_input_vx_(0.02, 10.0), filter_input_vy_(0.02, 10.0),
      filter_input_omega_(0.02, 10.0) {}

TebPlanner::~TebPlanner() {}

bool TebPlanner::solver(std::vector<Eigen::Vector3d> waypoints,
                        const Eigen::Vector3d &initial_pose,
                        const Eigen::Vector3d &initial_velocity,
                        Eigen::Vector3d final_pose, bool constant_final_theta,
                        const planning::Transform3d &global_pose) {
  Eigen::Vector3d filtered_velocity(
      filter_input_vx_.filter(initial_velocity[0]),
      filter_input_vy_.filter(initial_velocity[1]),
      filter_input_omega_.filter(initial_velocity[2]));

  if (is_collision_func_(initial_pose[0], initial_pose[1], initial_pose[2])) {
    AERROR << "Initial pose is in collision";
    return false;
  }
  if (waypoints.size() < 2) {
    AERROR << "Input ref point size < 2";
    waypoints.emplace_back(initial_pose);
  }
  optimize_path_.clear();
  velocity_output_.clear();
  coarse_path_.clear();
  coarse_path_.emplace_back(initial_pose);

  // Transform prev_optimize_path_ from previous frame's local frame
  // to current frame's local frame
  std::vector<Eigen::Vector3d> transformed_prev_path;
  if (has_prev_global_pose_ && !prev_optimize_path_.empty()) {
    // relative transform: prev_local -> current_local
    planning::Transform3d rel_tf = global_pose.inverse() * prev_global_pose_;
    double dyaw = std::atan2(rel_tf.rotation()(1, 0), rel_tf.rotation()(0, 0));
    for (const auto &pt : prev_optimize_path_) {
      Eigen::Vector3d transformed = rel_tf * Eigen::Vector3d(pt.x(), pt.y(), 0.0);
      transformed_prev_path.emplace_back(
          Eigen::Vector3d(transformed.x(), transformed.y(),
                          normalize_angle(pt.z() + dyaw)));
    }
  }
  coarse_path_search_.setLastSuccessPath(transformed_prev_path);
  coarse_path_search_.setReferencePath(waypoints);
  if (!coarse_path_search_.plan(initial_pose, final_pose,
                                constant_final_theta)) {
    int index = waypoints.size() - 1;
    while (is_collision_func_(final_pose[0], final_pose[1], final_pose[2]) &&
           index >= 0) {
      index -= 3;
      final_pose = waypoints[index];
    }
    if (index <= 0) {
      AERROR << "Final pose is in collision";
      return false;
    }
    AWARN << "Coarse path search failed, try to find path by reverse search, "
             "constant_final_theta : "
          << constant_final_theta;
    coarse_path_search_.clearReferencePath();
    if (coarse_path_search_.plan(final_pose, initial_pose, true)) {
      coarse_path_ = coarse_path_search_.getPath();
      std::reverse(coarse_path_.begin(), coarse_path_.end());
    } else {
      AERROR << "reverse search failed";
      return false;
    }
  } else {
    coarse_path_ = coarse_path_search_.getPath();
  }
  if (coarse_path_.size() < 2) {
    AERROR << "Coarse path size < 2";
    return false;
  }
  AINFO << "coarse_path size: " << coarse_path_.size();
  std::vector<Eigen::Vector3d> sampled_path;
  for (size_t i = 0; i < coarse_path_.size(); i++) {
    if (i <= 10) {
      sampled_path.emplace_back(coarse_path_[i]);
    } else if (i <= 40 && (i % 2 == 0)) {
      sampled_path.emplace_back(coarse_path_[i]);
    } else if (i <= 60 && (i % 3 == 0)) {
      sampled_path.emplace_back(coarse_path_[i]);
    }
  }
  // if (constant_final_theta) {
  //   sampled_path.emplace_back(final_pose);
  // }
  coarse_path_ = std::move(sampled_path);
  AINFO << "coarse_path resampled size: " << coarse_path_.size();

  int num_poses = coarse_path_.size();
  x_.resize(num_poses);
  y_.resize(num_poses);
  theta_.resize(num_poses);
  time_deltas_.resize(num_poses - 1);
  for (int i = 0; i < num_poses; ++i) {
    x_[i] = coarse_path_[i][0];
    y_[i] = coarse_path_[i][1];
    theta_[i] = coarse_path_[i][2];
  }
  computeReferenceVelocities(filtered_velocity);
  buildProblem(filtered_velocity);

  const auto &oc = conf_.optimizer_conf();
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = oc.max_num_iterations();
  options.function_tolerance = oc.function_tolerance();
  options.gradient_tolerance = oc.gradient_tolerance();
  options.parameter_tolerance = oc.parameter_tolerance();
  options.num_threads = oc.num_threads();
  // options.dense_linear_algebra_library_type = ceres::CUDA;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem_.get(), &summary);

  if (summary.IsSolutionUsable()) {
    optimize_path_.clear();
    optimize_path_.emplace_back(Eigen::Vector3d(x_[0], y_[0], theta_[0]));
    velocity_output_.clear();
    velocity_output_.emplace_back(std::make_pair(0.0, filtered_velocity));
    for (int i = 1; i < static_cast<int>(x_.size()); ++i) {
      optimize_path_.emplace_back(Eigen::Vector3d(x_[i], y_[i], theta_[i]));
      double vx = 0.0;
      double vy = 0.0;
      double omega = 0.0;

      computeRobotVelocity(x_[i - 1], y_[i - 1], theta_[i - 1], x_[i], y_[i],
                           theta_[i], time_deltas_[i - 1], vx, vy, omega);
      double time = velocity_output_.back().first + time_deltas_[i - 1];
      velocity_output_.emplace_back(
          std::make_pair(time, Eigen::Vector3d(vx, vy, omega)));
    }
    prev_optimize_path_ = optimize_path_;
    prev_global_pose_ = global_pose;
    has_prev_global_pose_ = true;
  }
  return summary.IsSolutionUsable();
}

Eigen::Vector3d TebPlanner::getVelocityCmd(double t) {
  const auto &dc = conf_.dynamic_constraints();
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

  cmd[0] = filter_vx_.filter(cmd[0]);
  cmd[1] = filter_vy_.filter(cmd[1]);
  cmd[2] = filter_omega_.filter(cmd[2]);

  return cmd;
}

void TebPlanner::buildProblem(const Eigen::Vector3d &initial_velocity) {
  const auto &wc = conf_.weight_conf();
  const auto &dc = conf_.dynamic_constraints();
  const auto &tc = conf_.time_conf();

  int num_poses = x_.size();
  problem_.reset(new ceres::Problem);
  for (int i = 0; i < num_poses; ++i) {
    problem_->AddParameterBlock(&x_[i], 1);
    problem_->AddParameterBlock(&y_[i], 1);
    problem_->AddParameterBlock(&theta_[i], 1);
#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)
    problem_->SetManifold(&theta_[i],
                          new AngleLocalParameterization()); // ceres 2.2
#else
    problem_->SetParameterization(&theta_[i], new AngleLocalParameterization());
#endif
  }

  problem_->SetParameterBlockConstant(&x_[0]);
  problem_->SetParameterBlockConstant(&y_[0]);
  problem_->SetParameterBlockConstant(&theta_[0]);
  problem_->SetParameterBlockConstant(&x_.back());
  problem_->SetParameterBlockConstant(&y_.back());
  problem_->SetParameterBlockConstant(&theta_.back());
  problem_->AddParameterBlock(&time_deltas_[0], 1);
  for (int i = 0; i < num_poses - 1; ++i) {
    problem_->AddParameterBlock(&time_deltas_[i], 1);
    problem_->SetParameterLowerBound(&time_deltas_[i], 0,
                                     tc.time_delta_lower_bound());
    problem_->SetParameterUpperBound(&time_deltas_[i], 0,
                                     tc.time_delta_upper_bound());
  }

  if (!coarse_path_.empty() &&
      coarse_path_.size() == static_cast<size_t>(num_poses)) {
    for (int i = 1; i < num_poses - 1; ++i) {
      ceres::CostFunction *path_following_cost =
          new ceres::AutoDiffCostFunction<PathFollowingCost, 3, 1, 1, 1>(
              new PathFollowingCost(coarse_path_[i][0], coarse_path_[i][1],
                                    coarse_path_[i][2],
                                    wc.weight_path_following()));
      problem_->AddResidualBlock(path_following_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i]);
    }
  }

  if (num_poses >= 2) {
    ceres::CostFunction *initial_vel_cost =
        new ceres::AutoDiffCostFunction<InitialVelocityCost, 3, 1, 1, 1, 1, 1,
                                        1, 1>(new InitialVelocityCost(
            initial_velocity[0], initial_velocity[1], initial_velocity[2],
            wc.weight_initial_velocity()));
    problem_->AddResidualBlock(initial_vel_cost, nullptr, &x_[0], &y_[0],
                               &theta_[0], &x_[1], &y_[1], &theta_[1],
                               &time_deltas_[0]);
  }

  if (num_poses >= 2) {
    ceres::CostFunction *final_vel_cost =
        new ceres::AutoDiffCostFunction<FinalVelocityCost, 3, 1, 1, 1, 1, 1, 1,
                                        1>(
            new FinalVelocityCost(wc.weight_final_velocity()));
    problem_->AddResidualBlock(
        final_vel_cost, nullptr, &x_[num_poses - 2], &y_[num_poses - 2],
        &theta_[num_poses - 2], &x_[num_poses - 1], &y_[num_poses - 1],
        &theta_[num_poses - 1], &time_deltas_[num_poses - 2]);
  }

  for (int i = 0; i < num_poses - 2; ++i) {
    ceres::CostFunction *accel_limit_cost = new AccelerationLimitAnalyticCost(
        dc.max_accel_vx(), dc.max_accel_vy(), dc.max_accel_omega(),
        wc.weight_accel_smooth(), wc.weight_accel_limit());
    problem_->AddResidualBlock(
        accel_limit_cost, nullptr, &x_[i], &y_[i], &theta_[i], &x_[i + 1],
        &y_[i + 1], &theta_[i + 1], &time_deltas_[i], &x_[i + 2], &y_[i + 2],
        &theta_[i + 2], &time_deltas_[i + 1]);
  }

  for (int i = 0; i < num_poses - 1; ++i) {
    ceres::CostFunction *velocity_limit_cost = new VelocityLimitAnalyticCost(
        dc.max_vx(), dc.max_vy(), dc.max_omega(), wc.weight_velocity_limit());
    problem_->AddResidualBlock(velocity_limit_cost, nullptr, &x_[i], &y_[i],
                               &theta_[i], &x_[i + 1], &y_[i + 1],
                               &theta_[i + 1], &time_deltas_[i]);

    if (!ref_velocities_.empty() && ref_velocities_[i] > 0.0) {
      ceres::CostFunction *ref_vel_cost =
          new ceres::AutoDiffCostFunction<ReferenceVelocityCost, 1, 1, 1, 1, 1,
                                          1, 1, 1>(new ReferenceVelocityCost(
              ref_velocities_[i], wc.weight_reference_velocity()));
      problem_->AddResidualBlock(ref_vel_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1],
                                 &theta_[i + 1], &time_deltas_[i]);
    }

    double ref_vx_i = (!ref_velocities_.empty() && ref_velocities_[i] > 0.0)
                          ? ref_velocities_[i]
                          : dc.max_vx();
    if (ref_vx_i < wc.non_holonomic_min_speed()) {
      ceres::CostFunction *lateral_vel_cost = new LateralVelocityAnalyticCost(
          wc.weight_lateral_velocity(), ref_vx_i);
      problem_->AddResidualBlock(lateral_vel_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1],
                                 &theta_[i + 1], &time_deltas_[i]);
    }

    // double nh_weight = (ref_vx_i > wc.non_holonomic_min_speed())
    //                        ? wc.weight_non_holonomic()
    //                        : 0.0;
    if (ref_vx_i > wc.non_holonomic_min_speed()) {
      ceres::CostFunction *non_holonomic_cost =
          new NonHolonomicCost(wc.weight_non_holonomic());
      problem_->AddResidualBlock(non_holonomic_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1]);
    }

    if (ref_vx_i > wc.non_holonomic_min_speed()) {
      ceres::CostFunction *prefer_forward_cost =
          new PreferForwardCost(wc.weight_prefer_forward());
      problem_->AddResidualBlock(prefer_forward_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1]);
    }

    ceres::CostFunction *smoothness_cost =
        new ceres::AutoDiffCostFunction<PathSmoothnessCost, 3, 1, 1, 1, 1, 1,
                                        1>(
            new PathSmoothnessCost(wc.weight_path_smoothness()));
    problem_->AddResidualBlock(smoothness_cost, nullptr, &x_[i], &y_[i],
                               &theta_[i], &x_[i + 1], &y_[i + 1],
                               &theta_[i + 1]);

    ceres::CostFunction *time_cost =
        new ceres::AutoDiffCostFunction<TimeOptimalCost, 1, 1>(
            new TimeOptimalCost(wc.weight_time_optimal()));
    problem_->AddResidualBlock(time_cost, nullptr, &time_deltas_[i]);
  }

  for (int i = 1; i < num_poses - 1; ++i) {
    ceres::CostFunction *obstacle_cost = new ObstacleCost(
        obstacle_dist_func_, is_collision_func_,
        wc.weight_obstacle_warning(), wc.weight_obstacle_danger(),
        wc.obstacle_warning_distance(), wc.obstacle_danger_distance());
    problem_->AddResidualBlock(obstacle_cost, nullptr, &x_[i], &y_[i],
                               &theta_[i]);
  }
}

void TebPlanner::computeReferenceVelocities(
    const Eigen::Vector3d &initial_velocity) {
  const auto &dc = conf_.dynamic_constraints();
  int num_poses = coarse_path_.size();
  ref_velocities_.resize(num_poses - 1, 0.0);

  if (num_poses < 2) {
    return;
  }

  double max_v = dc.max_vx();
  double max_a = dc.max_accel_vx() * 0.2;
  double v0 = std::abs(initial_velocity[0]);

  std::vector<double> arc_length(num_poses, 0.0);
  for (int i = 1; i < num_poses; ++i) {
    double dx = coarse_path_[i][0] - coarse_path_[i - 1][0];
    double dy = coarse_path_[i][1] - coarse_path_[i - 1][1];
    arc_length[i] = arc_length[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  double total_length = arc_length.back();
  double v_end = total_length < 2.0 ? 0.0 : dc.reference_velocity();
  if (total_length < 1e-6 || max_a < 1e-6) {
    for (int i = 0; i < num_poses - 1; ++i) {
      ref_velocities_[i] =
          dc.reference_velocity() > 0.0 ? dc.reference_velocity() : max_v;
    }
    return;
  }

  std::vector<double> fwd_vel(num_poses, 0.0);
  fwd_vel[0] = v0;
  for (int i = 1; i < num_poses; ++i) {
    double ds = arc_length[i] - arc_length[i - 1];
    double v_reach =
        std::sqrt(fwd_vel[i - 1] * fwd_vel[i - 1] + 2.0 * max_a * ds);
    fwd_vel[i] = std::min(v_reach, dc.reference_velocity());
  }

  std::vector<double> bwd_vel(num_poses, 0.0);
  bwd_vel[num_poses - 1] = v_end;
  for (int i = num_poses - 2; i >= 0; --i) {
    double ds = arc_length[i + 1] - arc_length[i];
    double v_reach =
        std::sqrt(bwd_vel[i + 1] * bwd_vel[i + 1] + 2.0 * max_a * ds);
    bwd_vel[i] = std::min(v_reach, dc.reference_velocity());
  }

  for (int i = 0; i < num_poses; ++i) {
    double v = std::min(fwd_vel[i], bwd_vel[i]);
    if (i < num_poses - 1) {
      ref_velocities_[i] = v;
    }
  }

  const auto &tc = conf_.time_conf();
  double dt_lower = tc.time_delta_lower_bound();
  double dt_upper = tc.time_delta_upper_bound();
  for (int i = 0; i < num_poses - 1; ++i) {
    double ds = arc_length[i + 1] - arc_length[i];
    double v = std::max(ref_velocities_[i], 0.01);
    double dt = ds / v;
    time_deltas_[i] = std::max(dt_lower, std::min(dt_upper, dt));
  }
}

} // namespace teb_planner