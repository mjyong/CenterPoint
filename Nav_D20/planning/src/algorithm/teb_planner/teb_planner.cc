#include "src/algorithm/teb_planner/teb_planner.h"
#include "src/algorithm/teb_planner/coarse_path_search.h"
#include <limits>

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
                        const planning::Transform3d &global_pose,
                        const bool &stop_at_end) {
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

  constexpr bool kUseWarmStart = true;
  std::vector<Eigen::Vector3d> transformed_prev_path;
  if (kUseWarmStart && has_prev_global_pose_ && !prev_optimize_path_.empty()) {
    // relative transform: prev_local -> current_local
    planning::Transform3d rel_tf = global_pose.inverse() * prev_global_pose_;
    double dyaw = std::atan2(rel_tf.rotation()(1, 0), rel_tf.rotation()(0, 0));
    for (const auto &pt : prev_optimize_path_) {
      Eigen::Vector3d transformed =
          rel_tf * Eigen::Vector3d(pt.x(), pt.y(), 0.0);
      transformed_prev_path.emplace_back(Eigen::Vector3d(
          transformed.x(), transformed.y(), normalize_angle(pt.z() + dyaw)));
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
  for (size_t i = 0; i < coarse_path_.size(); i++)
  {
    AINFO << "index: " << i << ", x: " << coarse_path_[i][0] << ", y: " << coarse_path_[i][1] << ", theta: " << coarse_path_[i][2];
  }
  

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
  // Warm start
  if (transformed_prev_path.size() >= 2) {
    auto cumulativeArc = [](const std::vector<Eigen::Vector3d> &path) {
      std::vector<double> s(path.size(), 0.0);
      for (size_t k = 1; k < path.size(); ++k) {
        s[k] = s[k - 1] +
               (path[k].head<2>() - path[k - 1].head<2>()).norm();
      }
      return s;
    };
    const std::vector<double> s_prev = cumulativeArc(transformed_prev_path);
    const std::vector<double> s_new = cumulativeArc(coarse_path_);
    const double end_drift = (transformed_prev_path.back().head<2>() -
                              coarse_path_.back().head<2>())
                                 .norm();
    if (s_prev.back() > 1e-3 && s_new.back() > 1e-3 && end_drift < 2.0) {
      size_t j0 = 0;
      double d_min = std::numeric_limits<double>::max();
      for (size_t k = 0; k < transformed_prev_path.size(); ++k) {
        const double d = transformed_prev_path[k].head<2>().squaredNorm();
        if (d < d_min) {
          d_min = d;
          j0 = k;
        }
      }
      const double s_offset = s_prev[j0];
      size_t prev_k = j0;
      for (int i = 1; i < num_poses - 1; ++i) {
        const double s_target = s_offset + s_new[i];
        if (s_target >= s_prev.back()) {
          break;
        }
        while (prev_k + 2 < s_prev.size() && s_prev[prev_k + 1] < s_target) {
          ++prev_k;
        }
        const double seg = s_prev[prev_k + 1] - s_prev[prev_k];
        const double ratio =
            seg > 1e-9
                ? std::max(0.0, std::min(1.0,
                                         (s_target - s_prev[prev_k]) / seg))
                : 0.0;
        const Eigen::Vector3d &p0 = transformed_prev_path[prev_k];
        const Eigen::Vector3d &p1 = transformed_prev_path[prev_k + 1];
        const double warm_x = p0[0] + ratio * (p1[0] - p0[0]);
        const double warm_y = p0[1] + ratio * (p1[1] - p0[1]);

        const double anchor_theta = coarse_path_[i][2];
        const double anchor_dev =
            std::hypot(warm_x - coarse_path_[i][0],
                       warm_y - coarse_path_[i][1]);
        if (warm_x > 0.02 && anchor_dev < 0.3 &&
            !is_collision_func_(warm_x, warm_y, anchor_theta)) {
          x_[i] = warm_x;
          y_[i] = warm_y;
          theta_[i] = anchor_theta;
          // theta_[i] intentionally NOT seeded: keeps anchor theta.
        }
      }
    }
  }
  computeReferenceVelocities(filtered_velocity, stop_at_end);
  buildProblem(filtered_velocity, stop_at_end);

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
    AINFO << "t: 0, x: " << x_[0] << ", y: " << y_[0] << ", theta: " << theta_[0]
          << ", vx: " << filtered_velocity[0] << ", vy: " << filtered_velocity[1] << ", omega: " << filtered_velocity[2];
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
      AINFO << "t: " << time << ", x: " << x_[i] << ", y: " << y_[i] << ", theta: " << theta_[i]
            << ", vx: " << vx << ", vy: " << vy << ", omega: " << omega;
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

void TebPlanner::buildProblem(const Eigen::Vector3d &initial_velocity,
                              bool stop_at_end) {
  const auto &wc = conf_.weight_conf();
  const auto &dc = conf_.dynamic_constraints();
  const auto &tc = conf_.time_conf();

  int num_poses = x_.size();
  problem_.reset(new ceres::Problem);
  for (int i = 0; i < num_poses; ++i) {
    problem_->AddParameterBlock(&x_[i], 1);
    problem_->AddParameterBlock(&y_[i], 1);
    problem_->AddParameterBlock(&theta_[i], 1);
#ifdef ROS2_JAZZY
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

    ceres::CostFunction *initial_accel_cost =
        new ceres::AutoDiffCostFunction<InitialAccelLimitCost, 1, 1, 1, 1, 1,
                                        1>(new InitialAccelLimitCost(
            std::hypot(initial_velocity[0], initial_velocity[1]),
            dc.max_accel_vx(), wc.weight_accel_limit()));
    problem_->AddResidualBlock(initial_accel_cost, nullptr, &x_[0], &y_[0],
                               &x_[1], &y_[1], &time_deltas_[0]);
  }

  if (stop_at_end && num_poses >= 2) {
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

    if (!ref_velocities_.empty()) {
      ceres::CostFunction *ref_vel_cost =
          new ceres::AutoDiffCostFunction<ReferenceVelocityCost, 1, 1, 1, 1, 1,
                                          1, 1, 1>(new ReferenceVelocityCost(
              ref_velocities_[i], wc.weight_reference_velocity()));
      problem_->AddResidualBlock(ref_vel_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1],
                                 &theta_[i + 1], &time_deltas_[i]);
    }

    double ref_vx_i =
        (!ref_velocities_.empty()) ? ref_velocities_[i] : dc.max_vx();
    if (ref_vx_i < wc.non_holonomic_min_speed()) {
      ceres::CostFunction *lateral_vel_cost = new LateralVelocityAnalyticCost(
          wc.weight_lateral_velocity(), ref_vx_i);
      problem_->AddResidualBlock(lateral_vel_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1],
                                 &theta_[i + 1], &time_deltas_[i]);
    }

    if (ref_vx_i > wc.non_holonomic_min_speed()) {
      ceres::CostFunction *non_holonomic_cost =
          new NonHolonomicCost(wc.weight_non_holonomic());
      problem_->AddResidualBlock(non_holonomic_cost, nullptr, &x_[i], &y_[i],
                                 &theta_[i], &x_[i + 1], &y_[i + 1]);
    }

    ceres::CostFunction *prefer_forward_cost =
        new PreferForwardCost(wc.weight_prefer_forward());
    problem_->AddResidualBlock(prefer_forward_cost, nullptr, &x_[i], &y_[i],
                               &theta_[i], &x_[i + 1], &y_[i + 1]);

    const double rest_i =
        (coarse_path_[i + 1].head<2>() - coarse_path_[i].head<2>()).norm();
    ceres::CostFunction *smoothness_cost =
        new ceres::AutoDiffCostFunction<PathSmoothnessCost, 3, 1, 1, 1, 1, 1,
                                        1>(
            new PathSmoothnessCost(wc.weight_path_smoothness(), rest_i));
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
        obstacle_dist_func_, wc.weight_obstacle_warning(),
        wc.weight_obstacle_danger(), wc.obstacle_warning_distance(),
        wc.obstacle_danger_distance());
    problem_->AddResidualBlock(obstacle_cost, nullptr, &x_[i], &y_[i],
                               &theta_[i]);
  }
}

void TebPlanner::computeReferenceVelocities(
    const Eigen::Vector3d &initial_velocity, bool stop_at_end) {
  const auto &dc = conf_.dynamic_constraints();
  const int num_poses = coarse_path_.size();
  ref_velocities_.resize(num_poses > 0 ? num_poses - 1 : 0, 0.0);

  if (num_poses < 2) {
    return;
  }

  const double max_v = dc.max_vx();
  const double max_a = dc.max_accel_vx();
  const double v0 = std::hypot(initial_velocity[0], initial_velocity[1]);
  const double v_cruise = std::clamp(dc.reference_velocity(), 0.0, max_v);

  std::vector<double> arc_length(num_poses, 0.0);
  for (int i = 1; i < num_poses; ++i) {
    const double dx = coarse_path_[i][0] - coarse_path_[i - 1][0];
    const double dy = coarse_path_[i][1] - coarse_path_[i - 1][1];
    arc_length[i] = arc_length[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  const double total_length = arc_length.back();
  const double v_end = stop_at_end ? 0.0 : v_cruise;
  if (total_length < 1e-6 || max_a < 1e-6) {
    std::fill(ref_velocities_.begin(), ref_velocities_.end(), v_cruise);
    return;
  }

  std::vector<double> v_curve(num_poses, max_v);
  if (dc.max_omega() > 1e-6 && num_poses >= 3) {
    auto norm_angle = [](double a) {
      return std::atan2(std::sin(a), std::cos(a));
    };
    const int w = 3;
    for (int i = 1; i < num_poses - 1; ++i) {
      const int a = std::max(0, i - w);
      const int b = std::min(num_poses - 1, i + w);
      const double baseline = arc_length[b] - arc_length[a];
      if (b - a < 2 || baseline < 0.2) {
        continue;
      }
      const double dir_a =
          std::atan2(coarse_path_[i][1] - coarse_path_[a][1],
                     coarse_path_[i][0] - coarse_path_[a][0]);
      const double dir_b =
          std::atan2(coarse_path_[b][1] - coarse_path_[i][1],
                     coarse_path_[b][0] - coarse_path_[i][0]);
      const double dphi = norm_angle(dir_b - dir_a);
      const double kappa = std::abs(dphi) / baseline;
      if (kappa > 1e-6) {
        v_curve[i] = std::min(max_v, dc.max_omega() / kappa);
      }
    }
    v_curve.front() = v_curve[1];
    v_curve.back() = v_curve[num_poses - 2];
  }

  std::vector<double> fwd_vel(num_poses, 0.0);
  fwd_vel[0] = std::min(v0, v_curve[0]);
  for (int i = 1; i < num_poses; ++i) {
    const double ds = arc_length[i] - arc_length[i - 1];
    const double v_reach =
        std::sqrt(fwd_vel[i - 1] * fwd_vel[i - 1] + 2.0 * max_a * ds);
    fwd_vel[i] = std::min(std::min(v_reach, v_cruise), v_curve[i]);
  }

  std::vector<double> bwd_vel(num_poses, 0.0);
  bwd_vel[num_poses - 1] = std::min(v_end, v_curve[num_poses - 1]);
  for (int i = num_poses - 2; i >= 0; --i) {
    const double ds = arc_length[i + 1] - arc_length[i];
    const double v_reach =
        std::sqrt(bwd_vel[i + 1] * bwd_vel[i + 1] + 2.0 * max_a * ds);
    bwd_vel[i] = std::min(std::min(v_reach, v_cruise), v_curve[i]);
  }

  for (int i = 0; i < num_poses - 1; ++i) {
    ref_velocities_[i] = std::min(fwd_vel[i], bwd_vel[i]);
  }
  AINFO << "ref_velocities: n=" << ref_velocities_.size()
        << ", head=" << ref_velocities_.front()
        << ", peak=" << *std::max_element(ref_velocities_.begin(),
                                           ref_velocities_.end())
        << ", tail=" << ref_velocities_.back();

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