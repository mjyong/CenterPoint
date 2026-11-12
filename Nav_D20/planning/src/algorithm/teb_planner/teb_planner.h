#pragma once
#include "src/algorithm/teb_planner/ceres_cost_func.h"
#include "src/algorithm/teb_planner/coarse_path_search.h"
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/common/util/lowpass_filter.h"
#include "src/common/util/math.h"
#include "teb_planner_conf.pb.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

namespace teb_planner {

using planning::common::util::normalize_angle;

class TebPlanner {

public:
  TebPlanner(
      const planning::config::TebPlannerConf &conf,
      std::function<double(const double, const double)> is_obstacle_func,
      std::function<double(const double, const double)> obstacle_dist_func,
      std::function<double(const double, const double)> composite_cost_func,
      std::function<bool(const double, const double, const double)>
          is_collision_func);
  ~TebPlanner();

  bool solver(std::vector<Eigen::Vector3d> waypoints,
              const Eigen::Vector3d &initial_pose,
              const Eigen::Vector3d &initial_velocity,
              Eigen::Vector3d final_pose, bool constant_final_theta,
              const planning::Transform3d &global_pose,
              const bool &stop_at_end = false);
  void set_target_speed(double target_speed) {
    conf_.mutable_dynamic_constraints()->set_reference_velocity(target_speed);
  }
  const std::vector<Eigen::Vector3d> &getCoarsePath() const {
    return coarse_path_;
  }
  const std::vector<Eigen::Vector3d> &getOptimizePath() const {
    return optimize_path_;
  }
  void reset() {
    coarse_path_.clear();
    optimize_path_.clear();
    prev_optimize_path_.clear();
    has_prev_global_pose_ = false;
    x_.clear();
    y_.clear();
    theta_.clear();
    time_deltas_.clear();
    ref_velocities_.clear();
    velocity_output_.clear();
  }
  Eigen::Vector3d getVelocityCmd(double t);

private:
  void buildProblem(const Eigen::Vector3d &initial_velocity, bool stop_at_end);
  void computeRobotVelocity(const double &x1, const double &y1,
                            const double &theta1, const double &x2,
                            const double &y2, const double &theta2,
                            const double &dt, double &vx_robot,
                            double &vy_robot, double &omega) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dtheta = normalize_angle(theta2 - theta1);
    double vx_global = dx / dt;
    double vy_global = dy / dt;
    omega = dtheta / dt;
    double cos_theta = cos(theta1);
    double sin_theta = sin(theta1);
    vx_robot = vx_global * cos_theta + vy_global * sin_theta;
    vy_robot = -vx_global * sin_theta + vy_global * cos_theta;
  }

  void computeReferenceVelocities(const Eigen::Vector3d &initial_velocity,
                                  bool stop_at_end);

private:
  planning::config::TebPlannerConf conf_;
  std::function<double(const double, const double)> is_obstacle_func_;
  std::function<double(const double, const double)> obstacle_dist_func_;
  std::function<double(const double, const double)> composite_cost_func_;
  std::function<bool(const double, const double, const double)>
      is_collision_func_;
  std::unique_ptr<ceres::Problem> problem_;
  std::vector<Eigen::Vector3d> coarse_path_;
  std::vector<Eigen::Vector3d> optimize_path_;
  std::vector<Eigen::Vector3d> prev_optimize_path_;
  planning::Transform3d prev_global_pose_ = planning::Transform3d::Identity();
  bool has_prev_global_pose_ = false;
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> theta_;
  std::vector<double> time_deltas_;
  std::vector<double> ref_velocities_;
  std::vector<std::pair<double, Eigen::Vector3d>> velocity_output_;
  CoarsePathSearch coarse_path_search_;
  planning::common::util::Butterworth2dFilter filter_vx_;
  planning::common::util::Butterworth2dFilter filter_vy_;
  planning::common::util::Butterworth2dFilter filter_omega_;
  planning::common::util::Butterworth2dFilter filter_input_vx_;
  planning::common::util::Butterworth2dFilter filter_input_vy_;
  planning::common::util::Butterworth2dFilter filter_input_omega_;
};

} // namespace teb_planner