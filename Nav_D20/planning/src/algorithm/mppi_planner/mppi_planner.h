#pragma once
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

#include "mppi_planner_conf.pb.h"
#include "src/algorithm/mppi_planner/critic_data.h"
#include "src/algorithm/mppi_planner/optimizer.h"
#include "src/algorithm/mppi_cuda_planner/mppi_cuda_backend.h"
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include "src/common/util/math.h"
#include "src/common/util/mean_filter.h"

namespace planning {
namespace algorithm {

// MPPI local planner facade following the TebPlanner interface convention.
// All geometry is expressed in the robot (base_link) frame, matching the
// costmap callbacks handed in at construction.
class MppiPlanner {

public:
  MppiPlanner(
      const planning::config::MppiPlannerConf &conf,
      std::function<bool(const double, const double)> is_obstacle_func,
      std::function<double(const double, const double)> obstacle_dist_func,
      std::function<mppi::cuda::MppiCudaMapField()> cuda_map_field_func =
          nullptr);
  ~MppiPlanner();

  bool solver(std::vector<Eigen::Vector3d> waypoints,
              const Eigen::Vector3d &initial_pose,
              const Eigen::Vector3d &initial_velocity);

  void set_target_speed(double target_speed) { target_speed_ = target_speed; }

  const std::vector<Eigen::Vector3d> &getOptimizePath() const {
    return optimize_path_;
  }

  void reset();

  Eigen::Vector3d getVelocityCmd(double t);

private:
  // Verify that the actual solver period matches model_dt; a mismatch
  // invalidates control sequence shifting (it would desynchronize the
  // sequence from the execution timeline).
  void checkControlPeriod();

private:
  planning::config::MppiPlannerConf conf_;
  double target_speed_{0.0};
  std::unique_ptr<mppi::Optimizer> optimizer_;
  // CUDA hot-path backend; null unless enable_cuda is set and init succeeded
  // (destroyed before optimizer_ via reverse member destruction order).
  std::unique_ptr<mppi::cuda::MppiCudaBackend> cuda_backend_;

  std::vector<Eigen::Vector3d> optimize_path_;
  std::vector<std::pair<double, Eigen::Vector3d>> velocity_output_;

  std::chrono::steady_clock::time_point last_solver_time_;
  bool has_last_solver_time_{false};
  bool control_period_checked_{false};

  // Filtered odometry velocity seeding the rollout (see solver()).
  Eigen::Vector3d filtered_velocity_{Eigen::Vector3d::Zero()};
  planning::common::util::MeanFilter vx_filter_;
  planning::common::util::MeanFilter vy_filter_;
  planning::common::util::MeanFilter wz_filter_;
  bool filter_initialized_{false};
};

} // namespace algorithm
} // namespace planning
