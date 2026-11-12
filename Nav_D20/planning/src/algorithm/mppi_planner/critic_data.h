// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). The pluginlib/costmap_2D context is
// replaced by a lightweight struct carrying the proto configuration and the
// robot-frame costmap query callbacks.

#pragma once

#include <Eigen/Dense>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "mppi_planner_conf.pb.h"
#include "src/algorithm/mppi_planner/models/path.h"
#include "src/algorithm/mppi_planner/models/state.h"
#include "src/algorithm/mppi_planner/models/trajectories.h"
#include "src/algorithm/mppi_planner/motion_models.h"

namespace mppi {

/**
 * @struct mppi::CriticContext
 * @brief Shared context handed to every critic: planner configuration plus
 * robot-frame costmap query functions (same callbacks the TebPlanner uses).
 */
struct CriticContext {
  const planning::config::MppiPlannerConf *conf{nullptr};
  // (x, y) -> true if the cell is an obstacle (robot frame)
  std::function<bool(double, double)> is_obstacle_func;
  // (x, y) -> clearance to the nearest obstacle in meters (robot frame)
  std::function<double(double, double)> obstacle_dist_func;
  // Externally injected reference speed (e.g. from the mission goal),
  // overriding reference_velocity_critic.reference_velocity when > 0.
  // Updated by the optimizer every cycle; read via CriticData::ctx.
  double runtime_reference_velocity{0.0};
};

/**
 * @struct mppi::CriticData
 * @brief Data to pass to critics for scoring, including state, trajectories,
 * pruned path, global goal, costs, and important parameters to share
 */
struct CriticData {
  const models::State &state;
  const models::Trajectories &trajectories;
  const models::Path &path;
  const models::Pose &goal;

  Eigen::ArrayXf &costs;
  float &model_dt;

  bool fail_flag;
  std::shared_ptr<MotionModel> motion_model;
  std::optional<std::vector<bool>> path_pts_valid;
  std::optional<size_t> furthest_reached_path_point;
  const CriticContext *ctx;
};

} // namespace mppi
