// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"
#include "src/algorithm/mppi_planner/critics/obstacles_critic.h"

#include <cmath>
#include <limits>

#include "src/algorithm/mppi_planner/tools/utils.h"
#include "src/common/util/config_helper.h"
#include "src/common/util/log.h"

namespace mppi {
namespace critics {

void ObstaclesCritic::initialize() {
  const auto &c = ctx_.conf->obstacles_critic();
  enabled_ = c.enabled();
  power_ = static_cast<unsigned int>(c.power());
  repulsion_weight_ = static_cast<float>(c.repulsion_weight());
  critical_weight_ = static_cast<float>(c.critical_weight());
  collision_cost_ = static_cast<float>(c.collision_cost());
  collision_margin_distance_ = static_cast<float>(c.collision_margin_distance());
  near_goal_distance_ = static_cast<float>(c.near_goal_distance());
  repulsion_radius_ = static_cast<float>(c.repulsion_radius());

  // Multi-circle footprint model: collision / clearance queries evaluate
  // every bounding circle against the obstacle distance field.
  circles_.clear();
  try {
    const auto circles = planning::common::util::ConfigHelper::GetBoundingCircles(
        0.0, 0.0, 0.0);
    circles_.reserve(circles.size());
    for (const auto &circle : circles) {
      circles_.push_back({static_cast<float>(circle.center_x),
                          static_cast<float>(circle.center_y),
                          static_cast<float>(circle.radius)});
    }
  } catch (const std::exception &e) {
    AERROR << "ObstaclesCritic failed to get bounding circles: " << e.what();
  }
  AINFO << "ObstaclesCritic footprint circles: " << circles_.size();
}

float ObstaclesCritic::footprintClearance(float x, float y,
                                          float theta) const {
  const float cos_t = std::cos(theta);
  const float sin_t = std::sin(theta);
  float min_clearance = std::numeric_limits<float>::max();
  for (const auto &circle : circles_) {
    // Circle center rotated into the pose
    const float cx = x + cos_t * circle.x - sin_t * circle.y;
    const float cy = y + sin_t * circle.x + cos_t * circle.y;
    const float clearance = static_cast<float>(ctx_.obstacle_dist_func(
                                static_cast<double>(cx),
                                static_cast<double>(cy))) -
                            circle.r;
    if (clearance < min_clearance) {
      min_clearance = clearance;
    }
  }
  return min_clearance;
}

void ObstaclesCritic::score(CriticData &data) {
  if (!enabled_) {
    return;
  }

  // If near the goal, don't apply the preferential term since the goal is
  // near obstacles
  bool near_goal = false;
  if (utils::withinPositionGoalTolerance(
          near_goal_distance_, data.state.pose, data.goal)) {
    near_goal = true;
  }

  Eigen::ArrayXf raw_cost = Eigen::ArrayXf::Zero(data.costs.size());
  Eigen::ArrayXf repulsive_cost = Eigen::ArrayXf::Zero(data.costs.size());

  const unsigned int traj_len = data.trajectories.x.cols();
  const unsigned int batch_size = data.trajectories.x.rows();
  bool all_trajectories_collide = true;

  for (unsigned int i = 0; i != batch_size; i++) {
    bool trajectory_collide = false;
    float traj_cost = 0.0f;
    const auto &traj = data.trajectories;

    for (unsigned int j = 0; j != traj_len; j++) {
      const float x = traj.x(i, j);
      const float y = traj.y(i, j);
      const float theta = traj.yaws(i, j);
      // Multi-circle footprint detection (robot frame): minimum clearance of
      // all bounding circles. Out-of-costmap queries return a huge clearance,
      // i.e. free space.
      const float clearance = footprintClearance(x, y, theta);

      if (inCollision(clearance)) {
        trajectory_collide = true;
        break;
      }

      if (clearance > repulsion_radius_ &&
          clearance > collision_margin_distance_) {
        continue; // Far from any obstacle
      }

      // Let near-collision trajectory points be punished severely
      if (clearance < collision_margin_distance_) {
        traj_cost += (collision_margin_distance_ - clearance);
      }

      // Generally prefer trajectories further from obstacles
      if (!near_goal) {
        repulsive_cost[i] += repulsion_radius_ - clearance;
      }
    }

    if (!trajectory_collide) {
      all_trajectories_collide = false;
    }
    raw_cost(i) = trajectory_collide ? collision_cost_ : traj_cost;
  }

  // Normalize repulsive cost by trajectory length & lowest score to not
  // overweight importance. This is a preferential cost, not collision cost,
  // to be tuned relative to desired behaviors.
  auto repulsive_cost_normalized =
      (repulsive_cost - repulsive_cost.minCoeff()) / traj_len;

  if (power_ > 1u) {
    data.costs += ((critical_weight_ * raw_cost) +
                   (repulsion_weight_ * repulsive_cost_normalized))
                      .pow(power_);
  } else {
    data.costs += (critical_weight_ * raw_cost) +
                  (repulsion_weight_ * repulsive_cost_normalized);
  }

  data.fail_flag = all_trajectories_collide;
}

} // namespace critics
} // namespace mppi

MPPI_REGISTER_CRITIC("ObstaclesCritic", mppi::critics::ObstaclesCritic)
