// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). The costmap_2d inflation-layer reverse
// distance computation is replaced by the obstacle distance field callback;
// collision checking uses the multi-circle footprint model from
// ConfigHelper::GetBoundingCircles.

#pragma once

#include <vector>

#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::ObstaclesCritic
 * @brief Critic objective function for avoiding obstacles, allowing it to
 * deviate off the planned path. Uses the obstacle distance field directly
 * instead of reverse-computing distances from inflation layer costs.
 * Collision detection evaluates every bounding circle of the footprint
 * against the distance field.
 */
class ObstaclesCritic : public CriticFunction {
public:
  void initialize() override;
  void score(CriticData &data) override;

protected:
  /**
   * @brief Checks if the pose clearance indicates a collision
   * @param clearance Minimum distance-field clearance over all bounding
   *        circles (distance at circle center minus circle radius)
   * @return bool if in collision
   */
  inline bool inCollision(float clearance) const {
    return clearance < 0.0f;
  }

  /**
   * @brief Minimum footprint clearance at a pose over all bounding circles
   * @param x X of pose (robot frame)
   * @param y Y of pose (robot frame)
   * @param theta Yaw of pose
   * @return Minimum (distance-field value at circle center - circle radius)
   */
  float footprintClearance(float x, float y, float theta) const;

protected:
  // Multi-circle footprint model (robot frame, at identity pose)
  struct BoundingCircle {
    float x, y, r;
  };
  std::vector<BoundingCircle> circles_;

  float collision_cost_{0};
  float repulsion_radius_{0};

  float collision_margin_distance_{0};
  float near_goal_distance_{0};

  unsigned int power_{0};
  float repulsion_weight_{0}, critical_weight_{0};
};

} // namespace critics
} // namespace mppi
