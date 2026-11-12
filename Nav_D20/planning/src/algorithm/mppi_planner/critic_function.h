// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). The ros/costmap_2d plumbing of the base
// class is replaced by the CriticContext.

#pragma once

#include <string>

#include "src/algorithm/mppi_planner/critic_data.h"

namespace mppi {
namespace critics {

/**
 * @class mppi::critics::CollisionCost
 * @brief Utility for storing cost information
 */
struct CollisionCost {
  float cost{0};
  bool using_footprint{false};
};

/**
 * @class mppi::critics::CriticFunction
 * @brief Abstract critic objective function to score trajectories
 */
class CriticFunction {
public:
  CriticFunction() = default;
  virtual ~CriticFunction() = default;

  /**
   * @brief Configure critic on bringup
   * @param ctx Shared critic context (configuration + costmap callbacks)
   * @param name Name of the critic
   */
  void on_configure(const CriticContext &ctx, const std::string &name) {
    ctx_ = ctx;
    name_ = name;
    initialize();
  }

  /**
   * @brief Main function to score trajectory
   */
  virtual void score(CriticData &data) = 0;

  /**
   * @brief Get name of critic
   */
  std::string getName() const { return name_; }

protected:
  /**
   * @brief Initialize critic from the configuration in the context
   */
  virtual void initialize() = 0;

  bool enabled_{false};
  std::string name_;
  CriticContext ctx_;
};

} // namespace critics
} // namespace mppi
