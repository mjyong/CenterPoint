// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). pluginlib class loading is replaced by
// a compile-time registry.

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/algorithm/mppi_planner/critic_data.h"
#include "src/algorithm/mppi_planner/critic_function.h"

namespace mppi {

/**
 * @class mppi::CriticManager
 * @brief Manager of objective functions for scoring trajectories
 */
class CriticManager {
public:
  typedef std::vector<std::unique_ptr<critics::CriticFunction>> Critics;
  typedef std::function<std::unique_ptr<critics::CriticFunction>()> Creator;

  CriticManager() = default;
  ~CriticManager() = default;

  /**
   * @brief Registry of critic creators (function-local static to avoid static
   * initialization order issues). Critic implementations register themselves
   * via MPPI_REGISTER_CRITIC in their .cc files.
   */
  static std::map<std::string, Creator> &registry() {
    static std::map<std::string, Creator> instance;
    return instance;
  }

  static void registerCritic(const std::string &name, const Creator &creator) {
    registry()[name] = creator;
  }

  /**
   * @brief Configure critic manager on bringup and load critics
   * @param ctx Shared critic context
   */
  void on_configure(const CriticContext &ctx);

  /**
   * @brief Score trajectories by the set of loaded critic functions
   * @param data Critic data (costs accumulated in place)
   * @param per_critic Optional: when not null, receives the per-critic cost
   * contributions (name + weighted component) in execution order, for the
   * CUDA-vs-CPU debug comparison.
   */
  void evalTrajectoriesScores(
      CriticData &data,
      std::vector<std::pair<std::string, Eigen::ArrayXf>> *per_critic =
          nullptr) const;

protected:
  /**
   * @brief Load the critics listed in the configuration
   */
  void loadCritics();

  /**
   * @brief Default critic list used when the configuration lists none
   */
  static std::vector<std::string> defaultCritics();

protected:
  CriticContext ctx_;
  std::vector<std::string> critic_names_;
  Critics critics_;
};

} // namespace mppi

#define MPPI_REGISTER_CRITIC(name, Class)                                      \
  namespace {                                                                  \
  struct MppiCriticRegistrar {                                                 \
    MppiCriticRegistrar() {                                                    \
      mppi::CriticManager::registerCritic(                                     \
          name, []() -> std::unique_ptr<mppi::critics::CriticFunction> {       \
            return std::unique_ptr<mppi::critics::CriticFunction>(new Class());\
          });                                                                  \
    }                                                                          \
  };                                                                           \
  static MppiCriticRegistrar mppi_critic_registrar;                            \
  } // namespace
