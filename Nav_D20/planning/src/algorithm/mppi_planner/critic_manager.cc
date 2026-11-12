// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov).

#include "src/algorithm/mppi_planner/critic_manager.h"

#include "src/common/util/log.h"

namespace mppi {

void CriticManager::on_configure(const CriticContext &ctx) {
  ctx_ = ctx;

  const auto &critics = ctx.conf->critics();
  if (critics.empty()) {
    critic_names_ = defaultCritics();
  } else {
    critic_names_ = std::vector<std::string>(critics.begin(), critics.end());
  }
  loadCritics();
}

void CriticManager::loadCritics() {
  critics_.clear();
  for (const auto &name : critic_names_) {
    auto it = registry().find(name);
    if (it == registry().end()) {
      AERROR << "MPPI critic not registered: " << name << ", skipping";
      continue;
    }
    critics_.push_back(it->second());
    critics_.back()->on_configure(ctx_, name);
    AINFO << "MPPI critic loaded: " << name;
  }
}

std::vector<std::string> CriticManager::defaultCritics() {
  return {"PathAlignCritic",   "PathFollowCritic",   "PreferForwardCritic",
          "GoalCritic",        "GoalAngleCritic",    "PathAngleCritic",
          "TwirlingCritic",    "ConstraintCritic",
          "VelocityDeadbandCritic", "ObstaclesCritic",
          "ReferenceVelocityCritic"};
}

void CriticManager::evalTrajectoriesScores(
    CriticData &data,
    std::vector<std::pair<std::string, Eigen::ArrayXf>> *per_critic) const {
  for (const auto &critic : critics_) {
    if (data.fail_flag) {
      break;
    }
    const Eigen::ArrayXf before = data.costs;
    critic->score(data);
    if (per_critic != nullptr) {
      per_critic->emplace_back(critic->getName(), data.costs - before);
    }
  }
}

} // namespace mppi
