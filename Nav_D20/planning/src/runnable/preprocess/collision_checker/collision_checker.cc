#include "src/runnable/preprocess/collision_checker/collision_checker.h"
#include "src/common/util/config_helper.h"
using planning::common::util::ConfigHelper;
namespace planning {
namespace preprocess {
CollisionChecker::CollisionChecker(const PreprocessConf::Conf &conf) {
  conf_ = conf;
  config_ = conf.collision_checker_conf();
}

CollisionChecker::~CollisionChecker() {}

bool CollisionChecker::Init(PlanningEntry *planning_entry) {
  std::string urdf_dir =
      ConfigHelper::Instance()->GetConfDir() + config_.urdf_dir();
  robot_model_ = std::make_unique<FCLRobotModel>(urdf_dir, config_.urdf_name());
  //   auto links = robot_model_->getLinkNames();
  //   for (const auto link : links) {
  //     AINFO << "link name: " << link;
  //   }
  return true;
}

bool CollisionChecker::Proc() {
  contacts_.clear();
  // robot_model_->checkSelfCollision(contacts_);
  for (size_t i = 0; i < 10000; i++)
  {
      robot_model_->checkMapCollision(contacts_);

  }
  
  // robot_model_->checkMapCollision(contacts_, Eigen::Vector3d(-1, 0, 0));
  // robot_model_->getObstacleDistance(Eigen::Vector3d(0, 0, 0));
  return true;
}

} // namespace preprocess
} // namespace planning