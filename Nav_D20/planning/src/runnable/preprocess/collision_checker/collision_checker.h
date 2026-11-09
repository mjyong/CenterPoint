#pragma once
#include "src/common/structures.h"
#include "src/runnable/preprocess/collision_checker/fcl_robot_model.h"
#include "src/runnable/preprocess/preprocess_base.h"
#include <functional>
#include <urdf/model.h>
namespace planning {
namespace preprocess {

class CollisionChecker : public PreprocessBase {
public:
  explicit CollisionChecker(const PreprocessConf::Conf &conf);
  ~CollisionChecker();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  void updateTransforms(const std::unordered_map<std::string, Eigen::Isometry3d>
                            &link_transforms) {
    robot_model_->updateTransforms(link_transforms);
  }
  void updateOctomap(const std::shared_ptr<const octomap::OcTree> &octree) {
    robot_model_->updateOctomap(octree);
  }
  const std::vector<std::string> getLinkNames() const {
    return robot_model_->getLinkNames();
  }
  const std::vector<Eigen::Vector3d> contacts() const { return contacts_; }

private:
  PreprocessConf::CollisionCheckerConf config_;
  std::unique_ptr<FCLRobotModel> robot_model_;
  std::vector<Eigen::Vector3d> contacts_;
};
} // namespace preprocess
} // namespace planning