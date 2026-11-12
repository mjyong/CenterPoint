#pragma once
#include "grid_map_core/grid_map_core.hpp"
#include "src/algorithm/grid_path_planner/grid_path_planner.h"
#include "src/common/path2d.h"
#include "src/runnable/preprocess/preprocess_base.h"
#include "third_party/osm_map/osm_parser.h"
#include <opencv2/core.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace planning {
namespace preprocess {

using planning::algorithm::GridPathPlanner;

struct TrackedCell {
  double last_update_time;
  grid_map::Index index;
};

class GlobalMapManager : public PreprocessBase {
public:
  explicit GlobalMapManager(const PreprocessConf::Conf &conf);
  ~GlobalMapManager();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;

  const grid_map::GridMap &global_map() const { return global_map_; }
  grid_map::GridMap *mutable_global_map() { return &global_map_; }

  bool UpdateMap(const grid_map::GridMap &map);
  bool ReloadMap(const std::string &map_dir);

  bool PlanGlobalPath(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
                      std::optional<double> end_theta = std::nullopt);
  void ClearGlobalPath();
  const Path2d &global_path() const { return global_path_; }
  bool global_plan_flag() const { return global_plan_flag_; }

  std::vector<Pose2d> ExtractLocalPath(const Eigen::Vector2d &robot_pos,
                                       double ahead_s) const;

  // crosswalk polygons of the loaded map's OSM (map frame); they do not change the obstacle layers.
  // The yield gate checks vehicles where the global path crosses one (docs/PLANNING_INTEGRATION.md)
  const std::vector<std::vector<Eigen::Vector2d>> &crosswalks() const { return crosswalks_; }

private:
  bool LoadPGMMap(const std::string &pgm_file_path,
                  const std::string &yaml_file_path);
  void AddObstacleFromPerception();
  void ResetStaleCells();

  bool AddObstacleFromOSM(const std::string &osm_file_path);

private:
  PreprocessConf::GlobalMapManagerConf config_;
  grid_map::GridMap global_map_;
  bool map_loaded_{false};
  bool map_dirty_{true};
  std::unordered_map<std::string, TrackedCell> updated_cells_;
  double cell_timeout_seconds_;

  std::shared_ptr<GridPathPlanner> grid_path_planner_;
  Path2d global_path_;
  bool global_plan_flag_{false};

  lanelet::osm::OsmParser osm_parser_;
  std::vector<std::vector<Eigen::Vector2d>> crosswalks_;
};

} // namespace preprocess
} // namespace planning
