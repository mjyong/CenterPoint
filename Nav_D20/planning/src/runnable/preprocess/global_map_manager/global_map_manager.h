#pragma once
#include "grid_map_core/grid_map_core.hpp"
#include "src/runnable/preprocess/preprocess_base.h"
#include "third_party/osm_map/osm_parser.h"
#include <opencv2/core.hpp>
#include <unordered_map>
#include <unordered_set>

namespace planning {
namespace preprocess {

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
  std::unordered_map<std::string, TrackedCell> updated_cells_;
  double cell_timeout_seconds_;

  lanelet::osm::OsmParser osm_parser_;
};

} // namespace preprocess
} // namespace planning
