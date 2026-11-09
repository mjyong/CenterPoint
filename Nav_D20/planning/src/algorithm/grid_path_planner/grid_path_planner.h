#pragma once
#include "grid_path_planner_conf.pb.h"
#include "src/algorithm/grid_path_planner/dynamicvoronoi/dynamicvoronoi.h"
#include "src/common/structures.h"
#include <grid_map_core/grid_map_core.hpp>

#include <opencv2/opencv.hpp>

#include "src/algorithm/grid_path_planner/discrete_points_smoothing/fem_pos_deviation_osqp_interface.h"
using planning::algorithm::FemPosDeviationOsqpInterface;

using planning::config::GridPathPlannerConf;

namespace planning {
namespace algorithm {

class GridPathPlanner {

public:
  GridPathPlanner(const GridPathPlannerConf &config);
  ~GridPathPlanner();
  bool initGridMap();
  void SetGridMap(const grid_map::GridMap &map);
  bool initGridMapFromMatrix(const Eigen::MatrixXd &obstacle_matrix,
                             double resolution, const Eigen::Vector2d &origin,
                             double occupied_thresh, double free_thresh,
                             bool negate);
  bool plan(Eigen::Vector2d start, Eigen::Vector2d end,
            std::vector<Eigen::Vector2d> &path);
  const grid_map::GridMap &grid_map() const { return grid_map_; }
  std::function<double(double, double)> GetObstacleDistanceFunction() {
    return [this](double x, double y) {
      return this->getObstacleDistance(Eigen::Vector2d(x, y));
    };
  }
  std::function<bool(double, double, double)> GetIsCollisionFunction() {
    return [this](double x, double y, double theta) {
      return this->isCollisionExact(x, y, theta);
    };
  }

private:
  bool loadFromPGM(const std::string &file_name);
  void calcCostMap();
  bool isCollision(const Eigen::Vector2d &point);
  bool isCollision(const Eigen::Vector3d &pose);
  bool loadCostMap(const std::string &map_path,
                   const std::string &dist_map_path,
                   const std::string &yaml_path);
  bool saveCostMap(const std::string &map_path,
                   const std::string &dist_map_path,
                   const std::string &yaml_path);
  bool isCollisionExact(const double &x, const double &y, const double &theta);
  double getObstacleDistance(const Eigen::Vector2d &pt);

  void InitVoronoi();
  void SaveVoronoi(const std::string &file_path);
  bool LoadVoronoi(const std::string &file_path);

  double calculate_dynamic_bound(const Eigen::Vector2d &point,
                                 const grid_map::GridMap &map,
                                 double max_bound);
  Eigen::Vector2d findNearestFreePoint(const Eigen::Vector2d &end);

private:
  GridPathPlannerConf config_;
  grid_map::GridMap grid_map_;
  DynamicVoronoi voronoi_;
};

} // namespace algorithm
} // namespace planning
