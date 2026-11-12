#pragma once
#include "grid_map_core/grid_map_core.hpp"
#include "src/common/structures.h"
#include "src/runnable/preprocess/cost_map_production/cost_map_production_cuda.h"
#include "src/runnable/preprocess/preprocess_base.h"
#include <functional>
#include <limits>
#include <vector>

#include "src/runnable/preprocess/cost_map_production/eigen2cv.hpp"
#include <opencv2/core/eigen.hpp>

namespace planning {
namespace preprocess {

struct ScanObstaclePoint {
  double x = 0.0;
  double y = 0.0;
  double distance = std::numeric_limits<double>::quiet_NaN();
};

class CostMapProduction : public PreprocessBase {
  enum class Layer {
    NONE = 0,
    OBSTACLE,
    OBSTACLE_DISTANCE,
    REFERENCE_PATH,
    REFERENCE_OFFSET_DIST,
    COMPOSITE_OBSTACLE,
    COMPOSITE_OBSTACLE_DIST,
    COMPOSITE_COST,
    COMPOSITE_GRADIENT_X,
    COMPOSITE_GRADIENT_Y,
  };

public:
  explicit CostMapProduction(const PreprocessConf::Conf &conf);
  ~CostMapProduction();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;

public:
  std::function<double(double, double)> GetObstacleDistanceFunction() {
    return
        [this](double x, double y) { return this->GetObstacleDistance(x, y); };
  }
  std::function<double(double, double)> GetCompositeObstacleDistanceFunction() {
    return [this](double x, double y) {
      return this->GetCompositeObstacleDistance(x, y);
    };
  }
  std::function<bool(double, double, double)> GetIsCollisionFunction() {
    return [this](double x, double y, double theta) {
      return this->IsCollision(x, y, theta);
    };
  }
  std::function<bool(double, double)> GetIsObstacleFunction() {
    return [this](double x, double y) {
      return (this->GetObstacleDistance(x, y) < 0.01);
    };
  }
  std::function<bool(double, double)> GetIsCompositeObstacleFunction() {
    return [this](double x, double y) {
      return (this->GetCompositeObstacleDistance(x, y) < 0.01);
    };
  }
  std::function<double(double, double)> GetCompositeCostFunction() {
    return [this](double x, double y) { return this->GetCompositeCost(x, y); };
  }
  bool IsCollision(const double &relative_x, const double &relative_y,
                   const double &relative_theta);

  double GetObstacleDistance(const double relative_x, const double relative_y);
  double GetCompositeObstacleDistance(const double relative_x,
                                      const double relative_y);

  void SetLocalReferencePath(const std::vector<Eigen::Vector3d> &local_ref_path);
  double GetReferenceOffsetDist(const double relative_x,
                                const double relative_y);
  double GetCompositeCost(const double relative_x, const double relative_y);
  void GetCompositeGradient(const double relative_x, const double relative_y,
                            double &grad_x, double &grad_y);

  // 获取原点 360° 内每个方向上最近的障碍物点(OBSTACLE 层)
  const std::vector<ScanObstaclePoint> &GetScanObstacles() const {
    return scan_obstacles_;
  }

  grid_map::GridMap *GridMap() { return &grid_map_; }

  const CostMapQueryContext &GetCudaQueryContext() const {
    return cuda_pipeline_.GetQueryContext();
  }
  const FootprintCircles *GetCudaFootprint() const {
    return cuda_pipeline_.GetDeviceFootprint();
  }
  bool UseCuda() const { return use_cuda_; }

private:
  double GetObstacleDistance(const grid_map::Position &pos);
  bool IsCollisionExact(const double &relative_x, const double &relative_y,
                        const double &relative_theta);
  void UpdateObstacleDistance();
  void UpdateScanObstacles();
  void UpdateReferenceOffsetDist();
  void BuildCompositeObstacle();
  void UpdateCompositeCost();
  void AddObstacleFromMap();
  void AddObstacleFromPerception();

private:
  PreprocessConf::CostMapProductionConf config_;
  std::map<Layer, std::string> layers_;
  grid_map::GridMap grid_map_;
  CostMapProductionCuda cuda_pipeline_;
  bool use_cuda_ = false;
  std::vector<Eigen::Vector3d> local_ref_path_;
  std::vector<ScanObstaclePoint> scan_obstacles_;
};

} // namespace preprocess
} // namespace planning
