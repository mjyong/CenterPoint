#pragma once
#include <cstddef>

namespace planning {
namespace preprocess {

struct CostMapQueryContext {
  const float *distance_field;
  const float *composite_cost_field;
  const float *grad_x_field;
  const float *grad_y_field;
  const float *ref_offset_dist_field;
  const float *ref_offset_cost_field;
  int map_rows;
  int map_cols;
  float resolution;
  float origin_x;
  float origin_y;
};

struct FootprintCircles {
  float offsets_x[8];
  float offsets_y[8];
  float radii[8];
  int num_circles;
  float bounding_radius;
};

struct CudaResources {
  size_t block_size = 0;
  int npp_size_width = 0;
  int npp_size_height = 0;
  unsigned char *cuda_input = nullptr;
  float *cuda_output = nullptr;
  float *cuda_grad_x = nullptr;
  float *cuda_grad_y = nullptr;
  float *cuda_composite_cost = nullptr;
  unsigned char *cuda_composite_obstacle = nullptr;
  float *cuda_ref_offset_dist = nullptr;
  float *cuda_ref_offset_cost = nullptr;
  unsigned char *device_buffer = nullptr;
  void *npp_stream_ctx = nullptr;
  FootprintCircles *d_footprint = nullptr;
  FootprintCircles h_footprint;
  CostMapQueryContext query_context;
  bool initialized = false;
};

class CostMapProductionCuda {
public:
  CostMapProductionCuda() = default;
  ~CostMapProductionCuda() = default;

  bool Init(int map_rows, int map_cols, double resolution,
            double map_position_x, double map_position_y, double map_length_x,
            double map_length_y);
  void Release();

  bool ProcObstacle(const unsigned char *host_obstacle, int rows, int cols,
                    double safe_margin, double inflation_margin,
                    double max_obstacle_dist, double resolution);

  bool ProcReference(const unsigned char *host_ref_path, int rows, int cols,
                     double max_reference_offset, double resolution);

  bool ProcComposite(unsigned char *host_composite_obstacle, int rows,
                     int cols);

  bool ProcCompositeObstacle(const unsigned char *host_obstacle,
                             const float *host_obstacle_distance,
                             const float *host_ref_offset,
                             unsigned char *host_composite_obstacle, int rows,
                             int cols, double inflation_safe_margin,
                             double max_reference_offset, bool has_ref_path);

  void DownloadDistanceField(float *host_distance, int size);
  void DownloadCostField(float *host_cost, int size);
  void DownloadGradientXField(float *host_grad_x, int size);
  void DownloadGradientYField(float *host_grad_y, int size);
  void DownloadRefOffsetDistField(float *host_ref_offset_dist, int size);
  void DownloadRefOffsetCostField(float *host_ref_offset_cost, int size);
  void DownloadCompositeCostField(float *host_composite_cost, int size);

  const CostMapQueryContext &GetQueryContext() const {
    return resources_.query_context;
  }
  const FootprintCircles *GetDeviceFootprint() const {
    return resources_.d_footprint;
  }
  CudaResources &resources() { return resources_; }
  bool initialized() const { return resources_.initialized; }

private:
  CudaResources resources_;
};

} // namespace preprocess
} // namespace planning
