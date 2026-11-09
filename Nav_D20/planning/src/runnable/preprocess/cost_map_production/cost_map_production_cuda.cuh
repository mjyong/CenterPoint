#pragma once
#include "src/runnable/preprocess/cost_map_production/cost_map_production_cuda.h"

namespace planning {
namespace preprocess {

__device__ inline float bilinearInterpNpp(const float *field, int npp_width,
                                          int npp_height, float row_f,
                                          float col_f) {
  int y0 = max(0, min(__float2int_rd(col_f), npp_height - 1));
  int x0 = max(0, min(__float2int_rd(row_f), npp_width - 1));
  int y1 = min(y0 + 1, npp_height - 1);
  int x1 = min(x0 + 1, npp_width - 1);
  float dy = max(0.0f, min(1.0f, col_f - __int2float_rn(y0)));
  float dx = max(0.0f, min(1.0f, row_f - __int2float_rn(x0)));
  float v00 = field[y0 * npp_width + x0];
  float v10 = field[y1 * npp_width + x0];
  float v01 = field[y0 * npp_width + x1];
  float v11 = field[y1 * npp_width + x1];
  return (1 - dy) * (1 - dx) * v00 + dy * (1 - dx) * v10 + (1 - dy) * dx * v01 +
         dy * dx * v11;
}

__device__ inline void posToGridIndex(float x, float y, float origin_x,
                                      float origin_y, float resolution,
                                      float &row_f, float &col_f) {
  row_f = (origin_x - x) / resolution;
  col_f = (origin_y - y) / resolution;
}

__device__ inline float queryObstacleDistance(const CostMapQueryContext &ctx,
                                              float x, float y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    return 1e6f;
  }
  return bilinearInterpNpp(ctx.distance_field, ctx.map_rows, ctx.map_cols,
                           row_f, col_f);
}

__device__ inline float
queryObstacleDistanceCost(const CostMapQueryContext &ctx, float x, float y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    return 1.0f;
  }
  return bilinearInterpNpp(ctx.composite_cost_field, ctx.map_rows, ctx.map_cols,
                           row_f, col_f);
}

__device__ inline void queryObstacleGradient(const CostMapQueryContext &ctx,
                                             float x, float y, float &grad_x,
                                             float &grad_y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    grad_x = 0.0f;
    grad_y = 0.0f;
    return;
  }
  grad_x = bilinearInterpNpp(ctx.grad_x_field, ctx.map_rows, ctx.map_cols,
                             row_f, col_f);
  grad_y = bilinearInterpNpp(ctx.grad_y_field, ctx.map_rows, ctx.map_cols,
                             row_f, col_f);
}

__device__ inline bool queryIsCollisionFast(const CostMapQueryContext &ctx,
                                            const FootprintCircles &footprint,
                                            float x, float y, float theta) {
  float dist = queryObstacleDistance(ctx, x, y);
  return dist < footprint.bounding_radius;
}

__device__ inline bool queryIsCollisionExact(const CostMapQueryContext &ctx,
                                             const FootprintCircles &footprint,
                                             float x, float y, float theta) {
  float cos_t = cosf(theta);
  float sin_t = sinf(theta);
  for (int i = 0; i < footprint.num_circles; ++i) {
    float cx =
        x + footprint.offsets_x[i] * cos_t - footprint.offsets_y[i] * sin_t;
    float cy =
        y + footprint.offsets_x[i] * sin_t + footprint.offsets_y[i] * cos_t;
    float dist = queryObstacleDistance(ctx, cx, cy);
    if (dist < footprint.radii[i]) {
      return true;
    }
  }
  return false;
}

__device__ inline bool queryIsCollision(const CostMapQueryContext &ctx,
                                        const FootprintCircles &footprint,
                                        float x, float y, float theta) {
  if (queryIsCollisionFast(ctx, footprint, x, y, theta)) {
    return queryIsCollisionExact(ctx, footprint, x, y, theta);
  }
  return false;
}

__device__ inline bool queryIsObstacle(const CostMapQueryContext &ctx, float x,
                                       float y, float threshold = 0.01f) {
  return queryObstacleDistance(ctx, x, y) < threshold;
}

__device__ inline float queryReferenceOffsetDist(const CostMapQueryContext &ctx,
                                                 float x, float y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    return 1e6f;
  }
  return bilinearInterpNpp(ctx.ref_offset_dist_field, ctx.map_rows,
                           ctx.map_cols, row_f, col_f);
}

__device__ inline float queryReferenceOffsetCost(const CostMapQueryContext &ctx,
                                                 float x, float y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    return 1.0f;
  }
  return bilinearInterpNpp(ctx.ref_offset_cost_field, ctx.map_rows,
                           ctx.map_cols, row_f, col_f);
}

__device__ inline float queryCompositeCost(const CostMapQueryContext &ctx,
                                           float x, float y) {
  float row_f, col_f;
  posToGridIndex(x, y, ctx.origin_x, ctx.origin_y, ctx.resolution, row_f,
                 col_f);
  if (row_f < 0.0f || row_f >= ctx.map_rows || col_f < 0.0f ||
      col_f >= ctx.map_cols) {
    return 1.0f;
  }
  return bilinearInterpNpp(ctx.composite_cost_field, ctx.map_rows, ctx.map_cols,
                           row_f, col_f);
}

} // namespace preprocess
} // namespace planning
