// A coarse copy of the global map for display (T59 review): the site map ad68517c is 10293 x 11412 cells at 0.1 m,
// an OccupancyGrid of 117 MB, and the dog's iceoryx (CycloneDDS shared memory) takes chunks of 16 MB at most; a
// bigger publish throws. Built once when a map loads; /planning/debug/global_map_manager/occmap publishes this.
// Eigen only (tested on its own: tests/dog/test_map_overview.py).
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace planning {
namespace preprocess {

struct MapOverview {
  int width = 0;                // cells along x
  int height = 0;               // cells along y
  double resolution = 0.0;      // [m]
  double origin_x = 0.0;        // [m] the min corner, map frame
  double origin_y = 0.0;
  std::vector<int8_t> data;     // OccupancyGrid order (from the min corner, x fastest): 0 free, 100 blocked, -1 unknown
};

// layer: a grid_map layer (index (0, 0) is the cell at the max x, max y corner; rows go along -x, columns along -y),
// `resolution` its cell size, (max_x, max_y) its max corner. Cells are pooled k x k, k the smallest that keeps the
// overview at max_cells or fewer: blocked if any cell of the block is blocked (below half of free_value; the
// planner's obstacle layer is 255 free, 0 blocked), unknown if all are NaN, else free.
inline MapOverview BuildMapOverview(const Eigen::MatrixXf &layer, double resolution, double max_x, double max_y,
                                    size_t max_cells, float free_value = 255.0f) {
  MapOverview o;
  const int rows = static_cast<int>(layer.rows()), cols = static_cast<int>(layer.cols());
  if (rows == 0 || cols == 0 || max_cells == 0) return o;
  int k = 1;
  while (static_cast<size_t>((rows + k - 1) / k) * static_cast<size_t>((cols + k - 1) / k) > max_cells) ++k;
  const int nx = (rows + k - 1) / k, ny = (cols + k - 1) / k;
  o.width = nx;
  o.height = ny;
  o.resolution = resolution * k;
  o.origin_x = max_x - nx * o.resolution;
  o.origin_y = max_y - ny * o.resolution;
  o.data.assign(static_cast<size_t>(nx) * ny, -1);
  const float blocked_below = 0.5f * free_value;
  for (int cj = 0; cj < ny; ++cj) {
    for (int ci = 0; ci < nx; ++ci) {
      bool any = false, blocked = false;
      for (int j = cj * k; j < std::min(cols, cj * k + k) && !blocked; ++j) {     // column-major: j outer
        for (int i = ci * k; i < std::min(rows, ci * k + k); ++i) {
          const float v = layer(i, j);
          if (std::isnan(v)) continue;
          any = true;
          if (v < blocked_below) {
            blocked = true;
            break;
          }
        }
      }
      const int u = nx - 1 - ci, w = ny - 1 - cj;   // from the min corner
      o.data[static_cast<size_t>(w) * nx + u] = !any ? -1 : (blocked ? 100 : 0);
    }
  }
  return o;
}

}  // namespace preprocess
}  // namespace planning
