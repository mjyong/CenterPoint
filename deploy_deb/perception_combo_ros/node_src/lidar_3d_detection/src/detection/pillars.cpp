#include "lidar3d/detection/pillars.hpp"

#include <cmath>
#include <unordered_map>

namespace lidar3d {

std::array<int, 3> grid_size(const std::array<double, 3>& vs, const std::array<double, 6>& r) {
  std::array<int, 3> g{};
  for (int k = 0; k < 3; ++k) {
    const float d = (static_cast<float>(r[3 + k]) - static_cast<float>(r[k])) / static_cast<float>(vs[k]);
    g[k] = static_cast<int>(std::nearbyint(d));
  }
  return g;
}

Voxels voxelize(const float* pts, size_t n, int C, const std::array<double, 3>& voxel_size,
                const std::array<double, 6>& pc_range, int max_points, int max_voxels) {
  // the cell of a point is floor((p - range_min) / voxel_size) in float32, the numba kernel's dtype;
  // voxels are numbered in the order of their first point, a voxel keeps its first max_points points
  const float vs[3] = {static_cast<float>(voxel_size[0]), static_cast<float>(voxel_size[1]),
                       static_cast<float>(voxel_size[2])};
  const float lo[3] = {static_cast<float>(pc_range[0]), static_cast<float>(pc_range[1]),
                       static_cast<float>(pc_range[2])};
  const auto grid = grid_size(voxel_size, pc_range);
  Voxels v;
  v.max_points = max_points;
  v.channels = C;
  std::unordered_map<int64_t, int> id_of;
  id_of.reserve(n);
  int next_id = 0;
  for (size_t i = 0; i < n; ++i) {
    const float* p = pts + i * C;
    int64_t c[3];
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
      const float cell = std::floor((p[k] - lo[k]) / vs[k]);
      if (!(cell >= 0 && cell < static_cast<float>(grid[k]))) {
        ok = false;
        break;
      }
      c[k] = static_cast<int64_t>(cell);
    }
    if (!ok) continue;
    const int64_t lin = (c[2] * grid[1] + c[1]) * grid[0] + c[0];
    auto it = id_of.find(lin);
    int vid;
    if (it == id_of.end()) {
      vid = next_id++;
      id_of.emplace(lin, vid);
      if (vid < max_voxels) {
        v.voxels.resize(static_cast<size_t>(vid + 1) * max_points * C, 0.0f);
        v.coords.insert(v.coords.end(), {static_cast<int32_t>(c[2]), static_cast<int32_t>(c[1]), static_cast<int32_t>(c[0])});
        v.num_points.push_back(0);
      }
    } else {
      vid = it->second;
    }
    if (vid >= max_voxels) continue;               // a cell after max_voxels: its points are dropped
    int32_t& np = v.num_points[vid];
    if (np < max_points) {
      float* dst = &v.voxels[(static_cast<size_t>(vid) * max_points + np) * C];
      for (int k = 0; k < C; ++k) dst[k] = p[k];
      ++np;
    }
  }
  v.num_voxels = static_cast<int>(v.num_points.size());
  return v;
}

std::vector<float> decorate_pillars(const Voxels& v, const std::array<double, 3>& voxel_size,
                                    const std::array<double, 6>& pc_range) {
  const int P = v.num_voxels, N = v.max_points, C = v.channels, F = C + 5;
  std::vector<float> out(static_cast<size_t>(P) * N * F, 0.0f);
  // cx = coord * vs + vs / 2 + range_min: a float32 array with Python floats, so float32 arithmetic
  const float vs0 = static_cast<float>(voxel_size[0]), vs1 = static_cast<float>(voxel_size[1]);
  const float h0 = static_cast<float>(voxel_size[0] / 2), h1 = static_cast<float>(voxel_size[1] / 2);
  const float r0 = static_cast<float>(pc_range[0]), r1 = static_cast<float>(pc_range[1]);
  for (int p = 0; p < P; ++p) {
    const float* vox = &v.voxels[static_cast<size_t>(p) * N * C];
    float s[3] = {0.0f, 0.0f, 0.0f};               // float32 sum over the slots, padded ones included
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < 3; ++k) s[k] += vox[j * C + k];
    const int np = v.num_points[p];
    // float32 / int32 promotes to float64 in NumPy
    const double cnt = static_cast<double>(np > 1 ? np : 1);
    const double mean[3] = {static_cast<double>(s[0]) / cnt, static_cast<double>(s[1]) / cnt, static_cast<double>(s[2]) / cnt};
    const float cx = ((static_cast<float>(v.coords[3 * p + 2]) * vs0) + h0) + r0;
    const float cy = ((static_cast<float>(v.coords[3 * p + 1]) * vs1) + h1) + r1;
    for (int j = 0; j < np && j < N; ++j) {
      const float* q = vox + j * C;
      float* o = &out[(static_cast<size_t>(p) * N + j) * F];
      for (int k = 0; k < C; ++k) o[k] = q[k];
      for (int k = 0; k < 3; ++k) o[C + k] = static_cast<float>(static_cast<double>(q[k]) - mean[k]);
      o[C + 3] = q[0] - cx;
      o[C + 4] = q[1] - cy;
    }
  }
  return out;
}

std::vector<float> pad_pillars(const std::vector<float>& feats, int V, int N, int F, int P) {
  std::vector<float> out(static_cast<size_t>(F) * P * N, 0.0f);
  const int n = V < P ? V : P;
  for (int p = 0; p < n; ++p)
    for (int j = 0; j < N; ++j)
      for (int f = 0; f < F; ++f)
        out[(static_cast<size_t>(f) * P + p) * N + j] = feats[(static_cast<size_t>(p) * N + j) * F + f];
  return out;
}

std::vector<float> scatter_pillars(const float* pf, int C, int P, const std::vector<int32_t>& coords, int num_valid,
                                   int nx, int ny) {
  std::vector<float> canvas(static_cast<size_t>(C) * ny * nx, 0.0f);
  for (int p = 0; p < num_valid; ++p) {
    const size_t idx = static_cast<size_t>(coords[3 * p + 1]) * nx + coords[3 * p + 2];
    for (int c = 0; c < C; ++c) canvas[static_cast<size_t>(c) * ny * nx + idx] = pf[static_cast<size_t>(c) * P + p];
  }
  return canvas;
}

}  // namespace lidar3d
