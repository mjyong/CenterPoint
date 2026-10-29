// Host side of the Pillar network: voxelize, decorate, pad, scatter (detection/voxelize.py, pillar_host.py).
// The CPU reference of the CUDA kernels (src/cuda/pillars.cu); voxels identical to det3d's points_to_voxel.
#pragma once

#include <cstdint>
#include <vector>

#include "dogperc/detection/model_meta.hpp"

namespace dogperc {

struct Voxels {
  int num_voxels = 0;
  int max_points = 0;
  int channels = 0;
  std::vector<float> voxels;          // (V, max_points, C), zero padded
  std::vector<int32_t> coords;        // (V, 3) z, y, x
  std::vector<int32_t> num_points;    // (V,)
};

// points (N, C) float32 in the model frame
Voxels voxelize(const float* points, size_t n, int channels, const std::array<double, 3>& voxel_size,
                const std::array<double, 6>& pc_range, int max_points, int max_voxels);
std::array<int, 3> grid_size(const std::array<double, 3>& voxel_size, const std::array<double, 6>& pc_range);

// (V, N, C + 5) float32: [raw, xyz - pillar mean, xy - pillar center], padded slots zero
std::vector<float> decorate_pillars(const Voxels& v, const std::array<double, 3>& voxel_size,
                                    const std::array<double, 6>& pc_range);
// (1, F, P, N) PFN input from decorated pillars (V, N, F): the first min(V, P) pillars, zero padded
std::vector<float> pad_pillars(const std::vector<float>& feats, int num_voxels, int max_points, int features,
                               int pillars);
// (C, P) pillar features (PFN output (1, C, P, 1)) -> (1, C, ny, nx) BEV canvas; the first num_valid pillars
std::vector<float> scatter_pillars(const float* pillar_features, int channels, int pillars, const std::vector<int32_t>& coords,
                                   int num_valid, int nx, int ny);

}  // namespace dogperc
