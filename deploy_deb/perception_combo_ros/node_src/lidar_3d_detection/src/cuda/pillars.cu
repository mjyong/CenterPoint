// GPU voxelization, pillar decoration and scatter, identical to the CPU reference (detection/pillars.cpp).
//
// Voxels are numbered in the order of their first point and keep their first max_points points: the
// points are stably sorted by cell, the cells ordered by their first point, and a point's slot is its
// position within its cell. Float32 arithmetic uses round-to-nearest intrinsics (no FMA contraction),
// so the cells, and therefore the voxels, are the CPU's exactly.
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "pillars_gpu.hpp"

namespace lidar3d::cuda {

namespace {

constexpr int64_t kInvalid = std::numeric_limits<int64_t>::max();
constexpr int kBlock = 256;

int blocks(size_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

struct Params {
  float R[3][3];
  float dz;
  float z0, z1;
  float lo[3], vs[3];
  int grid[3];
  int C, N, F, max_voxels;
  float h0, h1, r0, r1;   // decorate: voxel_size / 2 and range min as float32
};

__global__ void k_prepare(const float* __restrict__ pts, int n, Params P, float* __restrict__ model,
                          int64_t* __restrict__ lin, int* __restrict__ idx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  idx[i] = i;
  lin[i] = kInvalid;
  const float* p = pts + 5 * i;
  const float z = p[2];
  if (!(z >= P.z0 && z <= P.z1)) return;              // det-frame z crop
  const float x = p[0], y = p[1];
  float m[3];
  for (int r = 0; r < 3; ++r)
    m[r] = __fadd_rn(__fadd_rn(__fmul_rn(x, P.R[r][0]), __fmul_rn(y, P.R[r][1])), __fmul_rn(z, P.R[r][2]));
  m[2] = __fadd_rn(m[2], P.dz);
  float* o = model + static_cast<size_t>(i) * P.C;
  o[0] = m[0];
  o[1] = m[1];
  o[2] = m[2];
  for (int k = 3; k < P.C; ++k) o[k] = p[k];
  int64_t c[3];
  for (int k = 0; k < 3; ++k) {
    const float cell = floorf(__fdiv_rn(__fsub_rn(m[k], P.lo[k]), P.vs[k]));
    if (!(cell >= 0.0f && cell < static_cast<float>(P.grid[k]))) return;
    c[k] = static_cast<int64_t>(cell);
  }
  lin[i] = (c[2] * P.grid[1] + c[1]) * P.grid[0] + c[0];
}

__global__ void k_flags(const int64_t* __restrict__ lin, int nv, int* __restrict__ flag) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nv) return;
  flag[j] = (j == 0 || lin[j] != lin[j - 1]) ? 1 : 0;
}

__global__ void k_heads(const int64_t* __restrict__ lin, const int* __restrict__ idx, const int* __restrict__ flag,
                        int* __restrict__ seg, int nv, int* __restrict__ head_pos, int* __restrict__ first_idx,
                        int64_t* __restrict__ seg_lin) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nv) return;
  const int s = seg[j] - 1;                            // inclusive scan of the flags -> segment index
  seg[j] = s;
  if (flag[j]) {
    head_pos[s] = j;
    first_idx[s] = idx[j];
    seg_lin[s] = lin[j];
  }
}

__global__ void k_rank(const int* __restrict__ order, int U, int* __restrict__ rank) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < U) rank[order[r]] = r;
}

__global__ void k_assign(const int* __restrict__ seg, const int* __restrict__ rank, const int* __restrict__ head_pos,
                         const int* __restrict__ idx, int nv, int N, int max_voxels, int* __restrict__ point_index) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nv) return;
  const int s = seg[j];
  const int vid = rank[s];
  if (vid >= max_voxels) return;                       // a cell after max_voxels: its points are dropped
  const int slot = j - head_pos[s];
  if (slot < N) point_index[static_cast<size_t>(vid) * N + slot] = idx[j];
}

__global__ void k_segments(const int* __restrict__ rank, const int* __restrict__ head_pos, const int64_t* __restrict__ seg_lin,
                           int U, int nv, Params P, int* __restrict__ num_points, int* __restrict__ coords) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= U) return;
  const int vid = rank[s];
  if (vid >= P.max_voxels) return;
  const int len = (s + 1 < U ? head_pos[s + 1] : nv) - head_pos[s];
  num_points[vid] = len < P.N ? len : P.N;
  const int64_t l = seg_lin[s];
  coords[3 * vid] = static_cast<int>(l / (static_cast<int64_t>(P.grid[0]) * P.grid[1]));
  coords[3 * vid + 1] = static_cast<int>((l / P.grid[0]) % P.grid[1]);
  coords[3 * vid + 2] = static_cast<int>(l % P.grid[0]);
}

// one thread per pillar slot of the PFN input (1, F, Pl, N): pillars beyond `used` are zero
__global__ void k_decorate(const float* __restrict__ model, const int* __restrict__ point_index,
                           const int* __restrict__ num_points, const int* __restrict__ coords, int used, int Pl, Params P,
                           float* __restrict__ out) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= Pl) return;
  const int N = P.N, C = P.C, F = P.F;
  if (p >= used) {
    for (int f = 0; f < F; ++f)
      for (int j = 0; j < N; ++j) out[(static_cast<size_t>(f) * Pl + p) * N + j] = 0.0f;
    return;
  }
  const int np = num_points[p];
  float s[3] = {0.0f, 0.0f, 0.0f};                     // float32 sum in slot order
  for (int j = 0; j < np; ++j) {
    const float* q = model + static_cast<size_t>(point_index[static_cast<size_t>(p) * N + j]) * C;
    for (int k = 0; k < 3; ++k) s[k] = __fadd_rn(s[k], q[k]);
  }
  const double cnt = static_cast<double>(np > 1 ? np : 1);   // float32 / int32 -> float64, as NumPy
  const double mean[3] = {__ddiv_rn(static_cast<double>(s[0]), cnt), __ddiv_rn(static_cast<double>(s[1]), cnt),
                          __ddiv_rn(static_cast<double>(s[2]), cnt)};
  const float cx = __fadd_rn(__fadd_rn(__fmul_rn(static_cast<float>(coords[3 * p + 2]), P.vs[0]), P.h0), P.r0);
  const float cy = __fadd_rn(__fadd_rn(__fmul_rn(static_cast<float>(coords[3 * p + 1]), P.vs[1]), P.h1), P.r1);
  for (int j = 0; j < N; ++j) {
    float v[16];
    if (j < np) {
      const float* q = model + static_cast<size_t>(point_index[static_cast<size_t>(p) * N + j]) * C;
      for (int k = 0; k < C; ++k) v[k] = q[k];
      for (int k = 0; k < 3; ++k) v[C + k] = __double2float_rn(__dsub_rn(static_cast<double>(q[k]), mean[k]));
      v[C + 3] = __fsub_rn(q[0], cx);
      v[C + 4] = __fsub_rn(q[1], cy);
    } else {
      for (int f = 0; f < F; ++f) v[f] = 0.0f;
    }
    for (int f = 0; f < F; ++f) out[(static_cast<size_t>(f) * Pl + p) * N + j] = v[f];
  }
}

__global__ void k_scatter(const float* __restrict__ pf, int Pl, int used, int C, const int* __restrict__ coords, int nx,
                          int ny, float* __restrict__ bev) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= used * C) return;
  const int p = t % used, c = t / used;
  const size_t cell = static_cast<size_t>(coords[3 * p + 1]) * nx + coords[3 * p + 2];
  bev[static_cast<size_t>(c) * ny * nx + cell] = pf[static_cast<size_t>(c) * Pl + p];
}

Params params(const ModelMeta& m) {
  Params P{};
  const Mat3 R = m.frame.R();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) P.R[i][j] = static_cast<float>(R(i, j));
  P.dz = static_cast<float>(m.frame.z_offset);
  P.z0 = static_cast<float>(m.z_range[0]);
  P.z1 = static_cast<float>(m.z_range[1]);
  const auto g = grid_size(m.voxel_size, m.pc_range);
  for (int k = 0; k < 3; ++k) {
    P.lo[k] = static_cast<float>(m.pc_range[k]);
    P.vs[k] = static_cast<float>(m.voxel_size[k]);
    P.grid[k] = g[k];
  }
  P.C = m.num_point_features;
  P.N = m.max_points;
  P.F = m.num_point_features + 5;
  P.max_voxels = m.max_voxels;
  P.h0 = static_cast<float>(m.voxel_size[0] / 2);
  P.h1 = static_cast<float>(m.voxel_size[1] / 2);
  P.r0 = static_cast<float>(m.pc_range[0]);
  P.r1 = static_cast<float>(m.pc_range[1]);
  if (P.F > 16 || P.C > 11) throw std::runtime_error("pillars.cu: at most 11 point features");
  return P;
}

}  // namespace

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

PillarPreprocessor::PillarPreprocessor(const ModelMeta& meta) : meta_(meta) {
  if (grid_size(meta.voxel_size, meta.pc_range) != meta.grid_size) throw std::runtime_error("model_meta: grid_size");
}

int PillarPreprocessor::voxelize(const float* d_points, size_t n, cudaStream_t s) {
  const Params P = params(meta_);
  n_ = n;
  model_.resize(n * P.C);
  lin_.resize(n);
  lin_sorted_.resize(n);
  idx_.resize(n);
  V_ = used_ = 0;
  if (n == 0) return 0;
  k_prepare<<<blocks(n), kBlock, 0, s>>>(d_points, static_cast<int>(n), P, model_.get(), lin_.get(), idx_.get());
  check(cudaGetLastError(), "k_prepare");
  check(cudaMemcpyAsync(lin_sorted_.get(), lin_.get(), n * sizeof(int64_t), cudaMemcpyDeviceToDevice, s), "copy");
  const auto pol = thrust::cuda::par(alloc_).on(s);
  thrust::stable_sort_by_key(pol, lin_sorted_.get(), lin_sorted_.get() + n, idx_.get());
  const int nv = static_cast<int>(thrust::lower_bound(pol, lin_sorted_.get(), lin_sorted_.get() + n, kInvalid) - lin_sorted_.get());
  if (nv == 0) return 0;
  flag_.resize(nv);
  seg_.resize(nv);
  k_flags<<<blocks(nv), kBlock, 0, s>>>(lin_sorted_.get(), nv, flag_.get());
  thrust::inclusive_scan(pol, flag_.get(), flag_.get() + nv, seg_.get());
  int U = 0;
  check(cudaMemcpyAsync(&U, seg_.get() + nv - 1, sizeof(int), cudaMemcpyDeviceToHost, s), "copy U");
  check(cudaStreamSynchronize(s), "sync");
  head_pos_.resize(U);
  first_idx_.resize(U);
  seg_lin_.resize(U);
  seg_order_.resize(U);
  rank_.resize(U);
  k_heads<<<blocks(nv), kBlock, 0, s>>>(lin_sorted_.get(), idx_.get(), flag_.get(), seg_.get(), nv, head_pos_.get(),
                                        first_idx_.get(), seg_lin_.get());
  // order the cells by their first point (first indices are unique)
  keys_.resize(U);
  check(cudaMemcpyAsync(keys_.get(), first_idx_.get(), U * sizeof(int), cudaMemcpyDeviceToDevice, s), "copy");
  thrust::sequence(pol, seg_order_.get(), seg_order_.get() + U);
  thrust::sort_by_key(pol, keys_.get(), keys_.get() + U, seg_order_.get());
  k_rank<<<blocks(U), kBlock, 0, s>>>(seg_order_.get(), U, rank_.get());
  V_ = U < P.max_voxels ? U : P.max_voxels;
  point_index_.resize(static_cast<size_t>(V_) * P.N);
  num_points_.resize(V_);
  coords_.resize(3 * static_cast<size_t>(V_));
  check(cudaMemsetAsync(point_index_.get(), 0xff, point_index_.size() * sizeof(int), s), "memset");
  k_assign<<<blocks(nv), kBlock, 0, s>>>(seg_.get(), rank_.get(), head_pos_.get(), idx_.get(), nv, P.N, P.max_voxels,
                                         point_index_.get());
  k_segments<<<blocks(U), kBlock, 0, s>>>(rank_.get(), head_pos_.get(), seg_lin_.get(), U, nv, P, num_points_.get(),
                                          coords_.get());
  check(cudaGetLastError(), "voxelize kernels");
  used_ = V_ < meta_.max_pillars ? V_ : meta_.max_pillars;
  return V_;
}

void PillarPreprocessor::decorate(float* d_pfn_in, int Pl, cudaStream_t s) {
  const Params P = params(meta_);
  if (Pl <= 0) return;
  k_decorate<<<blocks(Pl), kBlock, 0, s>>>(model_.get(), point_index_.get(), num_points_.get(), coords_.get(),
                                            used_ < Pl ? used_ : Pl, Pl, P, d_pfn_in);
  check(cudaGetLastError(), "k_decorate");
}

void PillarPreprocessor::scatter(const float* d_pf, int Pl, float* d_bev, cudaStream_t s) {
  const int C = meta_.bev_channels, nx = meta_.grid_size[0], ny = meta_.grid_size[1];
  check(cudaMemsetAsync(d_bev, 0, static_cast<size_t>(C) * nx * ny * sizeof(float), s), "memset bev");
  const int used = used_ < Pl ? used_ : Pl;
  if (used == 0) return;
  k_scatter<<<blocks(static_cast<size_t>(used) * C), kBlock, 0, s>>>(d_pf, Pl, used, C, coords_.get(), nx, ny, d_bev);
  check(cudaGetLastError(), "k_scatter");
}

Voxels PillarPreprocessor::download(cudaStream_t s) const {
  const int C = meta_.num_point_features, N = meta_.max_points;
  Voxels v;
  v.num_voxels = V_;
  v.max_points = N;
  v.channels = C;
  if (V_ == 0) return v;
  std::vector<int> pidx(static_cast<size_t>(V_) * N);
  std::vector<float> model(n_ * C);
  v.num_points.resize(V_);
  v.coords.resize(3 * static_cast<size_t>(V_));
  check(cudaMemcpyAsync(pidx.data(), point_index_.get(), pidx.size() * sizeof(int), cudaMemcpyDeviceToHost, s), "d2h");
  check(cudaMemcpyAsync(model.data(), model_.get(), model.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "d2h");
  check(cudaMemcpyAsync(v.num_points.data(), num_points_.get(), V_ * sizeof(int), cudaMemcpyDeviceToHost, s), "d2h");
  check(cudaMemcpyAsync(v.coords.data(), coords_.get(), v.coords.size() * sizeof(int), cudaMemcpyDeviceToHost, s), "d2h");
  check(cudaStreamSynchronize(s), "sync");
  v.voxels.assign(static_cast<size_t>(V_) * N * C, 0.0f);
  for (size_t k = 0; k < pidx.size(); ++k)
    if (pidx[k] >= 0)
      for (int c = 0; c < C; ++c) v.voxels[k * C + c] = model[static_cast<size_t>(pidx[k]) * C + c];
  return v;
}

}  // namespace lidar3d::cuda
