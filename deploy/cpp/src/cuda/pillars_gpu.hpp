// GPU pillar preprocessing (pillars.cu): the same voxels, PFN input and BEV canvas as the CPU reference
// (detection/pillars.cpp), computed on the device. Internal to the CUDA backend.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "dogperc/detection/pillars.hpp"

namespace dogperc::cuda {

void check(cudaError_t e, const char* what);

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void resize(size_t n) {
    if (n <= cap_) {
      n_ = n;
      return;
    }
    release();
    check(cudaMalloc(&p_, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    cap_ = n_ = n;
  }
  T* get() { return p_; }
  const T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  void release() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    cap_ = n_ = 0;
  }
  T* p_ = nullptr;
  size_t n_ = 0, cap_ = 0;
};

class PillarPreprocessor {
 public:
  explicit PillarPreprocessor(const ModelMeta& meta);
  // d_points: (n, 5) det-frame points on the device. Writes the PFN input (1, F, P, N) into d_pfn_in with
  // P = pfn_pillars (the engine's pillar count: max_pillars, or the used count for a dynamic PFN, which
  // is then returned by used_pillars() before this call fills it: call voxelize() first).
  // Returns the number of voxels found (> max_pillars: some are dropped).
  int voxelize(const float* d_points, size_t n, cudaStream_t s);
  int used_pillars() const { return used_; }
  void decorate(float* d_pfn_in, int pfn_pillars, cudaStream_t s);
  // d_pf: (C, P) PFN output -> (1, C, ny, nx) canvas in d_bev
  void scatter(const float* d_pf, int pfn_pillars, float* d_bev, cudaStream_t s);
  // the voxels of the last frame, as voxelize() on the CPU returns them (parity tests)
  Voxels download(cudaStream_t s) const;

 private:
  ModelMeta meta_;
  int V_ = 0, used_ = 0;
  size_t n_ = 0;
  DeviceBuffer<float> model_;          // (n, C) model-frame points
  DeviceBuffer<int64_t> lin_, lin_sorted_, seg_lin_;
  DeviceBuffer<int> idx_, flag_, seg_, head_pos_, first_idx_, keys_, seg_order_, rank_, point_index_, num_points_,
      coords_;
};

}  // namespace dogperc::cuda
