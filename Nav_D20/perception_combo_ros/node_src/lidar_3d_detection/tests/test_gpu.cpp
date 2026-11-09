// GPU pillar preprocessing against the CPU reference (identical voxels, PFN input and canvas), and with
// LIDAR3D_MODEL_DIR (an exported model directory with its engines built on this machine) the GPU detector
// against the host-copy TensorRT detector on the golden frames.
#include <cuda_runtime.h>

#include <cstdlib>

#include "cuda/pillars_gpu.hpp"
#include "lidar3d/detection/gpu_detector.hpp"
#include "lidar3d/detection/trt_engine.hpp"
#include "golden.hpp"

namespace lidar3d::test {

namespace {

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

template <class T>
std::vector<T> d2h(const T* d, size_t n) {
  std::vector<T> h(n);
  cuda::check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
  return h;
}

}  // namespace

TEST(Gpu, PillarPreprocessingMatchesTheCpu) {
  REQUIRE_GOLDEN();
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const ModelMeta m = load_model_meta(golden("model/model_meta.json"));
  const Npz z = load("pillars.npz");
  const auto pts = member(z, "points_det").to<float>();
  // CPU reference
  PillarDetector cpu(m, nullptr, {});
  const auto model = cpu.model_input(pts);
  const Voxels v = voxelize(model.data(), model.size() / 5, 5, m.voxel_size, m.pc_range, m.max_points, m.max_voxels);
  const auto feats = decorate_pillars(v, m.voxel_size, m.pc_range);
  // GPU
  cuda::DeviceBuffer<float> d_pts(pts.size());
  cuda::check(cudaMemcpy(d_pts.get(), pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  cuda::PillarPreprocessor pre(m);
  ASSERT_EQ(pre.voxelize(d_pts.get(), pts.size() / 5, nullptr), v.num_voxels);
  const Voxels g = pre.download(nullptr);
  EXPECT_EQ(g.coords, v.coords);
  EXPECT_EQ(g.num_points, v.num_points);
  EXPECT_EQ(g.voxels, v.voxels);
  for (const int P : {m.max_pillars, pre.used_pillars()}) {     // static and dynamic PFN layouts
    cuda::DeviceBuffer<float> d_in(static_cast<size_t>(10) * P * m.max_points);
    pre.decorate(d_in.get(), P, nullptr);
    EXPECT_EQ(d2h(d_in.get(), d_in.size()), pad_pillars(feats, v.num_voxels, m.max_points, 10, P)) << "P " << P;
  }
  const auto pf = member(z, "pfn_out").to<float>();
  cuda::DeviceBuffer<float> d_pf(pf.size()), d_bev(static_cast<size_t>(m.bev_channels) * m.grid_size[0] * m.grid_size[1]);
  cuda::check(cudaMemcpy(d_pf.get(), pf.data(), pf.size() * sizeof(float), cudaMemcpyHostToDevice), "h2d");
  pre.scatter(d_pf.get(), m.max_pillars, d_bev.get(), nullptr);
  EXPECT_EQ(d2h(d_bev.get(), d_bev.size()), member(z, "bev").to<float>());
}

TEST(Gpu, GpuDetectorMatchesTheHostCopyDetector) {
  REQUIRE_GOLDEN();
  const char* dir = std::getenv("LIDAR3D_MODEL_DIR");
  if (!dir || !have_gpu()) GTEST_SKIP() << "set LIDAR3D_MODEL_DIR to an exported model with engines (and a GPU)";
  const char* prec = std::getenv("LIDAR3D_PRECISION");
  const std::string precision = prec ? prec : "fp16";
  auto host = make_trt_detector(dir, precision, DetectionFilterConfig{});
  auto gpu = make_gpu_detector(dir, precision, DetectionFilterConfig{});
  const Npz py = load("pipeline.npz");
  const auto off = member(py, "point_offsets").to<int64_t>();
  const auto all = member(py, "points").to<float>();
  for (size_t f = 0; f + 1 < off.size(); f += 5) {
    const std::vector<float> pts(all.begin() + 5 * off[f], all.begin() + 5 * off[f + 1]);
    const Detections a = host->detect(pts, 0.0), b = gpu->detect(pts, 0.0);
    // the same engines on the same input: identical up to the engines' own run-to-run noise
    expect_same_detections(a, b, 1e-4, "frame " + std::to_string(f));
    expect_same_detections(host->last_raw(), gpu->last_raw(), 1e-4, "raw frame " + std::to_string(f));
    std::printf("frame %zu: %zu boxes, gpu %.2f ms (voxelize %.2f, pfn %.2f, rpn %.2f), host-copy network %.2f ms\n", f,
                b.size(), gpu->last_timing().at("gpu_ms"), gpu->last_timing().at("voxelize_ms"),
                gpu->last_timing().at("pfn_ms"), gpu->last_timing().at("rpn_ms"), host->last_timing().at("network_ms"));
  }
}

}  // namespace lidar3d::test
