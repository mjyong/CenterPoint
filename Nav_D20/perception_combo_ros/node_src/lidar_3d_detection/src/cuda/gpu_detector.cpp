#include "lidar3d/detection/gpu_detector.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>

#include "lidar3d/detection/filters.hpp"
#include "lidar3d/detection/trt_engine.hpp"
#include "pillars_gpu.hpp"

namespace lidar3d {

namespace {

using cuda::check;
using cuda::DeviceBuffer;

int64_t volume(const std::vector<int64_t>& v) {
  int64_t n = 1;
  for (int64_t x : v) n *= x;
  return n;
}

class GpuPillarDetector : public Detector {
 public:
  GpuPillarDetector(ModelMeta m, const std::string& model_dir, const std::string& precision, DetectionFilterConfig filter)
      : meta_(std::move(m)), filter_(filter), pre_(meta_), pfn_(engine_file(model_dir, "pfn", precision)),
        rpn_(engine_file(model_dir, "rpn_head", precision)) {
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    for (auto& e : ev_) check(cudaEventCreate(&e), "cudaEventCreate");
    dynamic_ = pfn_.dynamic(meta_.pfn_input);
    pfn_in_.resize(volume(pfn_.max_shape(meta_.pfn_input)));
    pfn_out_.resize(volume(pfn_.max_shape(meta_.pfn_output)));
    bev_.resize(volume(rpn_.shape(meta_.rpn_input)));
    for (const auto& name : meta_.rpn_outputs) heads_.emplace_back(new DeviceBuffer<float>(volume(rpn_.shape(name))));
    pfn_.set_address(meta_.pfn_input, pfn_in_.get());
    pfn_.set_address(meta_.pfn_output, pfn_out_.get());
    rpn_.set_address(meta_.rpn_input, bev_.get());
    for (size_t k = 0; k < meta_.rpn_outputs.size(); ++k) rpn_.set_address(meta_.rpn_outputs[k], heads_[k]->get());
  }

  ~GpuPillarDetector() override {
    for (auto& e : ev_) cudaEventDestroy(e);
    cudaStreamDestroy(stream_);
  }

  Detections detect(const std::vector<float>& points_det, double stamp) override {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = points_det.size() / 5;
    Detections empty;
    empty.stamp = stamp;
    points_.resize(points_det.size());
    cudaEventRecord(ev_[0], stream_);
    check(cudaMemcpyAsync(points_.get(), points_det.data(), points_det.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "h2d");
    cudaEventRecord(ev_[7], stream_);
    const int V = pre_.voxelize(points_.get(), n, stream_);      // waits for the cell count
    cudaEventRecord(ev_[1], stream_);
    const int used = pre_.used_pillars();
    if (V == 0) {
      check(cudaStreamSynchronize(stream_), "detect");
      last_raw_ = empty;
      return empty;
    }
    const int P = dynamic_ ? std::max(used, 1) : meta_.max_pillars;
    if (dynamic_) pfn_.set_input_shape(meta_.pfn_input, {1, meta_.pfn_channels, P, meta_.max_points});
    pre_.decorate(pfn_in_.get(), P, stream_);
    cudaEventRecord(ev_[2], stream_);
    pfn_.enqueue(stream_);
    cudaEventRecord(ev_[3], stream_);
    pre_.scatter(pfn_out_.get(), P, bev_.get(), stream_);
    cudaEventRecord(ev_[4], stream_);
    rpn_.enqueue(stream_);
    cudaEventRecord(ev_[5], stream_);
    std::vector<std::vector<float>> heads;
    for (auto& h : heads_) {
      heads.emplace_back(h->size());
      check(cudaMemcpyAsync(heads.back().data(), h->get(), h->size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h");
    }
    cudaEventRecord(ev_[6], stream_);
    check(cudaStreamSynchronize(stream_), "detect");
    const auto t1 = std::chrono::steady_clock::now();
    RawBoxes raw = decode_centerpoint(split_head_outputs(std::move(heads), meta_), meta_);
    Detections dets = postprocess(raw, meta_, stamp);
    last_raw_ = dets;
    if (filter_.enabled) dets = filter_detections(dets, points_det.data(), n, 5, filter_);
    const auto t2 = std::chrono::steady_clock::now();
    auto el = [&](int a, int b) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev_[a], ev_[b]);
      return static_cast<double>(ms);
    };
    // voxelize_ms: voxels and the PFN input; h2d_ms: the point cloud upload before it (pageable memory)
    last_timing_ = {{"h2d_ms", el(0, 7)}, {"voxelize_ms", el(7, 2)}, {"pfn_ms", el(2, 3)},
                    {"scatter_ms", el(3, 4)}, {"rpn_ms", el(4, 5)}, {"d2h_ms", el(5, 6)},
                    {"network_ms", el(2, 5)}, {"gpu_ms", el(0, 6)},
                    {"post_ms", std::chrono::duration<double, std::milli>(t2 - t1).count()},
                    {"total_ms", std::chrono::duration<double, std::milli>(t2 - t0).count()},
                    {"pillars", static_cast<double>(used)}, {"voxels", static_cast<double>(V)}};
    return dets;
  }

 private:
  ModelMeta meta_;
  DetectionFilterConfig filter_;
  cuda::PillarPreprocessor pre_;
  TrtEngine pfn_, rpn_;
  bool dynamic_ = false;
  cudaStream_t stream_{};
  cudaEvent_t ev_[8]{};
  DeviceBuffer<float> points_, pfn_in_, pfn_out_, bev_;
  std::vector<std::unique_ptr<DeviceBuffer<float>>> heads_;
};

}  // namespace

std::unique_ptr<Detector> make_gpu_detector(const std::string& model_dir, const std::string& precision,
                                            const DetectionFilterConfig& filter) {
  ModelMeta m = load_model_meta((std::filesystem::path(model_dir) / "model_meta.json").string());
  return std::make_unique<GpuPillarDetector>(std::move(m), model_dir, precision, filter);
}

}  // namespace lidar3d
