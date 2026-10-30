// TensorRT engines (TensorRT 10, LIDAR3D_WITH_CUDA): one engine with named IO tensors, and the Pillar
// network on two of them with host copies in and out (the reference of GpuPillarDetector, as
// ExportedPillarDetector(runtime="trt") is in Python).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lidar3d/detection/detector.hpp"

namespace lidar3d {

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& path);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const std::string& path() const;
  const std::vector<std::string>& inputs() const;
  const std::vector<std::string>& outputs() const;
  std::vector<int64_t> shape(const std::string& name) const;        // engine shape, -1 where dynamic
  std::vector<int64_t> max_shape(const std::string& name) const;    // largest allowed (profile 0 for inputs)
  bool dynamic(const std::string& name) const;
  void set_input_shape(const std::string& name, const std::vector<int64_t>& dims);
  std::vector<int64_t> tensor_shape(const std::string& name) const; // with the current input shapes
  void set_address(const std::string& name, void* device_ptr);
  void enqueue(void* cuda_stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// <model_dir>/<graph>.<precision>.engine; an int8* precision without an INT8 PFN uses pfn.fp16.engine
std::string engine_file(const std::string& model_dir, const std::string& graph, const std::string& precision);

// PillarDetector on the two engines, host copies around each (model_meta.json next to them)
std::unique_ptr<Detector> make_trt_detector(const std::string& model_dir, const std::string& precision,
                                            const DetectionFilterConfig& filter);

}  // namespace lidar3d
