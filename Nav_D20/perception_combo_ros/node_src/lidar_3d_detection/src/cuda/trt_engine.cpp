#include "lidar3d/detection/trt_engine.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>

#include "lidar3d/npz.hpp"
#include "pillars_gpu.hpp"

namespace lidar3d {

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

std::vector<int64_t> dims(const nvinfer1::Dims& d) {
  std::vector<int64_t> v;
  for (int i = 0; i < d.nbDims; ++i) v.push_back(d.d[i]);
  return v;
}

nvinfer1::Dims to_dims(const std::vector<int64_t>& v) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(v.size());
  for (size_t i = 0; i < v.size(); ++i) d.d[i] = v[i];
  return d;
}

int64_t volume(const std::vector<int64_t>& v) {
  int64_t n = 1;
  for (int64_t x : v) n *= x;
  return n;
}

}  // namespace

struct TrtEngine::Impl {
  std::string path;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<std::string> inputs, outputs;
};

TrtEngine::TrtEngine(const std::string& path) : impl_(std::make_unique<Impl>()) {
  impl_->path = path;
  const std::vector<uint8_t> blob = read_file(path);
  impl_->runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!impl_->runtime) throw Error("TensorRT: createInferRuntime failed");
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw Error(path + ": TensorRT could not load the engine (built with another version / GPU?)");
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw Error(path + ": createExecutionContext failed");
  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    if (impl_->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
      throw Error(path + ": tensor " + name + " is not float32 (the exported graphs keep float32 IO)");
    (impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? impl_->inputs : impl_->outputs).push_back(name);
  }
  for (const auto& n : impl_->inputs)        // dynamic inputs start at their largest shape
    if (dynamic(n)) set_input_shape(n, max_shape(n));
}

TrtEngine::~TrtEngine() = default;

const std::string& TrtEngine::path() const { return impl_->path; }
const std::vector<std::string>& TrtEngine::inputs() const { return impl_->inputs; }
const std::vector<std::string>& TrtEngine::outputs() const { return impl_->outputs; }

std::vector<int64_t> TrtEngine::shape(const std::string& name) const {
  return dims(impl_->engine->getTensorShape(name.c_str()));
}

bool TrtEngine::dynamic(const std::string& name) const {
  for (int64_t d : shape(name))
    if (d < 0) return true;
  return false;
}

std::vector<int64_t> TrtEngine::max_shape(const std::string& name) const {
  if (!dynamic(name)) return shape(name);
  if (impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
    return dims(impl_->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX));
  // an output: its shape with every dynamic input at its maximum
  for (const auto& n : impl_->inputs)
    if (dynamic(n)) impl_->context->setInputShape(n.c_str(), to_dims(max_shape(n)));
  return tensor_shape(name);
}

void TrtEngine::set_input_shape(const std::string& name, const std::vector<int64_t>& d) {
  if (!impl_->context->setInputShape(name.c_str(), to_dims(d))) throw Error(impl_->path + ": invalid shape for " + name);
}

std::vector<int64_t> TrtEngine::tensor_shape(const std::string& name) const {
  return dims(impl_->context->getTensorShape(name.c_str()));
}

void TrtEngine::set_address(const std::string& name, void* ptr) {
  if (!impl_->context->setTensorAddress(name.c_str(), ptr)) throw Error(impl_->path + ": setTensorAddress " + name);
}

void TrtEngine::enqueue(void* stream) {
  if (!impl_->context->enqueueV3(static_cast<cudaStream_t>(stream))) throw Error(impl_->path + ": enqueueV3 failed");
}

std::string engine_file(const std::string& model_dir, const std::string& graph, const std::string& precision) {
  namespace fs = std::filesystem;
  const fs::path p = fs::path(model_dir) / (graph + "." + precision + ".engine");
  if (graph == "pfn" && precision.rfind("int8", 0) == 0 && !fs::exists(p))
    return (fs::path(model_dir) / "pfn.fp16.engine").string();
  return p.string();
}

namespace {

using cuda::check;
using cuda::DeviceBuffer;

// the Pillar network on two engines, every call copying its input in and its outputs out
class TrtNetwork : public PillarNetwork {
 public:
  TrtNetwork(const ModelMeta& m, const std::string& model_dir, const std::string& precision)
      : meta_(m), pfn_(engine_file(model_dir, "pfn", precision)), rpn_(engine_file(model_dir, "rpn_head", precision)) {
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    pfn_in_.resize(volume(pfn_.max_shape(m.pfn_input)));
    pfn_out_.resize(volume(pfn_.max_shape(m.pfn_output)));
    bev_.resize(volume(rpn_.shape(m.rpn_input)));
    for (const auto& name : m.rpn_outputs) heads_.emplace_back(new DeviceBuffer<float>(volume(rpn_.shape(name))));
    rpn_.set_address(m.rpn_input, bev_.get());
    for (size_t k = 0; k < m.rpn_outputs.size(); ++k) rpn_.set_address(m.rpn_outputs[k], heads_[k]->get());
    pfn_.set_address(m.pfn_input, pfn_in_.get());
    pfn_.set_address(m.pfn_output, pfn_out_.get());
  }
  ~TrtNetwork() override { cudaStreamDestroy(stream_); }

  bool dynamic_pillars() const override { return pfn_.dynamic(meta_.pfn_input); }

  std::vector<float> pfn(const std::vector<float>& in, int pillars) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (dynamic_pillars())
      pfn_.set_input_shape(meta_.pfn_input, {1, meta_.pfn_channels, pillars, meta_.max_points});
    check(cudaMemcpyAsync(pfn_in_.get(), in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "h2d");
    pfn_.enqueue(stream_);
    std::vector<float> out(volume(pfn_.tensor_shape(meta_.pfn_output)));
    check(cudaMemcpyAsync(out.data(), pfn_out_.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h");
    check(cudaStreamSynchronize(stream_), "pfn");
    timing_["pfn_host_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }

  std::vector<std::vector<float>> rpn(const std::vector<float>& bev) override {
    const auto t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(bev_.get(), bev.data(), bev.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "h2d");
    rpn_.enqueue(stream_);
    std::vector<std::vector<float>> out;
    for (auto& h : heads_) {
      out.emplace_back(h->size());
      check(cudaMemcpyAsync(out.back().data(), h->get(), h->size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h");
    }
    check(cudaStreamSynchronize(stream_), "rpn");
    timing_["rpn_host_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }

  std::map<std::string, double> timing() const override { return timing_; }

 private:
  ModelMeta meta_;
  TrtEngine pfn_, rpn_;
  cudaStream_t stream_{};
  DeviceBuffer<float> pfn_in_, pfn_out_, bev_;
  std::vector<std::unique_ptr<DeviceBuffer<float>>> heads_;
  std::map<std::string, double> timing_;
};

}  // namespace

std::unique_ptr<Detector> make_trt_detector(const std::string& model_dir, const std::string& precision,
                                            const DetectionFilterConfig& filter) {
  ModelMeta m = load_model_meta((std::filesystem::path(model_dir) / "model_meta.json").string());
  auto net = std::make_unique<TrtNetwork>(m, model_dir, precision);
  return std::make_unique<PillarDetector>(std::move(m), std::move(net), filter);
}

}  // namespace lidar3d
