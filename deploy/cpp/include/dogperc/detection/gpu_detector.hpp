// The deployed detector (DOGPERC_WITH_CUDA): voxelization, pillar decoration and scatter on the GPU
// (src/cuda/pillars.cu), the two TensorRT engines reading and writing device buffers, only the head maps
// copied back for the decoder. Same results as PillarDetector on the same engines (tests/test_gpu.cpp).
#pragma once

#include <memory>
#include <string>

#include "dogperc/detection/detector.hpp"

namespace dogperc {

std::unique_ptr<Detector> make_gpu_detector(const std::string& model_dir, const std::string& precision,
                                            const DetectionFilterConfig& filter);

}  // namespace dogperc
