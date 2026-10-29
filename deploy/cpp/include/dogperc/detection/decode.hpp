// CenterHead decoding and the post-processing after the network (detection/decode.py, model_meta.postprocess).
// Arithmetic follows the Python dtypes: the head maps are float32, centers are computed in float64.
#pragma once

#include <vector>

#include "dogperc/detection/model_meta.hpp"

namespace dogperc {

// one head map (C, H, W) float32
struct HeadMap {
  int channels = 0, height = 0, width = 0;
  std::vector<float> data;
  float at(int c, int idx) const { return data[static_cast<size_t>(c) * height * width + idx]; }
};
// per task, in head_keys order
using TaskMaps = std::vector<std::vector<HeadMap>>;
// the RPN engine's outputs (meta.rpn_outputs order) -> per task maps
TaskMaps split_head_outputs(std::vector<std::vector<float>> outputs, const ModelMeta& meta);   // moves in rvalues

struct RawBoxes {                    // decode output, model frame
  std::vector<std::array<double, 9>> box9;   // det3d [x, y, z, w, l, h, vx, vy, r]
  std::vector<double> scores;
  std::vector<int> labels;           // model class (all tasks)
};

RawBoxes decode_centerpoint(const TaskMaps& maps, const ModelMeta& meta);
// class mapping to the dog classes, model frame -> det frame, merged-class NMS
Detections postprocess(const RawBoxes& raw, const ModelMeta& meta, double stamp);

}  // namespace dogperc
