// Detectors behind one call (detection/exported.py): det-frame cloud in, det-frame Detections out.
//
// PillarDetector runs the host steps of the deployed Pillar model around two networks: crop, model frame,
// voxelize, decorate, pad -> PFN -> scatter -> RPN + heads -> decode, class mapping, merged NMS, geometric
// filters. The networks are a PillarNetwork: TensorRT engines (src/cuda, LIDAR3D_WITH_CUDA) or anything that
// returns the head maps (the parity tests replay the Python ones). GpuPillarDetector (src/cuda) does the
// voxelization and scatter on the GPU as well.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lidar3d/config.hpp"
#include "lidar3d/detection/decode.hpp"
#include "lidar3d/detection/model_meta.hpp"
#include "lidar3d/detection/pillars.hpp"

namespace lidar3d {

class Detector {
 public:
  virtual ~Detector() = default;
  // points_det: (N, 5) [x, y, z, intensity, dt] float32
  virtual Detections detect(const std::vector<float>& points_det, double stamp) = 0;
  const Detections& last_raw() const { return last_raw_; }   // before the geometric filters
  const std::map<std::string, double>& last_timing() const { return last_timing_; }

 protected:
  Detections last_raw_;
  std::map<std::string, double> last_timing_;
};

class PillarNetwork {
 public:
  virtual ~PillarNetwork() = default;
  // true: the PFN takes any pillar count up to max_pillars (only the used ones are sent)
  virtual bool dynamic_pillars() const { return false; }
  // (1, F, P, N) -> (1, C, P, 1)
  virtual std::vector<float> pfn(const std::vector<float>& pfn_in, int pillars) = 0;
  // (1, C, ny, nx) -> the RPN outputs in model_meta rpn_outputs order, each (1, C_k, H, W)
  virtual std::vector<std::vector<float>> rpn(const std::vector<float>& bev) = 0;
  virtual std::map<std::string, double> timing() const { return {}; }
};

struct PillarStages {               // the last frame's intermediate tensors (keep_stages)
  Voxels voxels;
  std::vector<float> pfn_in, pfn_out, bev;
  std::vector<std::vector<float>> heads;
  RawBoxes raw;
};

class PillarDetector : public Detector {
 public:
  PillarDetector(ModelMeta meta, std::unique_ptr<PillarNetwork> net, DetectionFilterConfig filter);
  Detections detect(const std::vector<float>& points_det, double stamp) override;
  const ModelMeta& meta() const { return meta_; }
  // pillars used / found in the last frame (found > used: some dropped beyond max_pillars)
  std::pair<int, int> last_pillars() const { return last_pillars_; }
  bool keep_stages = false;
  const PillarStages& stages() const { return stages_; }

  // det-frame cloud -> model-frame network input (crop to z_range, first num_point_features columns)
  std::vector<float> model_input(const std::vector<float>& points_det) const;

 private:
  ModelMeta meta_;
  std::unique_ptr<PillarNetwork> net_;
  DetectionFilterConfig filter_;
  std::pair<int, int> last_pillars_{0, 0};
  PillarStages stages_;
};

}  // namespace lidar3d
