// Tier-1 prediction: IMM rollout, one predicted mode per IMM model, modes closer than merge_dist over
// the whole horizon merged by moment matching (prediction/kinematic.py).
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "lidar3d/config.hpp"
#include "lidar3d/tracking/tracker.hpp"

namespace lidar3d {

struct Prediction {
  int track_id = 0;
  int label = 0;
  double stamp = 0.0;
  std::vector<double> times;                       // (T,) seconds ahead of stamp
  std::vector<std::vector<Vec2>> modes;            // (K, T) xy
  std::vector<double> probs;                       // (K,)
  std::vector<std::vector<Mat2>> covs;             // (K, T) 2x2
  std::string frame = "world";
  Prediction transform(const Mat4& T_new_old, const std::string& new_frame) const;
};

void merge_modes(const std::vector<std::vector<Vec2>>& means, const std::vector<std::vector<Mat2>>& covs,
                 const std::vector<double>& probs, double merge_dist, std::vector<std::vector<Vec2>>& out_m,
                 std::vector<std::vector<Mat2>>& out_c, std::vector<double>& out_p);

class IMMPredictor {
 public:
  explicit IMMPredictor(PredictorConfig cfg = {}) : cfg_(cfg) {}
  Prediction predict(const IMM& imm, int track_id, int label, double stamp) const;
  std::vector<Prediction> operator()(const MultiObjectTracker& tracker) const;   // the reported tracks

 private:
  PredictorConfig cfg_;
};

}  // namespace lidar3d
