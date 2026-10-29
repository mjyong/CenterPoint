// Detections, box conventions, circle NMS and the det <-> model frame (detection/boxes.py, frame_adapter.py).
//
// Standard box everywhere outside the network: [x, y, z, l, w, h, yaw] (geometric center, l along the
// heading, yaw CCW from +x); det3d's nuScenes box is [x, y, z, w, l, h, vx, vy, r] with r = -yaw - pi/2.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "dogperc/common.hpp"
#include "dogperc/geometry.hpp"

namespace dogperc {

using Box = std::array<double, 7>;

struct Detections {
  std::vector<Box> boxes;
  std::vector<std::array<double, 2>> velocities;   // over ground
  std::vector<double> scores;
  std::vector<int> labels;                         // kClasses
  std::vector<int> subclasses;                     // kSubclasses, -1 unknown
  std::string frame = "det";
  double stamp = kNaN;

  size_t size() const { return scores.size(); }
  bool empty() const { return scores.empty(); }
  void push(const Box& b, const std::array<double, 2>& v, double score, int label, int subclass = -2);
  Detections select(const std::vector<int>& idx) const;
  // score >= the class's threshold
  Detections above(const PerClass& thresholds) const;
  // into another gravity-aligned frame (yaw-only rotation + translation)
  Detections transform(const Mat4& T_new_old, const std::string& new_frame) const;
};

// greedy center-distance NMS (kept indices, by descending score; stable for ties). With weak_radius /
// weak_ratio a kept box also suppresses boxes within weak_radius scoring <= weak_ratio of its own
std::vector<int> circle_nms(const std::vector<std::array<double, 2>>& xy, const std::vector<double>& scores,
                            double radius, int max_keep = -1, double weak_radius = 0.0, double weak_ratio = 0.0);

struct MergedNms {           // one class of model_meta.json merged_nms: radius, or [radius, weak radius, ratio]
  bool set = false;
  double radius = 0.0;
  double weak_radius = 0.0;
  double weak_ratio = 0.0;
};
using MergedNmsMap = std::array<MergedNms, kNumClasses>;
Detections classwise_circle_nms(const Detections& d, const MergedNmsMap& radii);

// det3d box9 [x, y, z, w, l, h, vx, vy, r] -> standard box and velocity
void det3d_to_standard(const double* box9, Box& std_box, std::array<double, 2>& vel);

// det frame D -> network input frame M: x_M = Rz(+90 deg) x_D + [0, 0, z_offset] (virtual nuScenes lidar)
struct ModelFrame {
  bool nusc_axes = true;
  double z_offset = -1.84019;
  double yaw() const { return nusc_axes ? kPi / 2 : 0.0; }
  Mat3 R() const { return rot_z(yaw()); }
  // points [x, y, z, ...] (stride C) in place, float32 like the Python (R cast to float32)
  void points_to_model(float* pts, size_t n, int stride) const;
  void box_from_model(Box& b, std::array<double, 2>& v) const;
};

}  // namespace dogperc
