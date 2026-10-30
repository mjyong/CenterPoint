#include "lidar3d/detections.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lidar3d {

void Detections::push(const Box& b, const std::array<double, 2>& v, double score, int label, int subclass) {
  boxes.push_back(b);
  velocities.push_back(v);
  scores.push_back(score);
  labels.push_back(label);
  subclasses.push_back(subclass == -2 ? default_subclass(label) : subclass);
}

Detections Detections::select(const std::vector<int>& idx) const {
  Detections o;
  o.frame = frame;
  o.stamp = stamp;
  for (int i : idx) o.push(boxes[i], velocities[i], scores[i], labels[i], subclasses[i]);
  return o;
}

Detections Detections::above(const PerClass& thr) const {
  std::vector<int> idx;
  for (size_t i = 0; i < size(); ++i)
    if (scores[i] >= thr[labels[i]]) idx.push_back(static_cast<int>(i));
  return select(idx);
}

Detections Detections::transform(const Mat4& T, const std::string& new_frame) const {
  const Mat3 R = T.topLeftCorner<3, 3>();
  const double dyaw = std::atan2(R(1, 0), R(0, 0));
  Detections o = *this;
  o.frame = new_frame;
  for (size_t i = 0; i < size(); ++i) {
    const Box& b = boxes[i];
    // boxes[:, :3] @ R.T + t, element by element as the 3-term dot product
    for (int r = 0; r < 3; ++r) o.boxes[i][r] = (b[0] * R(r, 0) + b[1] * R(r, 1)) + b[2] * R(r, 2) + T(r, 3);
    o.boxes[i][6] = wrap_angle(b[6] + dyaw);
    const auto& v = velocities[i];
    o.velocities[i] = {v[0] * R(0, 0) + v[1] * R(0, 1), v[0] * R(1, 0) + v[1] * R(1, 1)};
  }
  return o;
}

std::vector<int> circle_nms(const std::vector<std::array<double, 2>>& xy, const std::vector<double>& scores,
                            double radius, int max_keep, double weak_radius, double weak_ratio) {
  const size_t n = scores.size();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -scores[a] < -scores[b]; });
  std::vector<char> suppressed(n, 0);
  std::vector<int> keep;
  const double r2 = radius * radius;
  const double w2 = weak_radius ? weak_radius * weak_radius : 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (suppressed[i]) continue;
    const int oi = order[i];
    keep.push_back(oi);
    if (max_keep >= 0 && static_cast<int>(keep.size()) >= max_keep) break;
    for (size_t j = i + 1; j < n; ++j) {
      const int oj = order[j];
      const double dx = xy[oj][0] - xy[oi][0], dy = xy[oj][1] - xy[oi][1];
      const double d2 = dx * dx + dy * dy;
      if (d2 <= r2) suppressed[j] = 1;
      if (w2 > r2 && d2 <= w2 && scores[oj] <= weak_ratio * scores[oi]) suppressed[j] = 1;
    }
  }
  return keep;
}

Detections classwise_circle_nms(const Detections& d, const MergedNmsMap& radii) {
  std::vector<int> keep;
  for (int c = 0; c < kNumClasses; ++c) {
    std::vector<int> idx;
    for (size_t i = 0; i < d.size(); ++i)
      if (d.labels[i] == c) idx.push_back(static_cast<int>(i));
    if (idx.empty()) continue;
    const MergedNms& r = radii[c];
    if (!r.set || (r.radius <= 0 && !r.weak_radius)) {
      keep.insert(keep.end(), idx.begin(), idx.end());
      continue;
    }
    std::vector<std::array<double, 2>> xy;
    std::vector<double> sc;
    for (int i : idx) {
      xy.push_back({d.boxes[i][0], d.boxes[i][1]});
      sc.push_back(d.scores[i]);
    }
    for (int k : circle_nms(xy, sc, r.radius, -1, r.weak_radius, r.weak_ratio)) keep.push_back(idx[k]);
  }
  std::sort(keep.begin(), keep.end());
  return d.select(keep);
}

void det3d_to_standard(const double* b, Box& s, std::array<double, 2>& v) {
  s = {b[0], b[1], b[2], b[4], b[3], b[5], wrap_angle(-b[8] - kPi / 2)};
  v = {b[6], b[7]};
}

void ModelFrame::points_to_model(float* pts, size_t n, int stride) const {
  const Mat3 R64 = R();
  float R[3][3];                                       // self.R.T.astype(float32), used as pts @ R.T
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i][j] = static_cast<float>(R64(i, j));
  const float dz = static_cast<float>(z_offset);
  for (size_t i = 0; i < n; ++i) {
    float* p = pts + i * stride;
    const float x = p[0], y = p[1], z = p[2];
    p[0] = (x * R[0][0] + y * R[0][1]) + z * R[0][2];
    p[1] = (x * R[1][0] + y * R[1][1]) + z * R[1][2];
    p[2] = ((x * R[2][0] + y * R[2][1]) + z * R[2][2]) + dz;
  }
}

void ModelFrame::box_from_model(Box& b, std::array<double, 2>& v) const {
  const Mat3 R64 = R();
  b[2] -= z_offset;
  const double x = b[0], y = b[1], z = b[2];
  // b[:, :3] @ R
  b[0] = (x * R64(0, 0) + y * R64(1, 0)) + z * R64(2, 0);
  b[1] = (x * R64(0, 1) + y * R64(1, 1)) + z * R64(2, 1);
  b[2] = (x * R64(0, 2) + y * R64(1, 2)) + z * R64(2, 2);
  b[6] = wrap_angle(b[6] - yaw());
  const double vx = v[0], vy = v[1];
  v = {vx * R64(0, 0) + vy * R64(1, 0), vx * R64(0, 1) + vy * R64(1, 1)};
}

}  // namespace lidar3d
