#include "lidar3d/detection/detector.hpp"

#include <algorithm>
#include <chrono>

#include "lidar3d/detection/filters.hpp"

namespace lidar3d {

namespace {

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

}  // namespace

PillarDetector::PillarDetector(ModelMeta meta, std::unique_ptr<PillarNetwork> net, DetectionFilterConfig filter)
    : meta_(std::move(meta)), net_(std::move(net)), filter_(filter) {}

std::vector<float> PillarDetector::model_input(const std::vector<float>& points_det) const {
  const size_t n = points_det.size() / 5;
  const int C = meta_.num_point_features;
  const float z0 = static_cast<float>(meta_.z_range[0]), z1 = static_cast<float>(meta_.z_range[1]);
  std::vector<float> out;
  out.reserve(n * C);
  for (size_t i = 0; i < n; ++i) {
    const float* p = &points_det[5 * i];
    if (!(p[2] >= z0 && p[2] <= z1)) continue;
    out.insert(out.end(), p, p + C);
  }
  meta_.frame.points_to_model(out.data(), out.size() / C, C);
  return out;
}

Detections PillarDetector::detect(const std::vector<float>& points_det, double stamp) {
  const ModelMeta& m = meta_;
  const auto t0 = Clock::now();
  const std::vector<float> pts = model_input(points_det);
  Detections empty;
  empty.stamp = stamp;
  if (pts.empty()) {
    last_raw_ = empty;
    return empty;
  }
  const int C = m.num_point_features;
  Voxels v = voxelize(pts.data(), pts.size() / C, C, m.voxel_size, m.pc_range, m.max_points, m.max_voxels);
  const int F = C + 5;
  const std::vector<float> feats = decorate_pillars(v, m.voxel_size, m.pc_range);
  const int used = std::min(v.num_voxels, m.max_pillars);
  const int P = net_->dynamic_pillars() ? std::max(used, 1) : m.max_pillars;
  std::vector<float> pfn_in = pad_pillars(feats, v.num_voxels, m.max_points, F, P);
  last_pillars_ = {used, v.num_voxels};
  const auto t2 = Clock::now();
  std::vector<float> pf = net_->pfn(pfn_in, P);
  const auto t3 = Clock::now();
  const int nx = m.grid_size[0], ny = m.grid_size[1];
  std::vector<float> bev = scatter_pillars(pf.data(), m.bev_channels, P, v.coords, used, nx, ny);
  const auto t4 = Clock::now();
  std::vector<std::vector<float>> heads = net_->rpn(bev);
  const auto t5 = Clock::now();
  RawBoxes raw = decode_centerpoint(split_head_outputs(heads, m), m);
  Detections dets = postprocess(raw, m, stamp);
  last_raw_ = dets;
  if (filter_.enabled) dets = filter_detections(dets, points_det.data(), points_det.size() / 5, 5, filter_);
  const auto t6 = Clock::now();
  last_timing_ = {{"voxelize_ms", ms(t0, t2)}, {"pfn_ms", ms(t2, t3)}, {"scatter_ms", ms(t3, t4)},
                  {"rpn_ms", ms(t4, t5)}, {"network_ms", ms(t2, t5)}, {"post_ms", ms(t5, t6)}};
  for (const auto& [k, val] : net_->timing()) last_timing_["net_" + k] = val;
  if (keep_stages) {
    stages_.voxels = std::move(v);
    stages_.pfn_in = std::move(pfn_in);
    stages_.pfn_out = std::move(pf);
    stages_.bev = std::move(bev);
    stages_.heads = std::move(heads);
    stages_.raw = std::move(raw);
  }
  return dets;
}

}  // namespace lidar3d
