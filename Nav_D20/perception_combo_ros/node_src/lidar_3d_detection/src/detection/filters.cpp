#include "lidar3d/detection/filters.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lidar3d {

namespace {

// radius of the disk around a box center that contains the box grown by `margin` (and a little more)
double visit_radius(const Box& b, double margin) {
  return 0.5 * std::hypot(b[3] * (1 + margin), b[4] * (1 + margin)) + 1e-3;
}

// the points under the boxes' disks binned on a dense 2D grid (CSR), to visit only the points near a box;
// a point inside a box lies in its disk, so the cells overlapping the disk hold every candidate
class PointGrid {
 public:
  PointGrid(const std::vector<float>& xyz, const std::vector<Box>& boxes, double margin) {
    for (const Box& b : boxes) {
      const double r = visit_radius(b, margin);
      for (int k = 0; k < 2; ++k) {
        lo_[k] = std::min(lo_[k], b[k] - r);
        hi_[k] = std::max(hi_[k], b[k] + r);
      }
    }
    if (!(hi_[0] >= lo_[0] && hi_[1] >= lo_[1])) {   // no finite box: an empty grid
      start_.assign(2, 0);
      return;
    }
    cell_ = std::max(2.0, std::max(hi_[0] - lo_[0], hi_[1] - lo_[1]) / 1024.0);
    nx_ = static_cast<long>(std::floor((hi_[0] - lo_[0]) / cell_)) + 1;
    ny_ = static_cast<long>(std::floor((hi_[1] - lo_[1]) / cell_)) + 1;
    const size_t n = xyz.size() / 3;
    std::vector<long> of(n, -1);
    start_.assign(static_cast<size_t>(nx_ * ny_) + 1, 0);
    for (size_t i = 0; i < n; ++i) {
      const double x = xyz[3 * i], y = xyz[3 * i + 1];
      if (!(x >= lo_[0] && x <= hi_[0] && y >= lo_[1] && y <= hi_[1])) continue;
      of[i] = cell(x, y);
      ++start_[of[i] + 1];
    }
    for (size_t c = 1; c < start_.size(); ++c) start_[c] += start_[c - 1];
    idx_.resize(start_.back());
    std::vector<int> fill(start_.begin(), start_.end() - 1);
    for (size_t i = 0; i < n; ++i)
      if (of[i] >= 0) idx_[fill[of[i]]++] = static_cast<int>(i);
  }
  template <class F>
  void visit(double cx, double cy, double radius, F&& f) const {
    if (idx_.empty() || !(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(radius))) return;
    const long x0 = clampx(std::floor((cx - radius - lo_[0]) / cell_));
    const long x1 = clampx(std::floor((cx + radius - lo_[0]) / cell_));
    const long y0 = clampy(std::floor((cy - radius - lo_[1]) / cell_));
    const long y1 = clampy(std::floor((cy + radius - lo_[1]) / cell_));
    for (long x = x0; x <= x1; ++x)
      for (long y = y0; y <= y1; ++y) {
        const long c = x * ny_ + y;
        for (int k = start_[c]; k < start_[c + 1]; ++k) f(idx_[k]);
      }
  }

 private:
  long cell(double x, double y) const {
    return clampx(std::floor((x - lo_[0]) / cell_)) * ny_ + clampy(std::floor((y - lo_[1]) / cell_));
  }
  long clampx(double v) const { return std::clamp(static_cast<long>(v), 0L, nx_ - 1); }
  long clampy(double v) const { return std::clamp(static_cast<long>(v), 0L, ny_ - 1); }
  double lo_[2] = {kInf, kInf}, hi_[2] = {-kInf, -kInf};
  double cell_ = 2.0;
  long nx_ = 1, ny_ = 1;
  std::vector<int> start_, idx_;
};

std::vector<int64_t> box_counts(const std::vector<float>& xyz, const PointGrid& grid, const std::vector<Box>& boxes,
                                double margin) {
  std::vector<int64_t> counts(boxes.size(), 0);
  for (size_t b = 0; b < boxes.size(); ++b) {
    const Box& bx = boxes[b];
    const double l = bx[3] * (1 + margin), w = bx[4] * (1 + margin), h = bx[5] * (1 + margin);
    const double c = std::cos(bx[6]), s = std::sin(bx[6]);
    int64_t n = 0;
    // the ball around the center that cKDTree queries contains the box; the box test decides
    grid.visit(bx[0], bx[1], visit_radius(bx, margin), [&](int i) {
      const double dx = static_cast<double>(xyz[3 * i]) - bx[0], dy = static_cast<double>(xyz[3 * i + 1]) - bx[1];
      const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
      if (std::abs(lx) <= l / 2 && std::abs(ly) <= w / 2 && std::abs(static_cast<double>(xyz[3 * i + 2]) - bx[2]) <= h / 2) ++n;
    });
    counts[b] = n;
  }
  return counts;
}

}  // namespace

std::vector<int64_t> points_in_box_counts(const std::vector<float>& xyz, const std::vector<Box>& boxes, double margin) {
  if (xyz.empty() || boxes.empty()) return std::vector<int64_t>(boxes.size(), 0);
  const PointGrid grid(xyz, boxes, margin);
  return box_counts(xyz, grid, boxes, margin);
}

std::vector<double> local_ground_heights(const std::vector<float>& xyz_all, const std::vector<Box>& boxes, double cell,
                                         double ring, double percentile, int min_cells) {
  std::vector<double> out(boxes.size(), kNaN);
  if (xyz_all.empty() || boxes.empty()) return out;
  const double pad = ring + cell;
  double lo[2] = {kInf, kInf}, hi[2] = {-kInf, -kInf};
  for (const Box& b : boxes)
    for (int k = 0; k < 2; ++k) {
      lo[k] = std::min(lo[k], b[k]);
      hi[k] = std::max(hi[k], b[k]);
    }
  for (int k = 0; k < 2; ++k) {
    lo[k] -= pad;
    hi[k] += pad;
  }
  const long sh0 = static_cast<long>(std::floor((hi[0] - lo[0]) / cell)) + 1;
  const long sh1 = static_cast<long>(std::floor((hi[1] - lo[1]) / cell)) + 1;
  std::vector<double> grid(static_cast<size_t>(sh0) * sh1, kInf);   // lowest point per cell
  bool any = false;
  const size_t n = xyz_all.size() / 3;
  for (size_t i = 0; i < n; ++i) {
    const double x = xyz_all[3 * i], y = xyz_all[3 * i + 1];
    if (!(x >= lo[0] && y >= lo[1] && x < hi[0] && y < hi[1])) continue;
    any = true;
    const long a = static_cast<long>(std::floor((x - lo[0]) / cell)), b = static_cast<long>(std::floor((y - lo[1]) / cell));
    double& g = grid[static_cast<size_t>(a) * sh1 + b];
    g = std::min(g, static_cast<double>(xyz_all[3 * i + 2]));
  }
  if (!any) return out;
  // scipy.ndimage.percentile_filter / correlate with a disk footprint, mode constant (cval 1e6 / 0)
  const int r = static_cast<int>(std::ceil(ring / cell));
  std::vector<std::pair<int, int>> disk;
  for (int di = -r; di <= r; ++di)
    for (int dj = -r; dj <= r; ++dj)
      if (std::hypot(static_cast<double>(di), static_cast<double>(dj)) * cell <= ring) disk.emplace_back(di, dj);
  const int fsize = static_cast<int>(disk.size());
  int rank = percentile >= 100.0 ? fsize - 1 : static_cast<int>(static_cast<double>(fsize) * percentile / 100.0);
  const double big = 1e6;                          // empty cells rank last
  std::vector<double> vals(fsize);
  for (size_t k = 0; k < boxes.size(); ++k) {
    const long bi = static_cast<long>(std::floor((boxes[k][0] - lo[0]) / cell));
    const long bj = static_cast<long>(std::floor((boxes[k][1] - lo[1]) / cell));
    if (bi < 0 || bi >= sh0 || bj < 0 || bj >= sh1) continue;
    int count = 0;
    for (int f = 0; f < fsize; ++f) {
      const long i = bi + disk[f].first, j = bj + disk[f].second;
      double v = big;
      if (i >= 0 && i < sh0 && j >= 0 && j < sh1) {
        const double g = grid[static_cast<size_t>(i) * sh1 + j];
        if (std::isfinite(g)) {
          v = g;
          ++count;
        }
      }
      vals[f] = v;
    }
    std::nth_element(vals.begin(), vals.begin() + rank, vals.end());
    const double g = vals[rank];
    // the percentile must land on an occupied cell: at least percentile % of the disk occupied
    if (count >= min_cells && g < big / 2) out[k] = g;
  }
  return out;
}

std::vector<char> cross_class_suppress(const Detections& d, double margin) {
  const size_t n = d.size();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -d.scores[a] < -d.scores[b]; });
  std::vector<char> keep(n, 1);
  for (size_t a = 0; a < n; ++a) {
    const int i = order[a];
    if (!keep[i]) continue;
    const Box& bi = d.boxes[i];
    const double c = std::cos(bi[6]), s = std::sin(bi[6]);
    for (size_t b = a + 1; b < n; ++b) {
      const int j = order[b];
      if (!keep[j] || d.labels[j] == d.labels[i]) continue;
      const double dx = d.boxes[j][0] - bi[0], dy = d.boxes[j][1] - bi[1];
      const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
      if (std::abs(lx) <= bi[3] / 2 + margin && std::abs(ly) <= bi[4] / 2 + margin) keep[j] = 0;
    }
  }
  return keep;
}

namespace {

std::vector<float> current_sweep(const float* pts, size_t n, int stride) {
  const float lag_max = 1e-6f;                    // lag <= 1e-6 (float32 compare)
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) m += pts[i * stride + stride - 1] <= lag_max;
  std::vector<float> cur(3 * m);
  for (size_t i = 0, j = 0; i < n; ++i)
    if (pts[i * stride + stride - 1] <= lag_max) {
      for (int c = 0; c < 3; ++c) cur[3 * j + c] = pts[i * stride + c];
      ++j;
    }
  return cur;
}

std::vector<double> bottom_above_ground(const Detections& d, const std::vector<float>& cur, const DetectionFilterConfig& cfg) {
  std::vector<double> ground(d.size(), 0.0);
  if (cfg.local_ground && !cur.empty()) {
    const auto g = local_ground_heights(cur, d.boxes, cfg.ground_cell, cfg.ground_ring, cfg.ground_percentile,
                                        cfg.ground_min_cells);
    for (size_t k = 0; k < g.size(); ++k) ground[k] = std::isfinite(g[k]) ? g[k] : 0.0;
  }
  std::vector<double> bottom(d.size());
  for (size_t k = 0; k < d.size(); ++k) bottom[k] = d.boxes[k][2] - d.boxes[k][5] / 2 - ground[k];
  return bottom;
}

}  // namespace

void filter_measurements(const Detections& d, const float* pts, size_t n, int stride, const DetectionFilterConfig& cfg,
                         std::vector<double>& bottom, std::vector<int64_t>& counts) {
  const auto cur = current_sweep(pts, n, stride);
  bottom = bottom_above_ground(d, cur, cfg);
  counts = points_in_box_counts(cur, d.boxes, cfg.box_margin);
}

std::vector<std::string> filter_reasons(const Detections& d, const float* pts, size_t n, int stride,
                                        const DetectionFilterConfig& cfg) {
  std::vector<std::string> why(d.size());
  if (d.empty()) return why;
  const auto cur = current_sweep(pts, n, stride);
  const auto bottom = bottom_above_ground(d, cur, cfg);
  for (size_t k = 0; k < d.size(); ++k) {
    if (!(bottom[k] <= cfg.max_float)) why[k] = "floating";
    else if (!(bottom[k] >= -cfg.max_sink)) why[k] = "sunk";
  }
  const bool have_min_points = cfg.min_points[0] != 0 || cfg.min_points[1] != 0 || cfg.min_points[2] != 0;
  if (have_min_points && pts != nullptr) {        // an empty current sweep supports nothing
    std::vector<int> cand;
    std::vector<Box> boxes;
    for (size_t k = 0; k < d.size(); ++k)
      if (why[k].empty() && cfg.min_points[d.labels[k]] > 0) {
        cand.push_back(static_cast<int>(k));
        boxes.push_back(d.boxes[k]);
      }
    const auto cnt = points_in_box_counts(cur, boxes, cfg.box_margin);
    for (size_t c = 0; c < cand.size(); ++c)
      if (static_cast<double>(cnt[c]) < cfg.min_points[d.labels[cand[c]]]) why[cand[c]] = "few points";
  }
  std::vector<int> kept;
  for (size_t k = 0; k < d.size(); ++k)
    if (why[k].empty()) kept.push_back(static_cast<int>(k));
  if (cfg.cross_class_nms && !kept.empty()) {
    const auto keep = cross_class_suppress(d.select(kept), cfg.cross_class_margin);
    for (size_t c = 0; c < kept.size(); ++c)
      if (!keep[c]) why[kept[c]] = "duplicate";
  }
  return why;
}

Detections filter_detections(const Detections& d, const float* pts, size_t n, int stride, const DetectionFilterConfig& cfg) {
  if (d.empty()) return d;
  const auto why = filter_reasons(d, pts, n, stride, cfg);
  std::vector<int> idx;
  for (size_t k = 0; k < why.size(); ++k)
    if (why[k].empty()) idx.push_back(static_cast<int>(k));
  return d.select(idx);
}

}  // namespace lidar3d
