#include "dogperc/preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "dogperc/parallel.hpp"
#include "flat_map.hpp"

namespace dogperc {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// np.round / np.rint (half to even) inline: std::nearbyint is a libm call on x86-64 (one instruction on
// aarch64), three per point in dedup_returns. Adding and subtracting 1.5 * 2^52 rounds in the default
// round-to-nearest-even mode, exactly for |x| < 2^51.
inline double round_half_even(double x) {
  constexpr double k = 6755399441055744.0;
  return std::abs(x) < 2251799813685248.0 ? (x + k) - k : std::nearbyint(x);
}

// transform_points(T, xyz) for one point: xyz @ T[:3, :3].T + T[:3, 3] in float64
inline void apply(const Mat4& T, double x, double y, double z, double* out) {
  for (int r = 0; r < 3; ++r) out[r] = ((x * T(r, 0) + y * T(r, 1)) + z * T(r, 2)) + T(r, 3);
}

}  // namespace

LidarScan LidarScan::shifted(double dt) const {
  if (!dt) return *this;
  LidarScan s = *this;
  s.stamp += dt;
  for (double& t : s.point_times) t += dt;
  return s;
}

LidarScan LidarScan::make(std::vector<float> xyz, std::vector<float> intensity, std::vector<double> times,
                          std::optional<double> stamp) {
  LidarScan s;
  s.xyz = std::move(xyz);
  s.intensity = std::move(intensity);
  s.point_times = std::move(times);
  if (stamp) {
    s.stamp = *stamp;
  } else {
    if (s.point_times.empty()) throw Error("LidarScan: need either stamp or point_times");
    s.stamp = *std::max_element(s.point_times.begin(), s.point_times.end());
  }
  return s;
}

std::vector<size_t> dedup_indices(const LidarScan& scan, double resolution) {
  const size_t n = scan.size();
  const float res = static_cast<float>(resolution);   // float32 points / Python float: float32 arithmetic
  const int64_t lim = (int64_t(1) << 20) - 1;
  thread_local FlatSet seen;
  seen.reset(n);
  std::vector<size_t> keep;
  keep.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const float* p = &scan.xyz[3 * i];
    if (!(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))) continue;
    int64_t q[3];
    for (int k = 0; k < 3; ++k) {
      const double r = round_half_even(static_cast<double>(p[k] / res));   // np.round: half to even
      q[k] = std::clamp(static_cast<int64_t>(r), -lim, lim) + lim;
    }
    const int64_t key = (q[0] << 42) | (q[1] << 21) | q[2];   // 3 x 21 bits, >= 0
    if (seen.insert(key)) keep.push_back(i);
  }
  return keep;
}

LidarScan dedup_returns(const LidarScan& scan, double resolution) {
  const std::vector<size_t> keep = dedup_indices(scan, resolution);
  if (keep.size() == scan.size()) return scan;
  LidarScan out;
  out.stamp = scan.stamp;
  const bool timed = !scan.point_times.empty();
  out.xyz.resize(3 * keep.size());
  out.intensity.resize(keep.size());
  if (timed) out.point_times.resize(keep.size());
  for (size_t k = 0; k < keep.size(); ++k) {
    const size_t i = keep[k];
    for (int c = 0; c < 3; ++c) out.xyz[3 * k + c] = scan.xyz[3 * i + c];
    out.intensity[k] = scan.intensity[i];
    if (timed) out.point_times[k] = scan.point_times[i];
  }
  return out;
}

std::vector<int64_t> neighbor_counts(const float* xyz, size_t n, double radius) {
  std::vector<int64_t> out(n, 0);
  if (n == 0) return out;
  const float r = static_cast<float>(radius);
  std::vector<std::array<int64_t, 3>> keys(n);
  std::array<int64_t, 3> lo{INT64_MAX, INT64_MAX, INT64_MAX};
  for (size_t i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) {
      keys[i][k] = static_cast<int64_t>(std::floor(xyz[3 * i + k] / r));
      lo[k] = std::min(lo[k], keys[i][k]);
    }
  std::array<int64_t, 3> dims{0, 0, 0};
  for (size_t i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) {
      keys[i][k] = keys[i][k] - lo[k] + 1;
      dims[k] = std::max(dims[k], keys[i][k] + 2);
    }
  const int64_t s1 = dims[1] * dims[2], s2 = dims[2];
  // voxel code -> index among the occupied voxels (codes >= 0: every key is >= 1)
  FlatMap index(n);
  std::vector<int64_t> codes, counts, of(n);
  bool inserted = false;
  for (size_t i = 0; i < n; ++i) {
    const int64_t c = keys[i][0] * s1 + keys[i][1] * s2 + keys[i][2];
    int64_t& v = index.at(c, inserted);
    if (inserted) {
      v = static_cast<int64_t>(codes.size());
      codes.push_back(c);
      counts.push_back(0);
    }
    ++counts[v];
    of[i] = v;
  }
  std::vector<int64_t> total(codes.size());
  for (size_t v = 0; v < codes.size(); ++v) {
    int64_t sum = 0;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
          if (const int64_t* w = index.find(codes[v] + dx * s1 + dy * s2 + dz)) sum += counts[*w];
    total[v] = sum;
  }
  for (size_t i = 0; i < n; ++i) out[i] = total[of[i]] - 1;
  return out;
}

std::vector<char> SelfFilter::operator()(const float* xyz, size_t n,
                                         const std::vector<std::array<double, 7>>* dyn) const {
  const SelfFilterConfig& c = cfg_;
  std::vector<char> keep(n);
  std::vector<float> rng(n);
  const float min_r = static_cast<float>(c.min_range), max_r = static_cast<float>(c.max_range);
  parallel_for(n, [&](size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i) {
      const float x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
      rng[i] = std::sqrt((x * x + y * y) + z * z);   // np.linalg.norm of float32 rows
      keep[i] = rng[i] >= min_r && rng[i] <= max_r;
      double b[3];
      apply(T_bl_, x, y, z, b);
      for (const auto& box : c.body_boxes)
        if (b[0] >= box[0] && b[0] <= box[1] && b[1] >= box[2] && b[1] <= box[3] && b[2] >= box[4] && b[2] <= box[5])
          keep[i] = 0;
      if (dyn)
        for (const auto& o : *dyn) {
          const double cy = std::cos(o[6]), sy = std::sin(o[6]);
          const double rx = b[0] - o[0], ry = b[1] - o[1], rz = b[2] - o[2];
          const double lx = cy * rx + sy * ry, ly = -sy * rx + cy * ry;
          if (std::abs(lx) <= o[3] / 2 && std::abs(ly) <= o[4] / 2 && std::abs(rz) <= o[5] / 2) keep[i] = 0;
        }
    }
  });
  if (c.outlier_min_neighbors > 0 && c.outlier_max_range > 0) {
    const float near_r = static_cast<float>(c.outlier_max_range);
    std::vector<size_t> idx;
    std::vector<float> pts;
    for (size_t i = 0; i < n; ++i)
      if (keep[i] && rng[i] < near_r) {
        idx.push_back(i);
        pts.insert(pts.end(), xyz + 3 * i, xyz + 3 * i + 3);
      }
    if (!idx.empty()) {
      const auto cnt = neighbor_counts(pts.data(), idx.size(), c.outlier_radius);
      for (size_t k = 0; k < idx.size(); ++k)
        if (cnt[k] < c.outlier_min_neighbors) keep[idx[k]] = 0;
    }
  }
  return keep;
}

std::vector<double> deskew_points(const float* xyz, const double* times, size_t n, const PoseBuffer& poses,
                                  double t_ref, const Mat4& T_bl, double res) {
  // points share one pose per time bin (0.1 ms: the XT32 fires a block every ~50 us)
  std::vector<int64_t> bins(n);
  std::vector<double> query;
  std::vector<int64_t> inv(n);
  if (res > 0 && n > 0) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (size_t i = 0; i < n; ++i) {
      bins[i] = static_cast<int64_t>(round_half_even((times[i] - t_ref) / res));
      lo = std::min(lo, bins[i]);
      hi = std::max(hi, bins[i]);
    }
    // np.unique(bins, return_inverse=True): sorted distinct bins and each point's rank among them
    if (static_cast<uint64_t>(hi - lo) < 4 * n + 4096) {   // a sweep spans ~1000 bins: rank them densely
      std::vector<int64_t> rank(static_cast<size_t>(hi - lo) + 1, -1);
      for (size_t i = 0; i < n; ++i) rank[bins[i] - lo] = 0;
      for (size_t b = 0; b < rank.size(); ++b)
        if (rank[b] == 0) {
          rank[b] = static_cast<int64_t>(query.size());
          query.push_back(t_ref + static_cast<double>(lo + static_cast<int64_t>(b)) * res);
        }
      for (size_t i = 0; i < n; ++i) inv[i] = rank[bins[i] - lo];
    } else {
      std::vector<int64_t> u(bins);
      std::sort(u.begin(), u.end());
      u.erase(std::unique(u.begin(), u.end()), u.end());
      for (int64_t b : u) query.push_back(t_ref + static_cast<double>(b) * res);
      for (size_t i = 0; i < n; ++i) inv[i] = std::lower_bound(u.begin(), u.end(), bins[i]) - u.begin();
    }
  } else if (n > 0) {
    query.assign(times, times + n);
    std::iota(inv.begin(), inv.end(), 0);
  }
  std::vector<Mat3> R;
  std::vector<Vec3> p;
  poses.interpolate(query, R, p);
  const Mat4 Ti = inv_T(poses.pose_at(t_ref));
  const Mat3 Rr = Ti.topLeftCorner<3, 3>();
  const Vec3 tr = Ti.topRightCorner<3, 1>();
  const Mat3 Rbl = T_bl.topLeftCorner<3, 3>();
  const Vec3 tbl = T_bl.topRightCorner<3, 1>();
  std::vector<Mat3> R_full(query.size());
  std::vector<Vec3> t_full(query.size());
  for (size_t q = 0; q < query.size(); ++q) {
    const Mat3 R_rel = Rr * R[q];
    const Vec3 t_rel = Rr * p[q] + tr;
    R_full[q] = R_rel * Rbl;
    t_full[q] = R_rel * tbl + t_rel;
  }
  std::vector<double> out(3 * n);
  parallel_for(n, [&](size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i) {
      const Mat3& Rf = R_full[inv[i]];
      const Vec3& tf = t_full[inv[i]];
      const double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
      for (int r = 0; r < 3; ++r) out[3 * i + r] = ((Rf(r, 0) * x + Rf(r, 1) * y) + Rf(r, 2) * z) + tf[r];
    }
  });
  return out;
}

void SweepAccumulator::push(double stamp, std::vector<double> xyz_world, std::vector<float> intensity) {
  sweeps_.push_back({stamp, std::move(xyz_world), std::move(intensity)});
  while (static_cast<int>(sweeps_.size()) > num_sweeps_) sweeps_.pop_front();
}

std::vector<float> SweepAccumulator::build(const Mat4& T_world_det, double stamp) const {
  const Mat4 T = inv_T(T_world_det);
  // newest first: when a pillar overflows max_points the voxelizer keeps the earliest points, the freshest
  struct Part {
    const Sweep* sweep;
    size_t begin;                                  // first output row
    float lag;
  };
  std::vector<Part> parts;
  size_t total = 0;
  for (auto it = sweeps_.rbegin(); it != sweeps_.rend(); ++it) {
    const double dt = stamp - it->stamp;
    if (dt < -1e-6 || dt > max_time_span_) continue;
    parts.push_back({&*it, total, static_cast<float>(std::max(dt, 0.0))});
    total += it->intensity.size();
  }
  std::vector<float> out(total * 5);
  parallel_for(total, [&](size_t r0, size_t r1) {
    for (const Part& p : parts) {
      const size_t n = p.sweep->intensity.size();
      const size_t a = std::max(r0, p.begin), b = std::min(r1, p.begin + n);
      for (size_t r = a; r < b; ++r) {
        const size_t i = r - p.begin;
        const double* x = &p.sweep->xyz[3 * i];
        float* o = &out[5 * r];
        double d[3];
        apply(T, x[0], x[1], x[2], d);
        o[0] = static_cast<float>(d[0]);
        o[1] = static_cast<float>(d[1]);
        o[2] = static_cast<float>(d[2]);
        o[3] = p.sweep->intensity[i];
        o[4] = p.lag;
      }
    }
  });
  return out;
}

Mat4 det_frame_from_body(const Mat4& T_wb, double base_height, double ground_offset, std::array<double, 2> tilt) {
  const Mat3 R = T_wb.topLeftCorner<3, 3>();
  Vec3 origin = T_wb.topRightCorner<3, 1>();
  origin[2] -= base_height - ground_offset;
  Mat3 R_d = rot_z(yaw_of(R));
  if (tilt[0] || tilt[1]) R_d = R_d * rot_zyx(0.0, tilt[1], tilt[0]);
  return make_T(R_d, origin);
}

std::optional<std::array<double, 2>> fit_ground_slope(const std::vector<float>& xyz, double trim, int min_points) {
  const size_t n = xyz.size() / 3;
  std::vector<char> keep(n, 1);
  Eigen::Vector3d coef = Eigen::Vector3d::Zero();
  for (int pass = 0; pass < 2; ++pass) {
    size_t m = 0;
    for (char k : keep) m += k;
    if (static_cast<int>(m) < min_points) return std::nullopt;
    Eigen::MatrixXd A(m, 3);
    Eigen::VectorXd z(m);
    size_t r = 0;
    for (size_t i = 0; i < n; ++i)
      if (keep[i]) {
        A(r, 0) = xyz[3 * i];
        A(r, 1) = xyz[3 * i + 1];
        A(r, 2) = 1.0;
        z[r++] = xyz[3 * i + 2];
      }
    coef = A.colPivHouseholderQr().solve(z);
    for (size_t i = 0; i < n; ++i) {
      const double pred = (static_cast<double>(xyz[3 * i]) * coef[0] + static_cast<double>(xyz[3 * i + 1]) * coef[1]) + coef[2];
      keep[i] = std::abs(static_cast<double>(xyz[3 * i + 2]) - pred) < trim;
    }
  }
  return std::array<double, 2>{coef[0], coef[1]};
}

double GroundHeightEstimator::update(const float* pts, size_t n, int stride) {
  const DetFrameConfig& c = cfg_;
  // float32 comparisons, as NumPy compares float32 arrays with Python floats
  const float r0 = static_cast<float>(c.ground_ring[0]), r1 = static_cast<float>(c.ground_ring[1]);
  const float band = static_cast<float>(c.ground_band);
  std::vector<float> sel;
  for (size_t i = 0; i < n; ++i) {
    const float* p = pts + i * stride;
    const float r = std::hypot(p[0], p[1]);
    if (r > r0 && r < r1 && std::abs(p[2]) < band) sel.insert(sel.end(), p, p + 3);
  }
  const size_t m = sel.size() / 3;
  if (static_cast<int>(m) < c.ground_min_points) return offset;
  if (c.auto_ground) {
    std::vector<float> z(m);
    for (size_t i = 0; i < m; ++i) z[i] = sel[3 * i + 2];
    float med;
    const size_t h = m / 2;
    std::nth_element(z.begin(), z.begin() + h, z.end());
    if (m % 2) {
      med = z[h];
    } else {
      const float hi = z[h];
      const float lo = *std::max_element(z.begin(), z.begin() + h);
      med = (lo + hi) / 2.0f;                     // np.median of float32: the mean of the middle two in float32
    }
    offset = std::clamp(offset + c.ground_alpha * static_cast<double>(med), -c.ground_max_correction, c.ground_max_correction);
  }
  if (c.auto_level) {
    const auto slope = fit_ground_slope(sel);
    if (slope) {
      // plane z = a x + b y + c -> roll = atan(b), pitch = -atan(a)
      const double res[2] = {std::atan((*slope)[1]), -std::atan((*slope)[0])};
      for (int k = 0; k < 2; ++k)
        tilt[k] = std::clamp(tilt[k] + c.ground_alpha * res[k], -c.ground_max_tilt, c.ground_max_tilt);
    }
  }
  return offset;
}

Preprocessor::Preprocessor(PreprocessConfig cfg, const PoseBuffer& poses)
    : cfg_(std::move(cfg)), poses_(poses), self_filter_(cfg_.self_filter, cfg_.T_body_lidar),
      accumulator_(cfg_.num_sweeps, cfg_.max_time_span) {
  if (cfg_.det_frame.auto_ground || cfg_.det_frame.auto_level) ground_.emplace(cfg_.det_frame);
}

void Preprocessor::reset() {
  accumulator_.reset();
  if (ground_) ground_.emplace(cfg_.det_frame);
}

void Preprocessor::sweep_to_body(const LidarScan& scan, std::vector<double>& xyz_b, std::vector<float>& inten,
                                 const std::vector<std::array<double, 7>>* dyn,
                                 std::map<std::string, double>* timings) const {
  auto t = std::chrono::steady_clock::now();
  auto lap = [&](const char* key) {
    if (timings) (*timings)[key] = ms_since(t);
    t = std::chrono::steady_clock::now();
  };
  // indices into the raw sweep: dedup (drops non-finite points too), else the finite points
  std::vector<size_t> fin;
  if (cfg_.dedup_returns) {
    fin = dedup_indices(scan);
  } else {
    fin.reserve(scan.size());
    for (size_t i = 0; i < scan.size(); ++i) {
      const float* p = &scan.xyz[3 * i];
      if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) fin.push_back(i);
    }
  }
  lap("dedup_ms");
  std::vector<float> fin_xyz(3 * fin.size());
  for (size_t k = 0; k < fin.size(); ++k)
    for (int c = 0; c < 3; ++c) fin_xyz[3 * k + c] = scan.xyz[3 * fin[k] + c];
  const auto keep = self_filter_(fin_xyz.data(), fin.size(), dyn);
  size_t m = 0;
  for (char k : keep) m += k != 0;
  const bool timed = !scan.point_times.empty();
  std::vector<float> xyz(3 * m);
  std::vector<double> times(timed ? m : 0);
  inten.resize(m);
  const float scale = static_cast<float>(cfg_.intensity_scale);
  for (size_t k = 0, j = 0; k < fin.size(); ++k) {
    if (!keep[k]) continue;
    const size_t i = fin[k];
    for (int c = 0; c < 3; ++c) xyz[3 * j + c] = fin_xyz[3 * k + c];
    inten[j] = scan.intensity[i] * scale;
    if (timed) times[j] = scan.point_times[i];
    ++j;
  }
  lap("self_filter_ms");
  if (cfg_.deskew && timed) {
    xyz_b = deskew_points(xyz.data(), times.data(), m, poses_, scan.stamp, cfg_.T_body_lidar, cfg_.time_resolution);
  } else {
    xyz_b.resize(3 * m);
    for (size_t i = 0; i < m; ++i) apply(cfg_.T_body_lidar, xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], &xyz_b[3 * i]);
  }
  lap("deskew_ms");
}

std::optional<PreprocessedFrame> Preprocessor::process(const LidarScan& scan,
                                                       const std::vector<std::array<double, 7>>* dyn) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<double> xyz_b;
  std::vector<float> inten;
  std::map<std::string, double> timings;
  Mat4 T_wb;
  try {
    sweep_to_body(scan, xyz_b, inten, dyn, &timings);
    T_wb = poses_.pose_at(scan.stamp);
  } catch (const PoseOutOfRange&) {
    return std::nullopt;
  }
  const double sweep_ms = ms_since(t0);
  const auto t1 = std::chrono::steady_clock::now();
  const size_t m = inten.size();
  std::vector<double> xyz_w(3 * m);
  parallel_for(m, [&](size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i) apply(T_wb, xyz_b[3 * i], xyz_b[3 * i + 1], xyz_b[3 * i + 2], &xyz_w[3 * i]);
  });
  accumulator_.push(scan.stamp, std::move(xyz_w), std::move(inten));

  const double offset = ground_ ? ground_->offset : 0.0;
  const std::array<double, 2> tilt = ground_ ? ground_->tilt : std::array<double, 2>{0.0, 0.0};
  PreprocessedFrame f;
  f.stamp = scan.stamp;
  f.T_world_body = T_wb;
  f.T_world_det = det_frame_from_body(T_wb, cfg_.det_frame.base_height, offset, tilt);
  f.points = accumulator_.build(f.T_world_det, scan.stamp);
  if (ground_) {
    std::vector<float> cur;      // the newest sweep: dt == 0
    for (size_t i = 0; i < f.num_points(); ++i)
      if (f.points[5 * i + 4] == 0.0f) cur.insert(cur.end(), &f.points[5 * i], &f.points[5 * i] + 3);
    ground_->update(cur.data(), cur.size() / 3, 3);
  }
  f.num_raw = scan.size();
  f.num_kept = m;
  // sweep_ms = dedup_ms + self_filter_ms + deskew_ms (T28 called the sum deskew_ms)
  f.timings = std::move(timings);
  f.timings["sweep_ms"] = sweep_ms;
  f.timings["accumulate_ms"] = ms_since(t1);
  return f;
}

}  // namespace dogperc
