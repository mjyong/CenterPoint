// Stage 1: raw lidar sweep -> detector-ready multi-sweep cloud in the det frame (preprocess/*.py).
//
// Per sweep: dedup returns -> self filter (body frame) -> deskew to the scan end (IMU-rate poses) ->
// world frame -> sweep buffer. Per frame: det frame from the body pose at the scan end (gravity aligned,
// yaw only, origin on the ground) -> the last N sweeps in it with a dt channel. Point arithmetic follows
// the Python dtypes: sweeps arrive float32, poses and world points are float64, the output is float32.
#pragma once

#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "dogperc/config.hpp"
#include "dogperc/pose_buffer.hpp"

namespace dogperc {

struct LidarScan {
  std::vector<float> xyz;            // (N, 3) lidar frame
  std::vector<float> intensity;      // (N,)
  std::vector<double> point_times;   // (N,) absolute [s]; empty: no per-point times
  double stamp = 0.0;                // scan end
  size_t size() const { return intensity.size(); }
  LidarScan shifted(double dt) const;   // every time moved by dt (lidar clock -> pose clock)
  // stamp = the last point time when it is not given
  static LidarScan make(std::vector<float> xyz, std::vector<float> intensity, std::vector<double> times,
                        std::optional<double> stamp = std::nullopt);
};

// repeated returns (same xyz on a 1 mm grid, the first kept) and non-finite points removed
LidarScan dedup_returns(const LidarScan& scan, double resolution = 1e-3);

// points within `radius`-sized voxels of the 3x3x3 block around each point, minus itself
std::vector<int64_t> neighbor_counts(const float* xyz, size_t n, double radius);

class SelfFilter {
 public:
  SelfFilter(SelfFilterConfig cfg, const Mat4& T_body_lidar) : cfg_(std::move(cfg)), T_bl_(T_body_lidar) {}
  // keep mask of (N, 3) lidar-frame points; dynamic_boxes: (M, 7) oriented boxes in the body frame
  std::vector<char> operator()(const float* xyz, size_t n, const std::vector<std::array<double, 7>>* dynamic_boxes = nullptr) const;

 private:
  SelfFilterConfig cfg_;
  Mat4 T_bl_;
};

// motion-compensate one sweep: (N, 3) body-frame points at t_ref (float64)
std::vector<double> deskew_points(const float* xyz_lidar, const double* point_times, size_t n, const PoseBuffer& poses,
                                  double t_ref, const Mat4& T_body_lidar, double time_resolution);

class SweepAccumulator {
 public:
  SweepAccumulator(int num_sweeps, double max_time_span) : num_sweeps_(num_sweeps), max_time_span_(max_time_span) {}
  void reset() { sweeps_.clear(); }
  size_t size() const { return sweeps_.size(); }
  void push(double stamp, std::vector<double> xyz_world, std::vector<float> intensity);
  // (N, 5) float32: xyz in the det frame, intensity, dt = stamp - t_sweep; newest sweep first
  std::vector<float> build(const Mat4& T_world_det, double stamp) const;

 private:
  struct Sweep {
    double stamp;
    std::vector<double> xyz;
    std::vector<float> intensity;
  };
  int num_sweeps_;
  double max_time_span_;
  std::deque<Sweep> sweeps_;
};

Mat4 det_frame_from_body(const Mat4& T_world_body, double base_height, double ground_offset = 0.0,
                         std::array<double, 2> tilt = {0.0, 0.0});

// trimmed least-squares ground plane z = a x + b y + c over (M, 3) float32 points; (a, b) or none
std::optional<std::array<double, 2>> fit_ground_slope(const std::vector<float>& xyz, double trim = 0.08,
                                                      int min_points = 100);

class GroundHeightEstimator {
 public:
  explicit GroundHeightEstimator(DetFrameConfig cfg) : cfg_(cfg) {}
  // newest-sweep det-frame points, stride `stride` floats
  double update(const float* pts, size_t n, int stride);
  double offset = 0.0;
  std::array<double, 2> tilt{0.0, 0.0};   // (roll, pitch)

 private:
  DetFrameConfig cfg_;
};

struct PreprocessedFrame {
  double stamp = 0.0;
  std::vector<float> points;          // (N, 5) [x, y, z, intensity, dt] det frame
  Mat4 T_world_det = Mat4::Identity();
  Mat4 T_world_body = Mat4::Identity();
  size_t num_raw = 0, num_kept = 0;
  std::map<std::string, double> timings;
  size_t num_points() const { return points.size() / 5; }
};

class Preprocessor {
 public:
  Preprocessor(PreprocessConfig cfg, const PoseBuffer& poses);
  void reset() { accumulator_.reset(); }
  // self filter + deskew one sweep: body-frame points at the scan stamp (float64) and their intensity
  void sweep_to_body(const LidarScan& scan, std::vector<double>& xyz_body, std::vector<float>& intensity,
                     const std::vector<std::array<double, 7>>* dynamic_boxes = nullptr) const;
  // none while no pose covers the scan
  std::optional<PreprocessedFrame> process(const LidarScan& scan,
                                           const std::vector<std::array<double, 7>>* dynamic_boxes = nullptr);
  const PreprocessConfig& config() const { return cfg_; }
  const std::optional<GroundHeightEstimator>& ground() const { return ground_; }

 private:
  PreprocessConfig cfg_;
  const PoseBuffer& poses_;
  SelfFilter self_filter_;
  SweepAccumulator accumulator_;
  std::optional<GroundHeightEstimator> ground_;
};

}  // namespace dogperc
