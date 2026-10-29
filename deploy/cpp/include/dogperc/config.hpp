// Parameters of every stage, one struct per Python dataclass with the same field names and defaults.
//
// The parameters have one source, the Python dataclasses: tools/deploy/export_cpp_config.py writes them to
// YAML (a robot preset, or a recorded sequence's meta.json) and load_config reads that file, every key
// required. The defaults below only mirror Python's; tests/test_io_geometry.cpp checks they still agree.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "dogperc/common.hpp"
#include "dogperc/geometry.hpp"

namespace dogperc {

struct SelfFilterConfig {                       // preprocess/self_filter.py
  std::vector<std::array<double, 6>> body_boxes{{-0.60, 0.60, -0.40, 0.40, -0.80, 0.30}};
  double min_range = 0.4;
  double max_range = 120.0;
  double outlier_radius = 0.2;
  int outlier_min_neighbors = 2;
  double outlier_max_range = 3.0;
};

struct DetFrameConfig {                         // preprocess/frames.py
  double base_height = 0.45;
  bool auto_ground = false;
  bool auto_level = false;
  std::array<double, 2> ground_ring{1.5, 8.0};
  double ground_band = 0.35;
  int ground_min_points = 200;
  double ground_alpha = 0.2;
  double ground_max_correction = 0.3;
  double ground_max_tilt = 5.0 * kPi / 180.0;
};

struct PreprocessConfig {                       // preprocess/pipeline.py
  Mat4 T_body_lidar = make_T(Mat3::Identity(), Vec3(0.2, 0.0, 0.15));
  int num_sweeps = 5;
  double max_time_span = 0.55;
  bool deskew = true;
  double time_resolution = 1e-4;
  bool dedup_returns = true;
  double intensity_scale = 1.0;
  SelfFilterConfig self_filter;
  DetFrameConfig det_frame;
};

struct PoseBufferConfig {                       // pose_buffer.PoseBuffer
  double max_age = 3.0;
  double max_extrapolation = 0.05;
  double extrapolation_window = 0.02;
};

struct ImuConfig {                              // pose_buffer.ImuPropagator
  Vec3 gyro_bias = Vec3::Zero();
  double imu_history = 1.0;
};

struct DetectionFilterConfig {                  // detection/filters.py
  bool enabled = true;                          // false: the detector reports its raw boxes (filter_cfg=None)
  PerClass min_points{5, 3, 3};
  double box_margin = 0.1;
  double max_float = 0.7;
  double max_sink = 1.2;
  bool local_ground = true;
  double ground_cell = 2.0;
  double ground_ring = 4.0;
  double ground_percentile = 10.0;
  int ground_min_cells = 5;
  bool cross_class_nms = true;
  double cross_class_margin = 0.2;
};

struct MotionModelSpec {                        // tracking/imm.py: ("cv", q_acc[, q_omega]) / ("ct", ...)
  std::string kind;
  double q_acc = 1.0;
  double q_omega = 0.0;
};

struct ClassParams {                            // tracking/tracker.py
  double gate = 0.0;
  double max_age = 0.0;
  int min_hits = 1;
  double pos_std = 0.0;
  double vel_std = 0.0;
  std::vector<MotionModelSpec> models;
  double stay_prob = 0.95;
  double init_omega_std = 0.3;
};

struct TrackerConfig {
  std::array<ClassParams, kNumClasses> class_params{
      ClassParams{3.0, 1.5, 3, 0.30, 0.8, {{"cv", 1.5, 1e-3}, {"ct", 1.5, 0.25}}},
      ClassParams{1.5, 1.5, 2, 0.15, 0.5, {{"cv", 0.4, 1e-3}, {"cv", 2.0, 1e-3}}},
      ClassParams{2.5, 1.0, 2, 0.20, 0.6, {{"cv", 1.5, 1e-3}, {"ct", 1.5, 0.4}}}};
  std::vector<int> classes{0};                  // tracked (and predicted) classes
  PerClass high_score{0.4, 0.3, 0.4};
  double low_score = 0.2;
  std::string matching = "greedy";             // or "hungarian"
  bool use_velocity = true;
  double tentative_max_age = 0.15;
  bool output_coasting = true;
  PerClass output_min_score{0.35, 0.25, 0.3};
  double history = 3.0;
  double size_alpha = 0.2;
  double z_alpha = 0.3;
  double yaw_alpha = 0.4;
  double heading_from_velocity = 1.0;
  PerClass motion_on_speed{0.7, 0.5, 0.7};
  PerClass motion_off_speed{0.3, 0.2, 0.3};
  int motion_window = 5;
  double motion_traj_window = 1.0;
  int motion_on_frames = 3;
  double motion_off_time = 1.0;
  double stop_memory = 10.0;
};

struct PredictorConfig {                        // prediction/kinematic.IMMPredictor
  bool enabled = true;
  double horizon = 3.0;
  double step = 0.5;
  double merge_dist = 0.25;
};

struct PipelineConfig {
  PreprocessConfig preprocess;
  PoseBufferConfig pose_buffer;
  ImuConfig imu;
  DetectionFilterConfig detection_filter;
  PerClass score_thresholds{0.4, 0.3, 0.4};   // DEFAULT_SCORE_THRESHOLDS: reporting / drawing
  TrackerConfig tracker;
  PredictorConfig predictor;
};

PipelineConfig load_config(const std::string& yaml_path);
PipelineConfig parse_config(const std::string& yaml_text);

}  // namespace dogperc
