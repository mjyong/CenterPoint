// lidar3d_replay's output: per-frame pipeline results in one .npz, the layout of tools/deploy/dump_golden.py's
// pipeline.npz (tools/deploy/cpp_parity.py compares and scores it). Also the detections of another run fed
// back frame by frame (dets.npz, cpp_parity.py python --dets). Shared by the sequence, message-log and ROS
// bag replays.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "lidar3d/npz.hpp"
#include "lidar3d/pipeline.hpp"

namespace lidar3d {

class ReplayWriter {
 public:
  // raw_detections: the detector's output before the filters (none without a detector)
  void add(const PipelineOutput& o, const Preprocessor& pre, bool save_points, const Detections* raw_detections,
           const DetectionFilterConfig& filter);
  void write(const std::string& path) const;
  size_t frames() const { return stamp.size(); }
  size_t track_rows() const { return trk_id.size(); }

 private:
  struct Dets {
    std::vector<int64_t> offsets{0}, labels, subclasses;
    std::vector<double> boxes, velocities, scores;
    void add(const Detections& d);
    void write(NpzWriter& w, const std::string& p) const;
  } det, raw;
  bool have_raw = false;
  std::vector<double> stamp, T_wd, T_wb, ground_offset, ground_tilt;
  std::vector<int64_t> num_raw, num_kept, point_offsets{0};
  std::vector<float> points;
  std::map<std::string, std::vector<double>> timings;
  std::vector<double> raw_bottom;
  std::vector<int64_t> raw_counts;
  std::vector<int64_t> trk_offsets{0}, trk_id, trk_label, trk_hits, trk_coasting, trk_motion, trk_history_len;
  std::vector<double> trk_position, trk_velocity, trk_yaw, trk_size, trk_score, trk_cov, trk_mode_probs, trk_age,
      trk_tsm, trk_sd, trk_sh, trk_meas;
  std::vector<int64_t> pred_offsets{0}, pred_track_id, pred_label, pred_num_modes;
  std::vector<double> pred_modes, pred_probs, pred_covs, pred_times;
  int64_t T = 0;
};

// the detections another run made (dets.npz), frame by frame in order: a scan that run made no frame for
// gets none here either
class GivenDetections {
 public:
  explicit GivenDetections(const std::string& path);
  // the next frame's detections if that frame is at `stamp` (else empty, and false)
  bool lookup(double stamp, Detections& out) const;
  void advance() { ++next_; }      // after a frame made with the looked-up detections

 private:
  Npz z_;
  std::vector<double> stamps_;
  size_t next_ = 0;
};

}  // namespace lidar3d
