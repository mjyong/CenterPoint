// The four stages wired together (pipeline.py): pose inputs at IMU / odometry rate, a frame per lidar scan.
//
//   PerceptionPipeline pipe(load_config("kygi.yaml"), &detector);
//   pipe.on_imu(t, gyro); pipe.on_odometry(t, R, p, v);   // high-rate callbacks
//   auto out = pipe.on_scan(scan);                        // 10 Hz; tracks / predictions in the world frame
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "lidar3d/config.hpp"
#include "lidar3d/detection/detector.hpp"
#include "lidar3d/pose_buffer.hpp"
#include "lidar3d/prediction/imm_predictor.hpp"
#include "lidar3d/preprocess.hpp"
#include "lidar3d/tracking/tracker.hpp"

namespace lidar3d {

struct PipelineOutput {
  double stamp = 0.0;
  PreprocessedFrame frame;
  Detections detections;          // det frame
  Detections detections_world;
  std::vector<TrackState> tracks;          // world frame
  std::vector<Prediction> predictions;     // world frame
  std::map<std::string, double> timings;
};

// planner-facing track in the det frame of the scan (gravity aligned, yaw with the body, origin on the ground)
struct LocalTrack {
  int id;
  int label;
  Vec3 position;
  Vec2 velocity;
  double yaw;
  Vec3 size;
  double score;
  bool coasting;
  Motion motion;
  Eigen::Matrix4d covariance;     // [px, py, vx, vy], rotated into the frame
  double age;                     // [s] since the track started
};

class PerceptionPipeline {
 public:
  explicit PerceptionPipeline(PipelineConfig cfg, Detector* detector = nullptr);
  void on_imu(double t, const Vec3& gyro) { imu_.on_imu(t, gyro); }
  void on_odometry(double t, const Mat3& R_wb, const Vec3& p_wb, const std::optional<Vec3>& v_w = std::nullopt) {
    imu_.on_odometry(t, R_wb, p_wb, v_w);
  }
  // none while no pose covers the scan; `detections` (det frame) bypass the detector
  std::optional<PipelineOutput> on_scan(const LidarScan& scan, const Detections* detections = nullptr);
  void to_local(const PipelineOutput& out, std::vector<LocalTrack>& tracks, std::vector<Prediction>& preds) const;
  // back to a newly constructed pipeline, same configuration and detector (OnlineInputs: the stamps jumped back)
  void reset();

  const PoseBuffer& poses() const { return poses_; }
  const Preprocessor& preprocessor() const { return pre_; }
  const MultiObjectTracker& tracker() const { return tracker_; }
  const PipelineConfig& config() const { return cfg_; }

 private:
  PipelineConfig cfg_;
  PoseBuffer poses_;
  ImuPropagator imu_;
  Preprocessor pre_;
  Detector* detector_;
  MultiObjectTracker tracker_;
  IMMPredictor predictor_;
};

}  // namespace lidar3d
