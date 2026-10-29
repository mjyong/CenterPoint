#include "dogperc/pipeline.hpp"

#include <chrono>
#include <cmath>

namespace dogperc {

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
}  // namespace

PerceptionPipeline::PerceptionPipeline(PipelineConfig cfg, Detector* detector)
    : cfg_(std::move(cfg)), poses_(cfg_.pose_buffer), imu_(poses_, cfg_.imu), pre_(cfg_.preprocess, poses_),
      detector_(detector), tracker_(cfg_.tracker), predictor_(cfg_.predictor) {}

std::optional<PipelineOutput> PerceptionPipeline::on_scan(const LidarScan& scan, const Detections* detections) {
  const auto t0 = Clock::now();
  auto frame = pre_.process(scan);
  if (!frame) return std::nullopt;
  const auto t1 = Clock::now();
  PipelineOutput out;
  out.stamp = frame->stamp;
  if (detections) {
    out.detections = *detections;
  } else {
    if (!detector_) throw Error("PerceptionPipeline: no detector and no detections given");
    out.detections = detector_->detect(frame->points, frame->stamp);
  }
  out.detections.stamp = frame->stamp;
  const auto t2 = Clock::now();
  out.detections_world = out.detections.transform(frame->T_world_det, "world");
  out.tracks = tracker_.step(out.detections_world, frame->stamp);
  const auto t3 = Clock::now();
  if (cfg_.predictor.enabled) out.predictions = predictor_(tracker_);
  const auto t4 = Clock::now();
  out.timings = frame->timings;
  out.timings["preprocess_ms"] = ms(t0, t1);
  out.timings["detect_ms"] = ms(t1, t2);
  out.timings["track_ms"] = ms(t2, t3);
  out.timings["predict_ms"] = ms(t3, t4);
  out.timings["total_ms"] = ms(t0, t4);
  if (detector_ && !detections)
    for (const auto& [k, v] : detector_->last_timing()) out.timings["det_" + k] = v;
  out.frame = std::move(*frame);
  return out;
}

void PerceptionPipeline::to_local(const PipelineOutput& out, std::vector<LocalTrack>& tracks,
                                  std::vector<Prediction>& preds) const {
  const Mat4 T = inv_T(out.frame.T_world_det);
  const Mat3 R = T.topLeftCorner<3, 3>();
  const double dyaw = std::atan2(R(1, 0), R(0, 0));
  tracks.clear();
  for (const auto& s : out.tracks)
    tracks.push_back({s.track_id, s.label, R * s.position + T.topRightCorner<3, 1>(), R.topLeftCorner<2, 2>() * s.velocity,
                      s.yaw + dyaw, s.size, s.score, s.coasting, s.motion});
  preds.clear();
  for (const auto& p : out.predictions) preds.push_back(p.transform(T, "det"));
}

}  // namespace dogperc
