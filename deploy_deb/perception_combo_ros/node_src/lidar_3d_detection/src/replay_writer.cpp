#include "lidar3d/replay_writer.hpp"

#include <cmath>

#include "lidar3d/detection/filters.hpp"

namespace lidar3d {

void ReplayWriter::Dets::add(const Detections& d) {
  for (size_t i = 0; i < d.size(); ++i) {
    boxes.insert(boxes.end(), d.boxes[i].begin(), d.boxes[i].end());
    velocities.insert(velocities.end(), d.velocities[i].begin(), d.velocities[i].end());
    scores.push_back(d.scores[i]);
    labels.push_back(d.labels[i]);
    subclasses.push_back(d.subclasses[i]);
  }
  offsets.push_back(offsets.back() + static_cast<int64_t>(d.size()));
}

void ReplayWriter::Dets::write(NpzWriter& w, const std::string& p) const {
  const int64_t n = static_cast<int64_t>(scores.size());
  w.add(p + "_offsets", offsets);
  w.add(p + "_boxes", boxes, {n, 7});
  w.add(p + "_velocities", velocities, {n, 2});
  w.add(p + "_scores", scores);
  w.add(p + "_labels", labels);
  w.add(p + "_subclasses", subclasses);
}

void ReplayWriter::add(const PipelineOutput& o, const Preprocessor& pre, bool save_points,
                       const Detections* raw_detections, const DetectionFilterConfig& filter) {
  stamp.push_back(o.stamp);
  for (int k = 0; k < 16; ++k) {
    T_wd.push_back(o.frame.T_world_det(k / 4, k % 4));
    T_wb.push_back(o.frame.T_world_body(k / 4, k % 4));
  }
  num_raw.push_back(static_cast<int64_t>(o.frame.num_raw));
  num_kept.push_back(static_cast<int64_t>(o.frame.num_kept));
  ground_offset.push_back(pre.ground() ? pre.ground()->offset : 0.0);
  ground_tilt.push_back(pre.ground() ? pre.ground()->tilt[0] : 0.0);
  ground_tilt.push_back(pre.ground() ? pre.ground()->tilt[1] : 0.0);
  if (save_points) points.insert(points.end(), o.frame.points.begin(), o.frame.points.end());
  point_offsets.push_back(point_offsets.back() + static_cast<int64_t>(save_points ? o.frame.num_points() : 0));
  det.add(o.detections);
  if (raw_detections) {
    have_raw = true;
    const Detections& raw_d = *raw_detections;
    raw.add(raw_d);
    std::vector<double> bottom;
    std::vector<int64_t> counts;
    filter_measurements(raw_d, o.frame.points.data(), o.frame.num_points(), 5, filter, bottom, counts);
    raw_bottom.insert(raw_bottom.end(), bottom.begin(), bottom.end());
    raw_counts.insert(raw_counts.end(), counts.begin(), counts.end());
  }
  for (const auto& [k, v] : o.timings) {
    auto& col = timings[k];
    col.resize(stamp.size() - 1, kNaN);
    col.push_back(v);
  }
  for (const auto& t : o.tracks) {
    trk_id.push_back(t.track_id);
    trk_label.push_back(t.label);
    trk_position.insert(trk_position.end(), t.position.data(), t.position.data() + 3);
    trk_velocity.insert(trk_velocity.end(), t.velocity.data(), t.velocity.data() + 2);
    trk_yaw.push_back(t.yaw);
    trk_size.insert(trk_size.end(), t.size.data(), t.size.data() + 3);
    trk_score.push_back(t.score);
    for (int k = 0; k < 16; ++k) trk_cov.push_back(t.covariance(k / 4, k % 4));
    for (int k = 0; k < 2; ++k) trk_mode_probs.push_back(k < static_cast<int>(t.mode_probs.size()) ? t.mode_probs[k] : kNaN);
    trk_age.push_back(t.age);
    trk_hits.push_back(t.hits);
    trk_coasting.push_back(t.coasting);
    trk_motion.push_back(static_cast<int64_t>(t.motion));
    trk_tsm.push_back(t.time_since_moving);
    trk_sd.push_back(t.speed_displacement);
    trk_sh.push_back(t.speed_head);
    for (int k = 0; k < 6; ++k) trk_meas.push_back(t.has_measurement ? t.measurement[k] : kNaN);
    trk_history_len.push_back(static_cast<int64_t>(t.history.size()));
  }
  trk_offsets.push_back(trk_offsets.back() + static_cast<int64_t>(o.tracks.size()));
  for (const auto& p : o.predictions) {
    if (pred_times.empty()) {
      pred_times = p.times;
      T = static_cast<int64_t>(p.times.size());
    }
    pred_track_id.push_back(p.track_id);
    pred_label.push_back(p.label);
    pred_num_modes.push_back(static_cast<int64_t>(p.probs.size()));
    for (int k = 0; k < 2; ++k) {
      const bool have = k < static_cast<int>(p.probs.size());
      pred_probs.push_back(have ? p.probs[k] : kNaN);
      for (int64_t s = 0; s < T; ++s) {
        for (int c = 0; c < 2; ++c) pred_modes.push_back(have ? p.modes[k][s][c] : kNaN);
        for (int c = 0; c < 4; ++c) pred_covs.push_back(have ? p.covs[k][s](c / 2, c % 2) : kNaN);
      }
    }
  }
  pred_offsets.push_back(pred_offsets.back() + static_cast<int64_t>(o.predictions.size()));
}

void ReplayWriter::write(const std::string& path) const {
  NpzWriter w;
  const int64_t F = static_cast<int64_t>(stamp.size());
  w.add("stamp", stamp);
  w.add("T_world_det", T_wd, {F, 4, 4});
  w.add("T_world_body", T_wb, {F, 4, 4});
  w.add("num_raw", num_raw);
  w.add("num_kept", num_kept);
  w.add("ground_offset", ground_offset);
  w.add("ground_tilt", ground_tilt, {F, 2});
  w.add("point_offsets", point_offsets);
  w.add("points", points, {static_cast<int64_t>(points.size() / 5), 5});
  det.write(w, "det");
  if (have_raw) {
    raw.write(w, "raw");
    w.add("raw_bottom", raw_bottom);
    w.add("raw_counts", raw_counts);
  }
  for (const auto& [k, col] : timings) {
    std::vector<double> v = col;                // frames before a stage first reported it: NaN
    v.resize(F, kNaN);
    w.add("timing_" + k, v);
  }
  const int64_t n = static_cast<int64_t>(trk_id.size()), q = static_cast<int64_t>(pred_track_id.size());
  w.add("trk_offsets", trk_offsets);
  w.add("trk_id", trk_id);
  w.add("trk_label", trk_label);
  w.add("trk_position", trk_position, {n, 3});
  w.add("trk_velocity", trk_velocity, {n, 2});
  w.add("trk_yaw", trk_yaw);
  w.add("trk_size", trk_size, {n, 3});
  w.add("trk_score", trk_score);
  w.add("trk_covariance", trk_cov, {n, 4, 4});
  w.add("trk_mode_probs", trk_mode_probs, {n, 2});
  w.add("trk_age", trk_age);
  w.add("trk_hits", trk_hits);
  w.add("trk_coasting", trk_coasting);
  w.add("trk_motion", trk_motion);
  w.add("trk_time_since_moving", trk_tsm);
  w.add("trk_speed_displacement", trk_sd);
  w.add("trk_speed_head", trk_sh);
  w.add("trk_measurement", trk_meas, {n, 6});
  w.add("trk_history_len", trk_history_len);
  w.add("pred_offsets", pred_offsets);
  w.add("pred_times", pred_times);
  w.add("pred_track_id", pred_track_id);
  w.add("pred_label", pred_label);
  w.add("pred_num_modes", pred_num_modes);
  w.add("pred_probs", pred_probs, {q, 2});
  w.add("pred_modes", pred_modes, {q, 2, T, 2});
  w.add("pred_covs", pred_covs, {q, 2, T, 2, 2});
  w.write(path);
}

GivenDetections::GivenDetections(const std::string& path) : z_(read_npz(path)) {
  stamps_ = member(z_, "stamp").to<double>();
}

bool GivenDetections::lookup(double stamp, Detections& d) const {
  d = Detections();
  if (!(next_ < stamps_.size() && std::abs(stamps_[next_] - stamp) < 1e-9)) return false;
  const auto& off = member(z_, "offsets");
  const auto& boxes = member(z_, "boxes");
  const auto& vel = member(z_, "velocities");
  for (int64_t r = off.at<int64_t>(next_); r < off.at<int64_t>(next_ + 1); ++r) {
    Box b;
    for (int c = 0; c < 7; ++c) b[c] = boxes.at<double>(7 * r + c);
    d.push(b, {vel.at<double>(2 * r), vel.at<double>(2 * r + 1)}, member(z_, "scores").at<double>(r),
           member(z_, "labels").at<int>(r), member(z_, "subclasses").at<int>(r));
  }
  return true;
}

}  // namespace lidar3d
