#include "dogperc/config.hpp"

#include <yaml-cpp/yaml.h>

namespace dogperc {

namespace {

YAML::Node need(const YAML::Node& n, const std::string& key, const std::string& where) {
  YAML::Node v = n[key];
  if (!v) throw Error("config: missing " + where + "." + key + " (write it with tools/deploy/export_cpp_config.py)");
  return v;
}

template <class T>
void get(const YAML::Node& n, const std::string& key, T& out, const std::string& where) {
  out = need(n, key, where).as<T>();
}

void get_per_class(const YAML::Node& n, const std::string& key, PerClass& out, const std::string& where) {
  const YAML::Node v = need(n, key, where);
  out = PerClass();
  if (v.IsScalar()) {
    out = PerClass(v.as<double>());
    return;
  }
  for (auto it = v.begin(); it != v.end(); ++it) {
    const int c = class_index(it->first.as<std::string>());
    if (c < 0) throw Error("config: " + where + "." + key + ": unknown class " + it->first.as<std::string>());
    out[c] = it->second.as<double>();
  }
}

}  // namespace

namespace {

PipelineConfig from_node(const YAML::Node& root) {
  if (!root["version"] || root["version"].as<int>() != 1) throw Error("config: version 1 expected");
  PipelineConfig c;

  const YAML::Node pre = need(root, "preprocess", "");
  auto& p = c.preprocess;
  const auto T = need(pre, "T_body_lidar", "preprocess").as<std::vector<double>>();
  if (T.size() != 16) throw Error("config: preprocess.T_body_lidar needs 16 values (row major)");
  for (int i = 0; i < 16; ++i) p.T_body_lidar(i / 4, i % 4) = T[i];
  get(pre, "num_sweeps", p.num_sweeps, "preprocess");
  get(pre, "max_time_span", p.max_time_span, "preprocess");
  get(pre, "deskew", p.deskew, "preprocess");
  get(pre, "time_resolution", p.time_resolution, "preprocess");
  get(pre, "dedup_returns", p.dedup_returns, "preprocess");
  get(pre, "intensity_scale", p.intensity_scale, "preprocess");
  const YAML::Node sf = need(pre, "self_filter", "preprocess");
  p.self_filter.body_boxes.clear();
  for (const auto& b : need(sf, "body_boxes", "self_filter")) {
    const auto v = b.as<std::vector<double>>();
    if (v.size() != 6) throw Error("config: self_filter.body_boxes: 6 values per box");
    p.self_filter.body_boxes.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
  }
  get(sf, "min_range", p.self_filter.min_range, "self_filter");
  get(sf, "max_range", p.self_filter.max_range, "self_filter");
  get(sf, "outlier_radius", p.self_filter.outlier_radius, "self_filter");
  get(sf, "outlier_min_neighbors", p.self_filter.outlier_min_neighbors, "self_filter");
  get(sf, "outlier_max_range", p.self_filter.outlier_max_range, "self_filter");
  const YAML::Node df = need(pre, "det_frame", "preprocess");
  auto& d = p.det_frame;
  get(df, "base_height", d.base_height, "det_frame");
  get(df, "auto_ground", d.auto_ground, "det_frame");
  get(df, "auto_level", d.auto_level, "det_frame");
  const auto ring = need(df, "ground_ring", "det_frame").as<std::vector<double>>();
  if (ring.size() != 2) throw Error("config: det_frame.ground_ring needs 2 values");
  d.ground_ring = {ring[0], ring[1]};
  get(df, "ground_band", d.ground_band, "det_frame");
  get(df, "ground_min_points", d.ground_min_points, "det_frame");
  get(df, "ground_alpha", d.ground_alpha, "det_frame");
  get(df, "ground_max_correction", d.ground_max_correction, "det_frame");
  get(df, "ground_max_tilt", d.ground_max_tilt, "det_frame");

  const YAML::Node pb = need(root, "pose_buffer", "");
  get(pb, "max_age", c.pose_buffer.max_age, "pose_buffer");
  get(pb, "max_extrapolation", c.pose_buffer.max_extrapolation, "pose_buffer");
  get(pb, "extrapolation_window", c.pose_buffer.extrapolation_window, "pose_buffer");

  const YAML::Node imu = need(root, "imu", "");
  const auto bias = need(imu, "gyro_bias", "imu").as<std::vector<double>>();
  if (bias.size() != 3) throw Error("config: imu.gyro_bias needs 3 values");
  c.imu.gyro_bias = Vec3(bias[0], bias[1], bias[2]);
  get(imu, "imu_history", c.imu.imu_history, "imu");

  const YAML::Node fl = need(root, "detection_filter", "");
  auto& f = c.detection_filter;
  get(fl, "enabled", f.enabled, "detection_filter");
  get_per_class(fl, "min_points", f.min_points, "detection_filter");
  get(fl, "box_margin", f.box_margin, "detection_filter");
  get(fl, "max_float", f.max_float, "detection_filter");
  get(fl, "max_sink", f.max_sink, "detection_filter");
  get(fl, "local_ground", f.local_ground, "detection_filter");
  get(fl, "ground_cell", f.ground_cell, "detection_filter");
  get(fl, "ground_ring", f.ground_ring, "detection_filter");
  get(fl, "ground_percentile", f.ground_percentile, "detection_filter");
  get(fl, "ground_min_cells", f.ground_min_cells, "detection_filter");
  get(fl, "cross_class_nms", f.cross_class_nms, "detection_filter");
  get(fl, "cross_class_margin", f.cross_class_margin, "detection_filter");

  get_per_class(root, "score_thresholds", c.score_thresholds, "");

  const YAML::Node tr = need(root, "tracker", "");
  auto& t = c.tracker;
  const YAML::Node cp = need(tr, "class_params", "tracker");
  for (int k = 0; k < kNumClasses; ++k) {
    const std::string w = std::string("tracker.class_params.") + kClasses[k];
    const YAML::Node n = need(cp, kClasses[k], "tracker.class_params");
    ClassParams q;
    get(n, "gate", q.gate, w);
    get(n, "max_age", q.max_age, w);
    get(n, "min_hits", q.min_hits, w);
    get(n, "pos_std", q.pos_std, w);
    get(n, "vel_std", q.vel_std, w);
    for (const auto& m : need(n, "models", w)) {
      if (!m.IsSequence() || m.size() != 3) throw Error("config: " + w + ".models: [kind, q_acc, q_omega] per model");
      q.models.push_back({m[0].as<std::string>(), m[1].as<double>(), m[2].as<double>()});
      if (q.models.back().kind != "cv" && q.models.back().kind != "ct") throw Error("config: " + w + ": model cv or ct");
    }
    get(n, "stay_prob", q.stay_prob, w);
    get(n, "init_omega_std", q.init_omega_std, w);
    t.class_params[k] = q;
  }
  t.classes.clear();
  for (const auto& n : need(tr, "classes", "tracker")) {
    const int k = class_index(n.as<std::string>());
    if (k < 0) throw Error("config: tracker.classes: unknown class " + n.as<std::string>());
    t.classes.push_back(k);
  }
  get_per_class(tr, "high_score", t.high_score, "tracker");
  get(tr, "low_score", t.low_score, "tracker");
  get(tr, "matching", t.matching, "tracker");
  if (t.matching != "greedy" && t.matching != "hungarian") throw Error("config: tracker.matching: greedy or hungarian");
  get(tr, "use_velocity", t.use_velocity, "tracker");
  get(tr, "tentative_max_age", t.tentative_max_age, "tracker");
  get(tr, "output_coasting", t.output_coasting, "tracker");
  get_per_class(tr, "output_min_score", t.output_min_score, "tracker");
  get(tr, "history", t.history, "tracker");
  get(tr, "size_alpha", t.size_alpha, "tracker");
  get(tr, "z_alpha", t.z_alpha, "tracker");
  get(tr, "yaw_alpha", t.yaw_alpha, "tracker");
  get(tr, "heading_from_velocity", t.heading_from_velocity, "tracker");
  get_per_class(tr, "motion_on_speed", t.motion_on_speed, "tracker");
  get_per_class(tr, "motion_off_speed", t.motion_off_speed, "tracker");
  get(tr, "motion_window", t.motion_window, "tracker");
  get(tr, "motion_traj_window", t.motion_traj_window, "tracker");
  get(tr, "motion_on_frames", t.motion_on_frames, "tracker");
  get(tr, "motion_off_time", t.motion_off_time, "tracker");
  get(tr, "stop_memory", t.stop_memory, "tracker");

  const YAML::Node pr = need(root, "predictor", "");
  get(pr, "enabled", c.predictor.enabled, "predictor");
  get(pr, "horizon", c.predictor.horizon, "predictor");
  get(pr, "step", c.predictor.step, "predictor");
  get(pr, "merge_dist", c.predictor.merge_dist, "predictor");
  return c;
}

}  // namespace

PipelineConfig parse_config(const std::string& text) {
  try {
    return from_node(YAML::Load(text));
  } catch (const YAML::Exception& e) {
    throw Error(std::string("config: ") + e.what());
  }
}

PipelineConfig load_config(const std::string& path) {
  try {
    return from_node(YAML::LoadFile(path));
  } catch (const YAML::Exception& e) {
    throw Error(path + ": " + e.what());
  }
}

}  // namespace dogperc
