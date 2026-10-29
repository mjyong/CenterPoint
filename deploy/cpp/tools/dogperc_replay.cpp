// Replay a recorded sequence through the C++ pipeline; per-frame results to one .npz
// (the layout of tools/deploy/dump_golden.py's pipeline.npz; tools/deploy/cpp_parity.py reads it).
//
//   dogperc_replay --seq data/seq/0921_164127_s000 --config seq.yaml --out cpp.npz
//       [--detections dets.npz]                                      the detections of another run, frame by frame
//       [--model-dir data/deploy/nusc_pp --precision fp16 [--gpu]]   TensorRT engines (DOGPERC_WITH_CUDA)
//       [--odom-latency 0.0] [--save-points] [--max-frames N]
//       [--threads 1]                                                  per-point preprocessing loops (same results)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "dogperc/detection/filters.hpp"
#include "dogperc/npz.hpp"
#include "dogperc/parallel.hpp"
#include "dogperc/pipeline.hpp"
#include "dogperc/sequence.hpp"
#ifdef DOGPERC_WITH_CUDA
#include "dogperc/detection/gpu_detector.hpp"
#include "dogperc/detection/trt_engine.hpp"
#endif

using namespace dogperc;

namespace {

void usage() {
  std::cerr << "usage: dogperc_replay --seq DIR --config YAML --out OUT.npz [--detections DETS.npz] "
               "[--model-dir DIR --precision fp16 [--gpu]] [--odom-latency S] [--save-points] [--max-frames N] "
               "[--threads N]\n";
}

struct Writer {
  std::vector<double> stamp, T_wd, T_wb, ground_offset, ground_tilt;
  std::vector<int64_t> num_raw, num_kept, point_offsets{0};
  std::vector<float> points;
  std::map<std::string, std::vector<double>> timings;
  // detections (final and raw)
  struct Dets {
    std::vector<int64_t> offsets{0}, labels, subclasses;
    std::vector<double> boxes, velocities, scores;
    void add(const Detections& d) {
      for (size_t i = 0; i < d.size(); ++i) {
        boxes.insert(boxes.end(), d.boxes[i].begin(), d.boxes[i].end());
        velocities.insert(velocities.end(), d.velocities[i].begin(), d.velocities[i].end());
        scores.push_back(d.scores[i]);
        labels.push_back(d.labels[i]);
        subclasses.push_back(d.subclasses[i]);
      }
      offsets.push_back(offsets.back() + static_cast<int64_t>(d.size()));
    }
    void write(NpzWriter& w, const std::string& p) const {
      const int64_t n = static_cast<int64_t>(scores.size());
      w.add(p + "_offsets", offsets);
      w.add(p + "_boxes", boxes, {n, 7});
      w.add(p + "_velocities", velocities, {n, 2});
      w.add(p + "_scores", scores);
      w.add(p + "_labels", labels);
      w.add(p + "_subclasses", subclasses);
    }
  } det, raw;
  std::vector<double> raw_bottom;
  std::vector<int64_t> raw_counts;
  std::vector<int64_t> trk_offsets{0}, trk_id, trk_label, trk_hits, trk_coasting, trk_motion, trk_history_len;
  std::vector<double> trk_position, trk_velocity, trk_yaw, trk_size, trk_score, trk_cov, trk_mode_probs, trk_age,
      trk_tsm, trk_sd, trk_sh, trk_meas;
  std::vector<int64_t> pred_offsets{0}, pred_track_id, pred_label, pred_num_modes;
  std::vector<double> pred_modes, pred_probs, pred_covs, pred_times;
  int64_t T = 0;

  void add(const PipelineOutput& o, const Preprocessor& pre, bool save_points, bool have_raw, const Detections& raw_d,
           const DetectionFilterConfig& filter) {
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
    if (have_raw) {
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

  void write(const std::string& path, bool have_raw) {
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
    for (auto& [k, v] : timings) {
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
};

}  // namespace

int main(int argc, char** argv) {
  std::string seq, config, out, dets_path, model_dir, precision = "fp16";
  double odom_latency = 0.0;
  bool save_points = false, gpu = false;
  long max_frames = -1;
  int threads = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--seq") seq = next();
    else if (a == "--config") config = next();
    else if (a == "--out") out = next();
    else if (a == "--detections") dets_path = next();
    else if (a == "--model-dir") model_dir = next();
    else if (a == "--precision") precision = next();
    else if (a == "--gpu") gpu = true;
    else if (a == "--odom-latency") odom_latency = std::stod(next());
    else if (a == "--save-points") save_points = true;
    else if (a == "--max-frames") max_frames = std::stol(next());
    else if (a == "--threads") threads = std::stoi(next());
    else {
      usage();
      return 2;
    }
  }
  if (seq.empty() || config.empty() || out.empty() || (dets_path.empty() == model_dir.empty())) {
    usage();
    std::cerr << "exactly one of --detections / --model-dir\n";
    return 2;
  }
  set_num_threads(threads);
  try {
    const PipelineConfig cfg = load_config(config);
    std::unique_ptr<Detector> detector;
    if (!model_dir.empty()) {
#ifdef DOGPERC_WITH_CUDA
      if (gpu) detector = make_gpu_detector(model_dir, precision, cfg.detection_filter);
      else detector = make_trt_detector(model_dir, precision, cfg.detection_filter);
#else
      (void)gpu;
      throw Error("--model-dir needs a build with DOGPERC_WITH_CUDA=ON");
#endif
    }
    Npz dets;
    std::vector<double> det_stamps;
    if (!dets_path.empty()) {
      dets = read_npz(dets_path);
      det_stamps = member(dets, "stamp").to<double>();
    }
    PerceptionPipeline pipe(cfg, detector.get());
    const SequenceReader reader(seq);
    Writer w;
    size_t k = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& ev : reader.events(odom_latency)) {
      if (const auto* i = std::get_if<ImuEvent>(&ev)) {
        pipe.on_imu(i->t, i->gyro);
        continue;
      }
      if (const auto* o = std::get_if<OdomEvent>(&ev)) {
        pipe.on_odometry(o->t, o->R, o->p, o->v);
        continue;
      }
      const auto& s = std::get<ScanEvent>(ev);
      std::optional<PipelineOutput> o;
      if (!dets_path.empty()) {
        // the other run's frames in order: a scan it made no frame for gets none here either
        Detections d;
        const bool match = k < det_stamps.size() && std::abs(det_stamps[k] - s.stamp) < 1e-9;
        if (match) {
          const auto& off = member(dets, "offsets");
          for (int64_t r = off.at<int64_t>(k); r < off.at<int64_t>(k + 1); ++r) {
            Box b;
            for (int c = 0; c < 7; ++c) b[c] = member(dets, "boxes").at<double>(7 * r + c);
            d.push(b, {member(dets, "velocities").at<double>(2 * r), member(dets, "velocities").at<double>(2 * r + 1)},
                   member(dets, "scores").at<double>(r), member(dets, "labels").at<int>(r), member(dets, "subclasses").at<int>(r));
          }
        }
        o = pipe.on_scan(reader.load_scan(s.index), &d);
        if (o && !match) std::fprintf(stderr, "warning: frame at %.6f has no detections in %s\n", s.stamp, dets_path.c_str());
        if (o && match) ++k;
      } else {
        o = pipe.on_scan(reader.load_scan(s.index));
      }
      if (!o) continue;
      w.add(*o, pipe.preprocessor(), save_points, detector != nullptr, detector ? detector->last_raw() : Detections{},
            cfg.detection_filter);
      if (max_frames > 0 && static_cast<long>(w.stamp.size()) >= max_frames) break;
    }
    w.write(out, detector != nullptr);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu frames, %zu tracks, %.1f s -> %s\n", w.stamp.size(), w.trk_id.size(), sec, out.c_str());
  } catch (const std::exception& e) {
    std::cerr << "dogperc_replay: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
