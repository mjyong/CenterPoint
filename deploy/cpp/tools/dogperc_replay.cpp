// Replay a recorded sequence through the C++ pipeline; per-frame results to one .npz
// (the layout of tools/deploy/dump_golden.py's pipeline.npz; tools/deploy/cpp_parity.py reads it).
//
//   dogperc_replay --seq data/seq/0921_164127_s000 --config seq.yaml --out cpp.npz
//   dogperc_replay --messages msgs.npz --config kygi.yaml --out cpp.npz      the online node's path (OnlineInputs)
//                                       on a bag's messages (cpp_parity.py python --bag --messages); the config
//                                       needs the node: section (export_cpp_config.py --robot)
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

#include "dogperc/message_log.hpp"
#include "dogperc/parallel.hpp"
#include "dogperc/pipeline.hpp"
#include "dogperc/replay_writer.hpp"
#include "dogperc/sequence.hpp"
#ifdef DOGPERC_WITH_CUDA
#include "dogperc/detection/gpu_detector.hpp"
#include "dogperc/detection/trt_engine.hpp"
#endif

using namespace dogperc;

namespace {

void usage() {
  std::cerr << "usage: dogperc_replay (--seq DIR | --messages MSGS.npz) --config YAML --out OUT.npz [--detections DETS.npz] "
               "[--model-dir DIR --precision fp16 [--gpu]] [--odom-latency S] [--save-points] [--max-frames N] "
               "[--threads N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string seq, messages, config, out, dets_path, model_dir, precision = "fp16";
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
    else if (a == "--messages") messages = next();
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
  if (seq.empty() == messages.empty() || config.empty() || out.empty() || (dets_path.empty() == model_dir.empty())) {
    usage();
    std::cerr << "exactly one of --seq / --messages, and one of --detections / --model-dir\n";
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
    std::unique_ptr<GivenDetections> given;
    if (!dets_path.empty()) given = std::make_unique<GivenDetections>(dets_path);
    PerceptionPipeline pipe(cfg, detector.get());
    ReplayWriter w;
    // one scan: the given detections (the other run's frames in order) or the detector
    auto frame = [&](double stamp, auto&& run) -> bool {
      std::optional<PipelineOutput> o;
      if (given) {
        Detections d;
        const bool match = given->lookup(stamp, d);
        o = run(&d);
        if (o && !match) std::fprintf(stderr, "warning: frame at %.6f has no detections in %s\n", stamp, dets_path.c_str());
        if (o && match) given->advance();
      } else {
        o = run(nullptr);
      }
      if (!o) return true;
      const Detections raw = detector ? detector->last_raw() : Detections{};
      w.add(*o, pipe.preprocessor(), save_points, detector ? &raw : nullptr, cfg.detection_filter);
      return !(max_frames > 0 && static_cast<long>(w.frames()) >= max_frames);
    };
    const auto t0 = std::chrono::steady_clock::now();
    if (!seq.empty()) {
      const SequenceReader reader(seq);
      for (const auto& ev : reader.events(odom_latency)) {
        if (const auto* i = std::get_if<ImuEvent>(&ev)) {
          pipe.on_imu(i->t, i->gyro);
        } else if (const auto* o = std::get_if<OdomEvent>(&ev)) {
          pipe.on_odometry(o->t, o->R, o->p, o->v);
        } else {
          const auto& s = std::get<ScanEvent>(ev);
          const LidarScan scan = reader.load_scan(s.index);
          if (!frame(s.stamp, [&](const Detections* d) { return pipe.on_scan(scan, d); })) break;
        }
      }
    } else {
      const MessageLog log(messages);
      OnlineInputs inputs(load_node_config(config), pipe, [](LogLevel level, const std::string& text) {
        std::fprintf(stderr, "%s: %s\n", level == LogLevel::Error ? "error" : level == LogLevel::Warn ? "warn" : "info",
                     text.c_str());
      });
      for (size_t i = 0; i < log.size(); ++i) {
        if (log.kind(i) != MessageLog::kCloud) {
          log.feed(inputs, i);
          continue;
        }
        const CloudView cloud = log.cloud(log.index(i));
        const LidarScan scan = inputs.scan(cloud);
        if (!frame(scan.stamp, [&](const Detections* d) { return pipe.on_scan(scan, d); })) break;
      }
    }
    w.write(out);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%zu frames, %zu tracks, %.1f s -> %s\n", w.frames(), w.track_rows(), sec, out.c_str());
  } catch (const std::exception& e) {
    std::cerr << "dogperc_replay: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
