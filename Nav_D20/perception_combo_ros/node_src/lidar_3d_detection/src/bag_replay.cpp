// A rosbag2 through the node's message handling, offline: the bag's messages in storage order (what the
// node gets from `ros2 bag play`), deserialized as the node receives them, into the same OnlineInputs.
// Writes lidar_3d_replay's .npz, so it compares with the Python node's replay of the same bag
// (tools/deploy/cpp_parity.py python --bag) and with lidar_3d_replay --messages on its message log:
//
//   lidar_3d_bag_replay --bag <bag> --config kygi.yaml --detections py_bag_dets.npz --out bag.npz
//   lidar_3d_bag_replay --bag <bag> --config kygi.yaml --model-dir nusc_pp_dyn --gpu --out bag.npz
//   lidar_3d_bag_replay --bag <bag> --config kygi.yaml --preprocess-only --out bag.npz   no detector (empty detections)
//       [--save-points] [--max-clouds N] [--threads N]
// One line per cloud on stdout (points in and out, preprocessing time, or why it was dropped), then a summary.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <string>

#include "lidar3d/parallel.hpp"
#include "lidar3d/replay_writer.hpp"
#include "ros_convert.hpp"
#ifdef LIDAR3D_WITH_CUDA
#include "lidar3d/detection/gpu_detector.hpp"
#include "lidar3d/detection/trt_engine.hpp"
#endif

using namespace lidar3d;

namespace {

template <class T>
T deserialize(const rosbag2_storage::SerializedBagMessage& bm) {
  rclcpp::SerializedMessage ser(*bm.serialized_data);
  T msg;
  rclcpp::Serialization<T>().deserialize_message(&ser, &msg);
  return msg;
}

void usage() {
  std::cerr << "usage: lidar_3d_bag_replay --bag DIR --config YAML --out OUT.npz "
               "(--detections DETS.npz | --model-dir DIR [--precision fp16] [--gpu] | --preprocess-only) "
               "[--save-points] [--max-clouds N] [--threads N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string bag, config, out, dets_path, model_dir, precision = "fp16";
  bool save_points = false, gpu = false, preprocess_only = false;
  long max_clouds = 0;
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
    if (a == "--bag") bag = next();
    else if (a == "--config") config = next();
    else if (a == "--out") out = next();
    else if (a == "--detections") dets_path = next();
    else if (a == "--model-dir") model_dir = next();
    else if (a == "--precision") precision = next();
    else if (a == "--gpu") gpu = true;
    else if (a == "--save-points") save_points = true;
    else if (a == "--preprocess-only") preprocess_only = true;
    else if (a == "--max-clouds") max_clouds = std::stol(next());
    else if (a == "--threads") threads = std::stoi(next());
    else {
      usage();
      return 2;
    }
  }
  if (bag.empty() || config.empty() || out.empty() ||
      (!preprocess_only && dets_path.empty() == model_dir.empty())) {
    usage();
    return 2;
  }
  set_num_threads(threads);
  try {
    const PipelineConfig cfg = load_config(config);
    const NodeConfig nc = load_node_config(config);
    std::unique_ptr<Detector> detector;
    if (!model_dir.empty()) {
#ifdef LIDAR3D_WITH_CUDA
      detector = gpu ? make_gpu_detector(model_dir, precision, cfg.detection_filter)
                     : make_trt_detector(model_dir, precision, cfg.detection_filter);
#else
      (void)gpu;
      throw Error("--model-dir needs a build with LIDAR3D_WITH_CUDA=ON");
#endif
    }
    std::unique_ptr<GivenDetections> given;
    if (!dets_path.empty()) given = std::make_unique<GivenDetections>(dets_path);
    PerceptionPipeline pipe(cfg, detector.get());
    OnlineInputs inputs(nc, pipe, [](LogLevel level, const std::string& text) {
      std::fprintf(stderr, "%s: %s\n", level == LogLevel::Error ? "error" : level == LogLevel::Warn ? "warn" : "info",
                   text.c_str());
    });
    ReplayWriter w;
    rosbag2_cpp::Reader reader;
    reader.open(bag);
    long clouds = 0, messages = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (reader.has_next()) {
      const auto bm = reader.read_next();
      const std::string& topic = bm->topic_name;
      if (topic == nc.imu_topic) {
        const auto m = deserialize<sensor_msgs::msg::Imu>(*bm);
        inputs.on_imu(ros::seconds(m.header.stamp), ros::angular_velocity(m));
      } else if (nc.localization == "odom" && topic == nc.odom_topic) {
        inputs.on_odom(ros::odom(deserialize<nav_msgs::msg::Odometry>(*bm)));
      } else if (nc.localization == "inspvax" && topic == nc.inspvax_topic) {
#ifdef LIDAR3D_HAVE_KYGI
        inputs.on_inspvax(ros::ins_sample(deserialize<kygi665_msgs::msg::INSPVAX>(*bm)));
#else
        throw Error("localization inspvax: built without kygi665_msgs");
#endif
      } else if (topic == nc.lidar_topic) {
        if (max_clouds > 0 && clouds >= max_clouds) break;
        ++clouds;
        const auto m = deserialize<sensor_msgs::msg::PointCloud2>(*bm);
        const LidarScan scan = inputs.scan(ros::cloud_view(m));
        std::optional<PipelineOutput> o;
        if (preprocess_only) {
          const Detections empty;
          o = inputs.on_scan(scan, &empty);
        } else if (given) {
          Detections d;
          const bool match = given->lookup(scan.stamp, d);
          o = inputs.on_scan(scan, &d);
          if (o && match) given->advance();
          if (o && !match) std::fprintf(stderr, "warning: frame at %.6f has no detections in %s\n", scan.stamp, dets_path.c_str());
        } else {
          o = inputs.on_scan(scan);
        }
        if (!o) std::printf("cloud %ld stamp %.3f points_in %zu dropped (no pose)\n", clouds, scan.stamp, scan.size());
        if (o) {
          std::printf("cloud %ld stamp %.3f points_in %zu points_out %zu preprocess %.1f ms\n", clouds, o->stamp,
                      scan.size(), o->frame.num_points(), o->timings.count("preprocess_ms") ? o->timings.at("preprocess_ms") : 0.0);
          const Detections raw = detector ? detector->last_raw() : Detections{};
          w.add(*o, pipe.preprocessor(), save_points, detector ? &raw : nullptr, cfg.detection_filter);
        }
      } else {
        continue;
      }
      ++messages;
    }
    w.write(out);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%ld messages, %ld clouds, %zu frames, %zu tracks, %.1f s -> %s\n", messages, clouds, w.frames(),
                w.track_rows(), sec, out.c_str());
  } catch (const std::exception& e) {
    std::cerr << "lidar_3d_bag_replay: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
