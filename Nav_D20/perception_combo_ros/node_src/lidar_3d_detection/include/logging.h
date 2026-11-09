#ifndef LIDAR_3D_DETECTION_LOGGING_H
#define LIDAR_3D_DETECTION_LOGGING_H

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <filesystem>
#include <string>

namespace seres {
namespace perception {

class Logger {
 public:
  static bool Init(const std::string& log_dir = "/tmp/lidar_3d_detection_logs",
                   const std::string& program_name = "lidar_3d_detection",
                   bool also_log_to_stderr = true,
                   int stderr_log_level = google::GLOG_INFO) {
    std::filesystem::create_directories(log_dir);
    FLAGS_log_dir = log_dir;
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = also_log_to_stderr;
    FLAGS_stderrthreshold = stderr_log_level;
    FLAGS_max_log_size = 50;
    FLAGS_logbuflevel = -1;
    FLAGS_logbufsecs = 10;
    FLAGS_stop_logging_if_full_disk = true;
    google::InitGoogleLogging(program_name.c_str());
    google::SetLogFilenameExtension(".log");
    google::InstallFailureSignalHandler();
    LOG(INFO) << "Lidar 3D detection logger initialized. Log directory: " << log_dir;
    return true;
  }

  static void Shutdown() {
    google::FlushLogFiles(google::INFO);
    google::ShutdownGoogleLogging();
  }
};

}  // namespace perception
}  // namespace seres

#endif
