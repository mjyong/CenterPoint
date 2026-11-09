#ifndef SERES_PERCEPTION_PTZ_LOGGING_H
#define SERES_PERCEPTION_PTZ_LOGGING_H

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <string>
#include <filesystem>

namespace seres {
namespace perception {

class PtzLogger {
public:
    static bool Init(const std::string& log_dir = "/tmp/ptz_data_detection_logs",
                     const std::string& program_name = "ptz_data_detection_2d",
                     bool also_log_to_stderr = true,
                     int stderr_log_level = google::GLOG_INFO,
                     int max_log_size = 50) {
        std::filesystem::create_directories(log_dir);
        FLAGS_log_dir = log_dir;
        FLAGS_logtostderr = false;
        FLAGS_alsologtostderr = also_log_to_stderr;
        FLAGS_stderrthreshold = stderr_log_level;
        FLAGS_max_log_size = max_log_size;
        FLAGS_logbuflevel = -1;
        FLAGS_logbufsecs = 10;
        FLAGS_stop_logging_if_full_disk = true;
        google::InitGoogleLogging(program_name.c_str());
        google::SetLogFilenameExtension(".log");
        google::InstallFailureSignalHandler();
        LOG(INFO) << "PTZ Data Detection Logger initialized.";
        LOG(INFO) << "Log directory: " << log_dir;
        return true;
    }
    static void Shutdown() {
        google::FlushLogFiles(google::INFO);
        google::ShutdownGoogleLogging();
    }
    static void Flush() { google::FlushLogFiles(google::INFO); }
};

} // namespace perception
} // namespace seres

#endif
