#ifndef GRID_MAP_LOGGING_H
#define GRID_MAP_LOGGING_H

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <string>
#include <memory>
#include <filesystem>

namespace grid_map {

class Logger {
public:
    static bool Init(const std::string& log_dir = "/tmp/grid_map_logs", 
                     const std::string& program_name = "grid_map_memory_node",
                     bool also_log_to_stderr = true,
                     int stderr_log_level = google::GLOG_INFO,
                     int max_log_size = 10,
                     int max_num_logs = 10) {
        
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
        
        LOG(INFO) << "GridMap Logger initialized.";
        LOG(INFO) << "Log directory: " << log_dir;
        LOG(INFO) << "alsologtostderr: " << FLAGS_alsologtostderr;
        LOG(INFO) << "max_log_size: " << FLAGS_max_log_size << " MB";
        
        return true;
    }
    
    static void Shutdown() {
        google::FlushLogFiles(google::INFO);
        google::ShutdownGoogleLogging();
    }
    
    static void SetMinLogLevel(int level) {
        FLAGS_minloglevel = level;
    }
    
    static void SetLogDestination(int severity, const std::string& path) {
        google::SetLogDestination(severity, path.c_str());
    }
    
    static void Flush() {
        google::FlushLogFiles(google::INFO);
    }
};

} // namespace grid_map

#endif // GRID_MAP_LOGGING_H
