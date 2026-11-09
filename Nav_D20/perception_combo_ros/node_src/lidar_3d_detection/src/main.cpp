#include <cstdlib>
#include <string>

#include "../include/lidar_3d_detection.h"
#include "../include/logging.h"

int main(int argc, char** argv) {
  const char* log_dir = std::getenv("LIDAR3D_LOG_DIR");
  seres::perception::Logger::Init(log_dir && *log_dir ? log_dir : "/tmp/lidar_3d_detection_logs", "lidar_3d_detection");
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<seres::perception::Lidar3dDetection>();
    if (!node->Init()) {
      LOG(ERROR) << "Init lidar_3d_detection failed";
      code = 1;
    } else {
      rclcpp::spin(node);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "lidar_3d_detection: " << e.what();
    code = 1;
  }
  rclcpp::shutdown();
  seres::perception::Logger::Shutdown();
  return code;
}
