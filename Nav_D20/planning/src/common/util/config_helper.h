#pragma once

#include "config_helper.pb.h"
#include "log.h"
#include "macros.h"
#include <Eigen/Core>
#include <string>
#include <sys/fcntl.h>
#include <sys/stat.h>
namespace planning {
namespace common {
namespace util {
using config::RobotConfig;
using config::RobotParam;

class ConfigHelper {
public:
  struct Circle2d {
    double center_x;
    double center_y;
    double radius;
    Circle2d() = default;
    Circle2d(const Eigen::Vector2d &_center, double _radius)
        : center_x(_center.x()), center_y(_center.y()), radius(_radius) {}
    Circle2d(double _center_x, double _center_y, double _radius)
        : center_x(_center_x), center_y(_center_y), radius(_radius) {}
  };

  static void Init();
  static void Init(const RobotConfig &config);
  static void Init(const std::string &config_file);
  static const RobotConfig &GetConfig();
  static const std::string &GetConfDir() { return robot_config_dir_; }
  static bool GetProtoFromFile(const std::string &file_name,
                               google::protobuf::Message *message) {
    using google::protobuf::TextFormat;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    int file_descriptor = open(file_name.c_str(), O_RDONLY);
    if (file_descriptor < 0) {
      std::cerr << "Failed to open file " << file_name << " in text mode."
                << std::endl;
      return false;
    }

    ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
    bool success = TextFormat::Parse(input, message);
    if (!success) {
      std::cerr << "Failed to parse file " << file_name << " as text proto."
                << std::endl;
    }
    delete input;
    close(file_descriptor);
    return success;
  }
  static Circle2d GetBoundingCircle(const double &x, const double &y,
                                    const double &theta);

  static std::vector<Circle2d>
  GetBoundingCircles(const double &x, const double &y, const double &theta);

private:
  static void InitBoundingCircles();

private:
  ConfigHelper();
  static RobotConfig robot_config_;
  static bool is_init_;
  static std::string robot_config_dir_;
  static std::vector<Circle2d> bounding_circles_;
  BASE_DECLARE_SINGLETON(ConfigHelper)
};

} // namespace util
} // namespace common
} // namespace planning