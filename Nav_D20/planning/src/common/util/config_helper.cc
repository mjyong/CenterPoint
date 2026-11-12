#include "config_helper.h"
#include <algorithm>
#include <cmath>
#include <fstream>
namespace planning {
namespace common {
namespace util {
RobotConfig ConfigHelper::robot_config_;
bool ConfigHelper::is_init_ = false;
std::string ConfigHelper::robot_config_dir_;
std::vector<ConfigHelper::Circle2d> ConfigHelper::bounding_circles_;
ConfigHelper::ConfigHelper() {
  auto PLANNING_ROOT_DIR_ENV = std::getenv("PLANNING_ROOT_DIR");
  ACHECK(PLANNING_ROOT_DIR_ENV != nullptr) << "PLANNING_ROOT_DIR is unset.";
  std::string PLANNING_ROOT_DIR(PLANNING_ROOT_DIR_ENV);
  std::string robot_vin_config =
      PLANNING_ROOT_DIR + std::string("/robot_factory/default_robot_vin");
  std::ifstream fin;
  fin.open(robot_vin_config, std::ios::in);
  ACHECK(fin.is_open()) << "Can not open file: " << robot_vin_config;
  std::string buff;
  getline(fin, buff);
  fin.close();
  robot_config_dir_ = PLANNING_ROOT_DIR + std::string("/robot_factory/") + buff;
  AINFO << "Load robot config:" << robot_config_dir_;
}

void ConfigHelper::Init() {
  Init(robot_config_dir_ + std::string("/robot_param.pb.txt"));
}

void ConfigHelper::Init(const std::string &config_file) {
  RobotConfig params;
  ACHECK(GetProtoFromFile(config_file, &params))
      << "Unable to parse robot config file " << config_file;
  Init(params);
}

void ConfigHelper::Init(const RobotConfig &robot_params) {
  robot_config_ = robot_params;
  is_init_ = true;
  InitBoundingCircles();
}

const RobotConfig &ConfigHelper::GetConfig() {
  if (!is_init_) {
    Init();
  }
  return robot_config_;
}

ConfigHelper::Circle2d ConfigHelper::GetBoundingCircle(const double &x,
                                                       const double &y,
                                                       const double &theta) {
  Eigen::Vector2d true_center(x, y);
  double radius = std::hypot(robot_config_.robot_param().length() / 2.0,
                             robot_config_.robot_param().width() / 2.0);
  return Circle2d(true_center, radius);
}

std::vector<ConfigHelper::Circle2d>
ConfigHelper::GetBoundingCircles(const double &x, const double &y,
                                 const double &theta) {
  std::vector<ConfigHelper::Circle2d> bounding_circles;
  for (auto &circle : bounding_circles_) {
    Eigen::Vector2d circle_center(x + circle.center_x * std::cos(theta) -
                                      circle.center_y * std::sin(theta),
                                  y + circle.center_x * std::sin(theta) +
                                      circle.center_y * std::cos(theta));
    bounding_circles.emplace_back(Circle2d{circle_center, circle.radius});
  }
  return bounding_circles;
}

void ConfigHelper::InitBoundingCircles() {
  bounding_circles_.clear();
  const auto &rp = robot_config_.robot_param();
  const double L = rp.length();
  const double W = rp.width();

  switch (rp.footprint_model()) {
  case RobotParam::SINGLE: {
    // 单大圆: 外接矩形对角线一半, 与 GetBoundingCircle 保持一致
    bounding_circles_.emplace_back(
        Circle2d{0.0, 0.0, std::hypot(L / 2.0, W / 2.0)});
    break;
  }
  case RobotParam::DOUBLE: {
    const double offset = L / 4.0;
    const double r = std::hypot(offset, W / 2.0);
    bounding_circles_.emplace_back(Circle2d{offset, 0.0, r});
    bounding_circles_.emplace_back(Circle2d{-offset, 0.0, r});
    break;
  }
  case RobotParam::MULTI:
  default: {
    Eigen::Vector2d fl(L / 2.0, W / 2.0);
    Eigen::Vector2d fr(L / 2.0, -W / 2.0);
    Eigen::Vector2d rl(-L / 2.0, W / 2.0);
    Eigen::Vector2d rr(-L / 2.0, -W / 2.0);

    double small_circle_shift = W / 4.0;
    double small_circle_radius = sqrt(2.0 * pow(small_circle_shift, 2));
    double large_circle_radius =
        sqrt(pow(W, 2) + pow((L - W) / 2.0, 2)) / 2.0;
    // rr
    bounding_circles_.emplace_back(Circle2d{rr.x() + small_circle_shift,
                                            rr.y() + small_circle_shift,
                                            small_circle_radius});
    // rl
    bounding_circles_.emplace_back(Circle2d{rl.x() + small_circle_shift,
                                            rl.y() - small_circle_shift,
                                            small_circle_radius});
    // fr
    bounding_circles_.emplace_back(Circle2d{fr.x() - small_circle_shift,
                                            fr.y() + small_circle_shift,
                                            small_circle_radius});
    // fl
    bounding_circles_.emplace_back(Circle2d{fl.x() - small_circle_shift,
                                            fl.y() - small_circle_shift,
                                            small_circle_radius});
    // fm
    bounding_circles_.emplace_back(Circle2d{(L - W) / 4, 0,
                                            large_circle_radius});
    // rm
    bounding_circles_.emplace_back(Circle2d{(W - L) / 4, 0,
                                            large_circle_radius});
    break;
  }
  }
}

} // namespace util
} // namespace common
} // namespace planning