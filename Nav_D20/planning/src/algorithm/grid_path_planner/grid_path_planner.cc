#include "src/algorithm/grid_path_planner/astar_search.h"
#include "src/algorithm/grid_path_planner/jump_point_search.h"
#include "src/algorithm/grid_path_planner/theta_astar_search.h"
#include "src/algorithm/grid_path_planner/voronoi_astar_search.h"
#include "src/common/util/config_helper.h"
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>   // OpenCV 5 moved the 2D geometry functions here (the server's build env)
#endif
#include <yaml-cpp/yaml.h>

#include "src/algorithm/grid_path_planner/eigen2cv.hpp"
#include "src/algorithm/grid_path_planner/grid_path_planner.h"
using planning::common::util::ConfigHelper;

namespace planning {
namespace algorithm {
GridPathPlanner::GridPathPlanner(const GridPathPlannerConf &config)
    : config_(config), grid_map_(std::vector<std::string>{
                           "obstacle", "distance", "distance_cost"}) {}

GridPathPlanner::~GridPathPlanner() {}

bool GridPathPlanner::initGridMap() {
  std::string map_file_name = ConfigHelper::Instance()->GetConfDir() +
                              config_.input_dir() + config_.input_name();
  std::string cost_map_path = map_file_name + "_costmap.pgm";
  std::string dist_map_path = map_file_name + "_distmap.pgm";
  std::string cost_map_conf_path = map_file_name + "_costmap.yaml";
  if (std::filesystem::exists(cost_map_path) &&
      std::filesystem::exists(cost_map_conf_path) &&
      std::filesystem::exists(dist_map_path)) {
    if (!loadCostMap(cost_map_path, dist_map_path, cost_map_conf_path)) {
      return false;
    }
  } else {
    switch (config_.input_type()) {
    case GridPathPlannerConf::PGM_MAP:
      if (!loadFromPGM(map_file_name)) {
        AERROR << "Failed to load map from pgm file";
        return false;
      }
      break;
    default:
      break;
    }
    calcCostMap();
    if (!saveCostMap(cost_map_path, dist_map_path, cost_map_conf_path)) {
      return false;
    }
  }

  return true;
}

void GridPathPlanner::SetGridMap(const grid_map::GridMap &map) {
  if (!map.exists("obstacle")) {
    AERROR << "SetGridMap: source map has no 'obstacle' layer";
    return;
  }
  grid_map_.setGeometry(map.getLength(), map.getResolution(), map.getPosition());
  grid_map_.setFrameId(map.getFrameId());
  auto &obstacle = grid_map_["obstacle"];
  obstacle = map["obstacle"];
  calcCostMap();
  AINFO << "SetGridMap: grid_map_ updated from GlobalMapManager, size: "
        << grid_map_.getSize()(0) << "x" << grid_map_.getSize()(1);
}

bool GridPathPlanner::initGridMapFromMatrix(
    const Eigen::MatrixXd &obstacle_matrix, double resolution,
    const Eigen::Vector2d &origin, double occupied_thresh, double free_thresh,
    bool negate) {
  if (obstacle_matrix.rows() <= 0 || obstacle_matrix.cols() <= 0) {
    AERROR << "Invalid obstacle matrix dimensions: " << obstacle_matrix.rows()
           << "x" << obstacle_matrix.cols();
    return false;
  }
  if (resolution <= 0) {
    AERROR << "Invalid resolution: " << resolution;
    return false;
  }
  const double length_x = resolution * obstacle_matrix.rows();
  const double length_y = resolution * obstacle_matrix.cols();
  grid_map::Length length(length_x, length_y);
  grid_map::Position position(origin.x() + length_x / 2.0,
                              origin.y() + length_y / 2.0);
  grid_map_.setGeometry(length, resolution, position);
  grid_map_.setFrameId("map");
  auto &obstacle = grid_map_.get("obstacle");
  obstacle = obstacle_matrix.cast<float>();
  if (config_.unknown_as_free()) {
    obstacle = (obstacle.array() > occupied_thresh)
                   .select(Eigen::MatrixXf::Constant(obstacle.rows(),
                                                     obstacle.cols(), 255.0f),
                           Eigen::MatrixXf::Constant(obstacle.rows(),
                                                     obstacle.cols(), 0.0f));
  } else {
    obstacle = (obstacle.array() < free_thresh)
                   .select(Eigen::MatrixXf::Constant(obstacle.rows(),
                                                     obstacle.cols(), 0.0f),
                           Eigen::MatrixXf::Constant(obstacle.rows(),
                                                     obstacle.cols(), 255.0f));
  }
  calcCostMap();
  AINFO << "Successfully loaded grid map from matrix. Dimensions: "
        << obstacle_matrix.rows() << "x" << obstacle_matrix.cols()
        << ", Resolution: " << resolution << ", Origin: (" << origin.x() << ", "
        << origin.y() << ")";
  return true;
}

bool GridPathPlanner::plan(Eigen::Vector2d start, Eigen::Vector2d end,
                           std::vector<Eigen::Vector2d> &path) {
  std::unique_ptr<PlannerBase> planner;
  switch (config_.planner_type()) {
  case GridPathPlannerConf::JPS:
    planner = std::make_unique<JumpPointSearch>();
    break;
  case GridPathPlannerConf::A_STAR:
    planner = std::make_unique<AstarSearch>();
    break;
  case GridPathPlannerConf::THETA_A_STAR:
    planner = std::make_unique<ThetaAstarSearch>();
    break;
  case GridPathPlannerConf::VORONOI_ASTAR:
    planner = std::make_unique<VoronoiAstarSearch>(&voronoi_);
    InitVoronoi();
    break;
  default:
    planner = std::make_unique<AstarSearch>();
  }
  if (config_.allow_collision_goal() && isCollision(end)) {
    Eigen::Vector2d adjusted_end = findNearestFreePoint(end);
    if (adjusted_end == end) {
      AERROR << "No free point found near goal within search radius";
      return false;
    }
    AINFO << "Goal adjusted from " << end.transpose() << " to "
          << adjusted_end.transpose();
    end = adjusted_end;
  }
  // The planners reject a start whose distance_cost is above 0.99 ("Start point is in collision"), and A* does not
  // step into a cell above 0.9 (AstarSearch::plan). A start between the two passes the first check and then has no
  // neighbour to go to: "GridPathPlanner plan path failed", the mission goes to fault (T49: the dog 0.22 m from the
  // edge of a 1.7 m CJGC road, distance_cost 0.96). So move the start whenever A* could not leave it.
  constexpr double kAstarStepCost = 0.9;   // AstarSearch::plan's neighbour check
  if (config_.allow_collision_start() && grid_map_.isInside(start) &&
      grid_map_.atPosition("distance_cost", start) > kAstarStepCost) {
    Eigen::Vector2d adjusted_start =
        findNearestFreePoint(start, config_.collision_start_search_radius());
    if (adjusted_start == start) {
      AERROR << "No free point found near start within search radius";
      return false;
    }
    AINFO << "Start adjusted from " << start.transpose() << " to "
          << adjusted_start.transpose();
    start = adjusted_start;
  }

  if (!config_.use_smoother()) {
    return planner->plan(start, end, grid_map_, path,
                         config_.max_obstacle_dist() * 0.5,
                         static_cast<int>(config_.heuristic()));
  } else {
    if (!planner->plan(start, end, grid_map_, path,
                       config_.max_obstacle_dist() * 0.5,
                       static_cast<int>(config_.heuristic()))) {
      std::cout << "GridPathPlanner plan path failed." << std::endl;
      return false;
    }
    if (path.size() < 4) {
      return true;
    }
    FemPosDeviationOsqpInterface smoother;
    std::vector<std::pair<double, double>> ref_points;
    for (const auto &point : path) {
      ref_points.push_back({point.x(), point.y()});
    }
    std::vector<double> bounds;
    for (const auto &point : path) {
      double min_obs_dis = grid_map_.atPosition("distance", point);
      double bound_size = (min_obs_dis - config_.smoother_safety_margin()) *
                          config_.smoother_safety_factor();
      bound_size = std::max(
          0.1, std::min(bound_size, config_.smoother_far_away_obstacle_dist()));
      bounds.push_back(bound_size);
    }
    bounds.front() = 0.0;
    bounds.back() = 0.0;
    smoother.set_ref_points(ref_points);
    smoother.set_bounds_around_refs(bounds);
    smoother.set_weight_fem_pos_deviation(
        config_.smoother_weight_fem_pos_deviation());
    smoother.set_weight_path_length(config_.smoother_weight_path_length());
    smoother.set_weight_ref_deviation(config_.smoother_weight_ref_deviation());
    if (smoother.Solve()) {
      const auto &smoothed_x = smoother.opt_x();
      const auto &smoothed_y = smoother.opt_y();

      path.clear();
      for (size_t i = 0; i < smoothed_x.size(); ++i) {
        path.push_back(Eigen::Vector2d(smoothed_x[i], smoothed_y[i]));
      }
      return true;
    } else {
      std::cout << "Path smoothing failed, using original path." << std::endl;
      return true;
    }
  }
}

bool GridPathPlanner::saveCostMap(const std::string &map_path,
                                  const std::string &dist_map_path,
                                  const std::string &yaml_path) {
  double max_dist = grid_map_.get("distance").maxCoeff();
  double min_dist = grid_map_.get("distance").minCoeff();
  auto distance = grid_map_.get("distance");
  auto distance_cost = grid_map_.get("distance_cost");
  distance = (distance.array() - min_dist) / max_dist * 255.0;
  cv::Mat dist_image = eigen2cv(distance);
  distance_cost = distance_cost.array() * 255.0;
  cv::Mat dist_cost_image = eigen2cv(distance_cost);
  cv::flip(dist_cost_image, dist_cost_image, 1);
  if (!cv::imwrite(map_path, dist_cost_image)) {
    AERROR << "Failed to save cost map to pgm file";
    return false;
  }
  cv::flip(dist_image, dist_image, 1);
  if (!cv::imwrite(dist_map_path, dist_image)) {
    AERROR << "Failed to save distance map to pgm file";
    return false;
  }

  YAML::Node config;
  config["resolution"] = grid_map_.getResolution();
  grid_map::Position origin = grid_map_.getPosition();
  grid_map::Length length = grid_map_.getLength();
  config["origin"][0] = origin.x() - length.x() / 2.0;
  config["origin"][1] = origin.y() - length.y() / 2.0;
  config["origin"][2] = 0.0;
  config["max_distance"] = max_dist;
  config["min_distance"] = min_dist;
  std::ofstream fout(yaml_path);
  fout << config;
  fout.close();
  return true;
}

bool GridPathPlanner::loadCostMap(const std::string &map_path,
                                  const std::string &dist_map_path,
                                  const std::string &yaml_path) {
  YAML::Node config = YAML::LoadFile(yaml_path);
  double resolution = config["resolution"].as<double>();
  double origin_x = config["origin"][0].as<double>();
  double origin_y = config["origin"][1].as<double>();
  double origin_z = config["origin"][2].as<double>();
  double max_dist = config["max_distance"].as<double>();
  double min_dist = config["min_distance"].as<double>();

  cv::Mat image = cv::imread(map_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    AERROR << "Failed to load cost map from pgm file";
    return false;
  }
  cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
  cv::Mat dist_image = cv::imread(dist_map_path, cv::IMREAD_GRAYSCALE);
  if (dist_image.empty()) {
    AERROR << "Failed to load distance map from pgm file";
    return false;
  }
  cv::rotate(dist_image, dist_image, cv::ROTATE_90_COUNTERCLOCKWISE);
  const double length_x = resolution * image.rows;
  const double length_y = resolution * image.cols;
  grid_map::Length length(length_x, length_y);
  grid_map::Position position(origin_x + length_x / 2.0,
                              origin_y + length_y / 2.0);
  grid_map_.setGeometry(length, resolution, position);
  grid_map_.setFrameId("map");
  auto &distance_cost = grid_map_.get("distance_cost");
  cv::cv2eigen(image, distance_cost);
  distance_cost = distance_cost.array() / 255.0;
  auto &distance = grid_map_.get("distance");
  cv::cv2eigen(dist_image, distance);
  distance = distance.array() / 255.0 * (max_dist - min_dist) + min_dist;
  auto &obstacle = grid_map_.get("obstacle");
  obstacle.setConstant(255.0);
  obstacle = (distance.array() < 0.01).select(0.0, obstacle);
  return true;
}

bool GridPathPlanner::isCollision(const Eigen::Vector2d &point) {
  if (!grid_map_.isInside(point)) {
    return true;
  }
  return grid_map_.atPosition("distance_cost", point) > 0.8;
}

bool GridPathPlanner::isCollision(const Eigen::Vector3d &pose) {
  auto footprint =
      ConfigHelper::GetBoundingCircle(pose.x(), pose.y(), pose.z());
  grid_map::Position pos(footprint.center_x, footprint.center_y);
  if (grid_map_.isInside(pos)) {
    double clearance = getObstacleDistance(pos);
    if (clearance < footprint.radius) {
      return (isCollisionExact(pose.x(), pose.y(), pose.z()));
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool GridPathPlanner::isCollisionExact(const double &x, const double &y,
                                       const double &theta) {
  auto footprint = ConfigHelper::GetBoundingCircles(x, y, theta);
  for (auto &circle_itr : footprint) {
    grid_map::Position pos(circle_itr.center_x, circle_itr.center_y);
    if (grid_map_.isInside(pos)) {
      double clearance = getObstacleDistance(pos);
      if (clearance < circle_itr.radius) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

double GridPathPlanner::getObstacleDistance(const Eigen::Vector2d &pt) {
  if (grid_map_.isInside(pt)) {
    return grid_map_.atPosition("distance", pt,
                                grid_map::InterpolationMethods::INTER_LINEAR);
  } else {
    return 0.0;
  }
}

void GridPathPlanner::calcCostMap() {
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary =
      grid_map_.get("obstacle").cast<unsigned char>();
  cv::distanceTransform(eigen2cv(binary), eigen2cv(grid_map_.get("distance")),
                        cv::DIST_L2, cv::DIST_MASK_PRECISE);
  grid_map_.get("distance") *= grid_map_.getResolution();
  const double safe_margin = config_.safe_margin();
  const double inflation_margin = config_.inflation_margin();
  const double ref_safe_margin = safe_margin + inflation_margin;
  const auto &dist_matrix = grid_map_.get("distance");
  auto &cost_matrix = grid_map_.get("distance_cost");
  double max_dist = config_.max_obstacle_dist();
  cost_matrix = dist_matrix.cwiseMin(max_dist);
  cost_matrix = cost_matrix.cwiseMax(ref_safe_margin);
  double dist_scale = max_dist - ref_safe_margin;
  cost_matrix = 1.0 - ((cost_matrix.array() - ref_safe_margin) / dist_scale);
}

bool GridPathPlanner::loadFromPGM(const std::string &file_name) {
  std::string pgm_file_name = file_name + ".pgm";
  std::string yaml_file_name = file_name + ".yaml";
  ACHECK(std::filesystem::exists(pgm_file_name))
      << "pgm file not exist, " << pgm_file_name;
  ACHECK(std::filesystem::exists(yaml_file_name))
      << "yaml file not exist, " << yaml_file_name;

  YAML::Node config = YAML::LoadFile(yaml_file_name);
  double resolution = config["resolution"].as<double>();
  double origin_x = config["origin"][0].as<double>();
  double origin_y = config["origin"][1].as<double>();
  double origin_z = config["origin"][2].as<double>();
  double occupied_thresh = config["occupied_thresh"].as<double>() * 255.0;
  double free_thresh = config["free_thresh"].as<double>() * 255.0;
  bool negate = config["negate"].as<int>() != 0;
  bool unknown_as_free = config_.unknown_as_free();

  cv::Mat image = cv::imread(pgm_file_name, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    throw std::runtime_error("Failed to load PGM file: " + pgm_file_name);
  }
  cv::flip(image, image, 1);
  const double length_x = resolution * image.cols;
  const double length_y = resolution * image.rows;
  grid_map::Length length(length_x, length_y);
  grid_map::Position position(origin_x + length_x / 2.0,
                              origin_y + length_y / 2.0);
  grid_map_.setGeometry(length, resolution, position);
  grid_map_.setFrameId("map");
  for (grid_map::GridMapIterator it(grid_map_); !it.isPastEnd(); ++it) {
    const grid_map::Index index(*it);
    double value = image.at<uchar>(index.y(), index.x());
    if (negate) {
      value = 255.0 - value;
    }
    if (unknown_as_free) {
      grid_map_.at("obstacle", index) =
          (value > occupied_thresh) ? 0.0 : 255.0; // 255 is free
    } else {
      grid_map_.at("obstacle", index) = (value < free_thresh) ? 255.0 : 0.0;
    }
  }
  return true;
}

void GridPathPlanner::InitVoronoi() {
  std::string voronoi_file = config_.input_dir() + "voronoi.dat";
  if (std::filesystem::exists(voronoi_file) && LoadVoronoi(voronoi_file)) {
    return;
  }

  unsigned int size_x = grid_map_.getSize()[0];
  unsigned int size_y = grid_map_.getSize()[1];
  voronoi_.initializeEmpty(static_cast<int>(size_x), static_cast<int>(size_y));

  for (int j = 0; j < static_cast<int>(size_y); ++j) {
    for (int i = 0; i < static_cast<int>(size_x); ++i) {
      if (voronoi_.isOccupied(i, j) &&
          grid_map_.at("distance_cost", grid_map::Index(i, j)) < 0.001) {
        voronoi_.clearCell(i, j);
      }
      if (i == 0 || i == (size_x - 1) || j == 0 || j == (size_y - 1)) {
        voronoi_.occupyCell(i, j);
      }
      if (!voronoi_.isOccupied(i, j) &&
          grid_map_.at("distance_cost", grid_map::Index(i, j)) > 0.1) {
        voronoi_.occupyCell(i, j);
      } // grid_map_.atPosition("distance_cost", point) > 0.1;
    }
  }
  voronoi_.update();
  voronoi_.prune();
  voronoi_.visualize();

  SaveVoronoi(voronoi_file);
}

void GridPathPlanner::SaveVoronoi(const std::string &file_path) {
  if (!voronoi_.saveToFile(file_path)) {
    AERROR << "Failed to save Voronoi data to file: " << file_path;
  }
}

bool GridPathPlanner::LoadVoronoi(const std::string &file_path) {
  return voronoi_.loadFromFile(file_path);
}

Eigen::Vector2d
GridPathPlanner::findNearestFreePoint(const Eigen::Vector2d &end) {
  return findNearestFreePoint(end, config_.collision_goal_search_radius());
}

Eigen::Vector2d
GridPathPlanner::findNearestFreePoint(const Eigen::Vector2d &end,
                                      double max_radius) {
  double res = grid_map_.getResolution();
  int max_rings = static_cast<int>(max_radius / res);
  for (int ring = 1; ring <= max_rings; ++ring) {
    int num_samples = static_cast<int>(2 * M_PI * ring);
    num_samples = std::max(num_samples, 8);
    for (int i = 0; i < num_samples; ++i) {
      double angle = 2.0 * M_PI * i / num_samples;
      Eigen::Vector2d candidate(end.x() + ring * res * std::cos(angle),
                                end.y() + ring * res * std::sin(angle));
      if (!isCollision(candidate)) {
        return candidate;
      }
    }
  }
  return end;
}

} // namespace algorithm
} // namespace planning
