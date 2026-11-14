#include "src/runnable/preprocess/global_map_manager/global_map_manager.h"
#include "src/common/util/config_helper.h"
#include "src/common/util/time_keeper.h"
#include "src/runnable/entry/planning_entry.h"
#include <filesystem>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

using planning::common::util::ConfigHelper;
using planning::common::util::TimeKeeper;

namespace planning {
namespace preprocess {

GlobalMapManager::GlobalMapManager(const PreprocessConf::Conf &conf)
    : config_(conf.global_map_manager_conf()),
      global_map_(std::vector<std::string>{"obstacle", "pgm_obstacle"}),
      cell_timeout_seconds_(
          conf.global_map_manager_conf().cell_timeout_seconds()) {
  // 全局规划使用obstacle层，会叠加局部感知到的障碍物
  // 局部规划使用pgm_obstacle层，原始障碍物地图
  conf_ = conf;
}

GlobalMapManager::~GlobalMapManager() {}

bool GlobalMapManager::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  std::string map_file_path = ConfigHelper::Instance()->GetConfDir() +
                              config_.input_dir() + config_.input_name();
  std::string pgm_file_path = map_file_path + ".pgm";
  std::string yaml_file_path = map_file_path + ".yaml";
  map_loaded_ = LoadPGMMap(pgm_file_path, yaml_file_path);
  if (!map_loaded_) {
    AERROR << "GlobalMapManager: Failed to load PGM map from " << map_file_path;
    return false;
  }
  if (!config_.has_grid_path_planner_conf()) {
    AWARN << "GlobalMapManager: grid_path_planner_conf not configured, "
             "PlanGlobalPath() will fail";
  } else {
    grid_path_planner_ =
        std::make_shared<GridPathPlanner>(config_.grid_path_planner_conf());
  }
  // ours (T53 review): the site map with its crosswalks from the start, so the dog does not depend on the App's
  // /map/switch (a later one still replaces it). On failure the PGM above stays.
  if (!config_.startup_map_dir().empty()) {
    std::string dir = config_.startup_map_dir();
    if (dir[0] != '/') dir = ConfigHelper::Instance()->GetConfDir() + dir;
    if (ReloadMap(dir)) {
      AINFO << "GlobalMapManager: startup map " << dir << ", " << crosswalks_.size() << " crosswalks";
    } else {
      AERROR << "GlobalMapManager: startup map " << dir << " failed, keeping " << map_file_path;
      map_loaded_ = LoadPGMMap(pgm_file_path, yaml_file_path);
      crosswalks_.clear();   // nothing half-read from that map's OSM
    }
  }
  BuildOverview();
  return true;
}

void GlobalMapManager::BuildOverview() {
  if (!map_loaded_ || !global_map_.exists("obstacle")) {
    overview_ = MapOverview();
    return;
  }
  const grid_map::Position max_corner = global_map_.getPosition() + 0.5 * global_map_.getLength().matrix();
  overview_ = BuildMapOverview(global_map_.get("obstacle"), global_map_.getResolution(), max_corner.x(),
                               max_corner.y(), 4000000);
  AINFO << "GlobalMapManager: overview " << overview_.width << "x" << overview_.height << " at "
        << overview_.resolution << " m";
}

bool GlobalMapManager::Proc() {
  if (!map_loaded_) {
    AWARN << "GlobalMapManager: PGM map not loaded";
    return false;
  }
  // auto is_standstill = [&]() -> bool {
  //   auto odom =
  //       planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  //   return odom->twist.linear.norm() < 0.1 && odom->twist.angular.norm() <
  //   0.1;
  // };
  if (config_.enable_overlay()) {
    AddObstacleFromPerception();
    ResetStaleCells();
  }
  return true;
}

bool GlobalMapManager::UpdateMap(const grid_map::GridMap &map) {
  if (!map.exists("obstacle")) {
    AERROR << "GlobalMapManager: Map has no obstacle layer";
    return false;
  }
  global_map_ = map;
  global_map_.add("pgm_obstacle", map["obstacle"]);
  map_dirty_ = true;
  return true;
}

bool GlobalMapManager::PlanGlobalPath(const Eigen::Vector2d &start,
                                      const Eigen::Vector2d &end,
                                      std::optional<double> end_theta) {
  if (grid_path_planner_ == nullptr) {
    AERROR << "GlobalMapManager: grid_path_planner_ is not initialized";
    global_plan_flag_ = false;
    return false;
  }
  if (map_dirty_) {
    grid_path_planner_->SetGridMap(global_map_);
    map_dirty_ = false;
  }

  std::vector<Eigen::Vector2d> path_pts;
  global_plan_flag_ = grid_path_planner_->plan(start, end, path_pts);
  if (global_plan_flag_) {
    while (path_pts.size() < 2) {
      path_pts.emplace_back(end);
    }
    global_path_ =
        Path2d(std::vector<Point2d>(path_pts.begin(), path_pts.end()));
    if (end_theta.has_value()) {
      global_path_.poses_ptr()->back().theta = *end_theta;
    }
  }
  return global_plan_flag_;
}

void GlobalMapManager::ClearGlobalPath() {
  global_path_.clear();
  global_plan_flag_ = false;
}

std::vector<Pose2d>
GlobalMapManager::ExtractLocalPath(const Eigen::Vector2d &robot_pos,
                                   double ahead_s) const {
  return global_path_.pose_subset(robot_pos, ahead_s);
}

void GlobalMapManager::AddObstacleFromPerception() {
  if (!map_loaded_) {
    AERROR << "GlobalMapManager: PGM map not loaded";
    return;
  }

  const auto &local_map = planning_entry_->communication_entry_ptr()
                              ->MainBridge()
                              ->occupied_map_2d();
  if (!local_map->exists("obstacle")) {
    return;
  }
  auto odom =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  grid_map::Position global_position;
  global_position.x() = odom->pose.position.x();
  global_position.y() = odom->pose.position.y();
  double yaw = odom->pose.euler()[2];
  const double cos_yaw = cos(yaw);
  const double sin_yaw = sin(yaw);
  double time_now = TimeKeeper::TimeNowSec();
  for (grid_map::GridMapIterator iterator(*local_map); !iterator.isPastEnd();
       ++iterator) {
    grid_map::Position local_position;
    local_map->getPosition(*iterator, local_position);
    grid_map::Position global_cell_position;
    global_cell_position.x() = global_position.x() +
                               local_position.x() * cos_yaw -
                               local_position.y() * sin_yaw;
    global_cell_position.y() = global_position.y() +
                               local_position.x() * sin_yaw +
                               local_position.y() * cos_yaw;
    if (!global_map_.isInside(global_cell_position)) {
      continue;
    }
    if (local_map->at("obstacle", *iterator) < 0.01) {
      grid_map::Index global_idx;
      global_map_.getIndex(global_cell_position, global_idx);
      std::string key =
          std::to_string(global_idx(0)) + "_" + std::to_string(global_idx(1));
      updated_cells_[key] = {time_now, global_idx};
      global_map_.atPosition("obstacle", global_cell_position) = 0;
      map_dirty_ = true;
    }
  }
}

void GlobalMapManager::ResetStaleCells() {
  double current_time = TimeKeeper::TimeNowSec();
  std::vector<std::string> keys_to_remove;

  for (const auto &cell_pair : updated_cells_) {
    if (current_time - cell_pair.second.last_update_time >
        cell_timeout_seconds_) {
      grid_map::Index idx = cell_pair.second.index;
      global_map_.at("obstacle", idx) = global_map_.at("pgm_obstacle", idx);
      map_dirty_ = true;
      keys_to_remove.push_back(cell_pair.first);
    }
  }

  for (const auto &key : keys_to_remove) {
    updated_cells_.erase(key);
  }
}

bool GlobalMapManager::LoadPGMMap(const std::string &pgm_file_path,
                                  const std::string &yaml_file_path) {
  if (!std::filesystem::exists(pgm_file_path)) {
    AERROR << "PGM file not exist: " << pgm_file_path;
    return false;
  }
  if (!std::filesystem::exists(yaml_file_path)) {
    AERROR << "YAML file not exist: " << yaml_file_path;
    return false;
  }

  YAML::Node config = YAML::LoadFile(yaml_file_path);
  double resolution = config["resolution"].as<double>();
  double origin_x = config["origin"][0].as<double>();
  double origin_y = config["origin"][1].as<double>();
  double occupied_thresh = config["occupied_thresh"].as<double>() * 255.0;
  double free_thresh = config["free_thresh"].as<double>() * 255.0;
  bool negate = config["negate"].as<int>() == 0;

  cv::Mat image = cv::imread(pgm_file_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    AERROR << "Failed to load PGM file: " << pgm_file_path;
    return false;
  }
  const double length_x = resolution * image.cols;
  const double length_y = resolution * image.rows;
  grid_map::Length length(length_x, length_y);
  grid_map::Position position(origin_x + length_x / 2.0,
                              origin_y + length_y / 2.0);
  global_map_.setGeometry(length, resolution, position);
  global_map_.setFrameId("map");
  global_map_.setBasicLayers(global_map_.getLayers());

  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      double value = image.at<uchar>(r, c);
      if (negate) {
        value = 255.0 - value;
      }
      grid_map::Position world_pos(origin_x + (c + 0.5) * resolution,
                                   origin_y +
                                       (image.rows - 1 - r + 0.5) * resolution);
      grid_map::Index idx;
      global_map_.getIndex(world_pos, idx);
      double cell_value = (value < free_thresh) ? 255.0 : 0.0;
      global_map_.at("obstacle", idx) = cell_value;
      global_map_.at("pgm_obstacle", idx) = cell_value;
    }
  }

  AINFO << "GlobalMapManager: PGM map loaded. Size: "
        << global_map_.getSize()(0) << "x" << global_map_.getSize()(1)
        << ", Resolution: " << resolution;
  return true;
}

bool GlobalMapManager::ReloadMap(const std::string &map_dir) {
  // Load global map yaml file
  std::string global_map_yaml = map_dir + "/global_map.yaml";
  if (!std::filesystem::exists(global_map_yaml)) {
    AERROR << "Global map yaml file not exist: " << global_map_yaml;
    return false;
  }
  YAML::Node root = YAML::LoadFile(global_map_yaml);
  std::string map_image_file;
  std::string map_yaml_file;
  std::string map_osm_file;
  for (const auto &floor_node : root["floors"]) {
    int id = floor_node["floor_id"].as<int>();
    if (id != 1) {
      continue;
    }
    double representative_z = floor_node["representative_z"].as<double>();
    map_image_file =
        map_dir + "/" + floor_node["map_image_file"].as<std::string>();
    map_yaml_file =
        map_dir + "/" + floor_node["map_yaml_file"].as<std::string>();
    if (floor_node["map_osm_file"]) {
      map_osm_file =
          map_dir + "/" + floor_node["map_osm_file"].as<std::string>();
    } else {
      map_osm_file =
          (std::filesystem::path(map_yaml_file).parent_path() / "map.osm")
              .string();
      // the site maps of 2026-11 (ad68517c, a70c6db6) name it semantic.osm
      const auto semantic_osm =
          std::filesystem::path(map_yaml_file).parent_path() / "semantic.osm";
      if (!std::filesystem::exists(map_osm_file) &&
          std::filesystem::exists(semantic_osm)) {
        map_osm_file = semantic_osm.string();
      }
    }
  }
  // Load PGM map
  if (!LoadPGMMap(map_image_file, map_yaml_file)) {
    return false;
  }
  if (!AddObstacleFromOSM(map_osm_file)) {
    return false;
  }
  map_dirty_ = true;
  BuildOverview();
  return true;
}

#if 0
bool GlobalMapManager::AddObstacleFromOSM(const std::string &osm_file_path) {
  if (!std::filesystem::exists(osm_file_path)) {
    AERROR << "OSM file not exist: " << osm_file_path;
    return false;
  }
  if (!osm_parser_.parse(osm_file_path)) {
    AERROR << "Failed to parse OSM file: " << osm_file_path;
    return false;
  }

  for (const auto &semantic : osm_parser_.getMap().keyPoints) {
    if (semantic.type == lanelet::SemanticType::ProhibitedArea) {
      grid_map::Polygon polygon;
      if (semantic.points.size() < 3) {
        continue;
      }
      bool polygon_inside = true;
      for (const auto &point : semantic.points) {
        grid_map::Position p(point.x, point.y);
        if (!global_map_.isInside(p)) {
          AWARN << " point out of map: " << p.x() << ", " << p.y();
          polygon_inside = false;
          break;
        }
        polygon.addVertex(p);
      }
      if (!polygon_inside) {
        continue;
      }
      for (grid_map::PolygonIterator iterator(global_map_, polygon);
           !iterator.isPastEnd(); ++iterator) {
        global_map_.at("pgm_obstacle", *iterator) = 0.0;
        global_map_.at("obstacle", *iterator) = 0.0;
      }
    } else if (semantic.type == lanelet::SemanticType::Fence ||
               semantic.type == lanelet::SemanticType::VirtualLine) {
      if (semantic.points.size() < 2) {
        continue;
      }
      for (int i = 1; i < semantic.points.size(); ++i) {
        grid_map::Position p1(semantic.points[i - 1].x,
                              semantic.points[i - 1].y);
        grid_map::Position p2(semantic.points[i].x, semantic.points[i].y);
        if (!global_map_.isInside(p1)) {
          AWARN << " point out of map: " << p1.x() << ", " << p1.y();
          continue;
        }
        if (!global_map_.isInside(p2)) {
          AWARN << " point out of map: " << p2.x() << ", " << p2.y();
          continue;
        }
        for (grid_map::LineIterator iterator(global_map_, p1, p2);
             !iterator.isPastEnd(); ++iterator) {
          global_map_.at("pgm_obstacle", *iterator) = 0.0;
          global_map_.at("obstacle", *iterator) = 0.0;
        }
      }
    }
  }
  return true;
}
#else
bool GlobalMapManager::AddObstacleFromOSM(const std::string &osm_file_path) {
  crosswalks_.clear();
  if (!std::filesystem::exists(osm_file_path)) {
    AERROR << "OSM file not exist: " << osm_file_path;
    return false;
  }
  if (!osm_parser_.parse(osm_file_path)) {
    AERROR << "Failed to parse OSM file: " << osm_file_path;
    return false;
  }
  for (const auto &semantic : osm_parser_.getMap().keyPoints) {
    if (semantic.type == lanelet::SemanticType::Crosswalk &&
        semantic.points.size() >= 3) {
      std::vector<Eigen::Vector2d> polygon;
      for (const auto &point : semantic.points) {
        polygon.emplace_back(point.x, point.y);
      }
      crosswalks_.push_back(std::move(polygon));
    }
  }
  AINFO << "GlobalMapManager: " << crosswalks_.size() << " crosswalks in "
        << osm_file_path;
  bool has_fence = std::any_of(
      osm_parser_.getMap().keyPoints.begin(),
      osm_parser_.getMap().keyPoints.end(), [](const auto &semantic) {
        return semantic.type == lanelet::SemanticType::Fence;
      });
  if (has_fence) {
    global_map_["pgm_obstacle"].setConstant(0.0);
    global_map_["obstacle"].setConstant(0.0);
  }

  for (const auto &semantic : osm_parser_.getMap().keyPoints) {
    if (semantic.type == lanelet::SemanticType::ProhibitedArea) {
      grid_map::Polygon polygon;
      if (semantic.points.size() < 3) {
        continue;
      }
      bool polygon_inside = true;
      for (const auto &point : semantic.points) {
        grid_map::Position p(point.x, point.y);
        if (!global_map_.isInside(p)) {
          AWARN << " point out of map: " << p.x() << ", " << p.y();
          polygon_inside = false;
          break;
        }
        polygon.addVertex(p);
      }
      if (!polygon_inside) {
        continue;
      }
      for (grid_map::PolygonIterator iterator(global_map_, polygon);
           !iterator.isPastEnd(); ++iterator) {
        global_map_.at("pgm_obstacle", *iterator) = 0.0;
        global_map_.at("obstacle", *iterator) = 0.0;
      }
    } else if (semantic.type == lanelet::SemanticType::VirtualLine) {
      if (semantic.points.size() < 2) {
        continue;
      }
      for (int i = 1; i < semantic.points.size(); ++i) {
        grid_map::Position p1(semantic.points[i - 1].x,
                              semantic.points[i - 1].y);
        grid_map::Position p2(semantic.points[i].x, semantic.points[i].y);
        if (!global_map_.isInside(p1)) {
          AWARN << " point out of map: " << p1.x() << ", " << p1.y();
          continue;
        }
        if (!global_map_.isInside(p2)) {
          AWARN << " point out of map: " << p2.x() << ", " << p2.y();
          continue;
        }
        for (grid_map::LineIterator iterator(global_map_, p1, p2);
             !iterator.isPastEnd(); ++iterator) {
          global_map_.at("pgm_obstacle", *iterator) = 0.0;
          global_map_.at("obstacle", *iterator) = 0.0;
        }
      }
    } else if (semantic.type == lanelet::SemanticType::Fence) {
      grid_map::Polygon polygon;
      if (semantic.points.size() < 3) {
        continue;
      }
      bool polygon_inside = true;
      for (const auto &point : semantic.points) {
        grid_map::Position p(point.x, point.y);
        if (!global_map_.isInside(p)) {
          AWARN << " point out of map: " << p.x() << ", " << p.y();
          polygon_inside = false;
          break;
        }
        polygon.addVertex(p);
      }
      if (!polygon_inside) {
        continue;
      }
      for (grid_map::PolygonIterator iterator(global_map_, polygon);
           !iterator.isPastEnd(); ++iterator) {
        global_map_.at("pgm_obstacle", *iterator) = 255.0;
        global_map_.at("obstacle", *iterator) = 255.0;
      }
    }
  }
  return true;
}
#endif

} // namespace preprocess
} // namespace planning
