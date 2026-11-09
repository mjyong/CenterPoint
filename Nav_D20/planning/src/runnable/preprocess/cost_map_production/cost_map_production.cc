#include "src/runnable/preprocess/cost_map_production/cost_map_production.h"
#include "src/common/util/config_helper.h"
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include "src/common/util/math.h"
#include "src/common/util/tf_buffer.h"
#include "src/common/util/time_keeper.h"
#include "src/runnable/entry/planning_entry.h"

using planning::common::util::ConfigHelper;
using planning::common::util::normalize_angle;
using planning::common::util::TfBuffer;
using planning::common::util::TimeKeeper;

namespace planning {
namespace preprocess {

CostMapProduction::CostMapProduction(const PreprocessConf::Conf &conf)
    : config_(conf.cost_map_production_conf()),
      grid_map_(std::vector<std::string>{
          "obstacle",
          "obstacle_distance",
          "reference_path",
          "reference_offset_dist",
          "composite_obstacle",
          "composite_cost",
          "composite_gradient_x",
          "composite_gradient_y",
      }) {
  conf_ = conf;
  layers_[Layer::OBSTACLE] = "obstacle";
  layers_[Layer::OBSTACLE_DISTANCE] = "obstacle_distance";
  layers_[Layer::REFERENCE_PATH] = "reference_path";
  layers_[Layer::REFERENCE_OFFSET_DIST] = "reference_offset_dist";
  layers_[Layer::COMPOSITE_OBSTACLE] = "composite_obstacle";
  layers_[Layer::COMPOSITE_COST] = "composite_cost";
  layers_[Layer::COMPOSITE_GRADIENT_X] = "composite_gradient_x";
  layers_[Layer::COMPOSITE_GRADIENT_Y] = "composite_gradient_y";

  double length =
      config_.front_to_base_length() + config_.back_to_base_length();
  double width = config_.left_to_base_width() + config_.right_to_base_width();
  grid_map::Position position(length / 2.0 - config_.back_to_base_length(),
                              width / 2.0 - config_.right_to_base_width());

  grid_map_.setGeometry(grid_map::Length(length, width), config_.resolution(),
                        position);
  grid_map_.setBasicLayers(grid_map_.getLayers());
  grid_map_.setFrameId("base_link");

  AINFO << "\nGrid_map init, length: " << grid_map_.getLength().x()
        << ", width: " << grid_map_.getLength().y()
        << ", cells: " << grid_map_.getSize()(0) << " x "
        << grid_map_.getSize()(1) << " \nThe center of the map is located at: "
        << grid_map_.getPosition().x() << ", " << grid_map_.getPosition().y();
}

CostMapProduction::~CostMapProduction() {
  if (use_cuda_) {
    cuda_pipeline_.Release();
  }
}

bool CostMapProduction::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  use_cuda_ = config_.enable_cuda();
  if (use_cuda_) {
    if (!cuda_pipeline_.Init(grid_map_.getSize()(0), grid_map_.getSize()(1),
                             config_.resolution(), grid_map_.getPosition().x(),
                             grid_map_.getPosition().y(),
                             grid_map_.getLength().x(),
                             grid_map_.getLength().y())) {
      AWARN << "CUDA pipeline init failed, fallback to CPU";
      use_cuda_ = false;
    }
  }
  return true;
};

bool CostMapProduction::Proc() {
  TimeKeeper time_keeper("CostMapProduction::Proc");
  time_keeper.RecordTime("AddObstacleFromMap");
  AddObstacleFromMap();
  time_keeper.RecordTime("AddObstacleFromPerception");
  AddObstacleFromPerception();
  time_keeper.RecordTime("UpdateObstacleDistance");
  UpdateObstacleDistance();

  time_keeper.RecordTime("ExtractLocalReferencePath");
  ExtractLocalReferencePath();
  time_keeper.RecordTime("UpdateReferenceOffsetDist");
  UpdateReferenceOffsetDist();

  time_keeper.RecordTime("BuildCompositeObstacle");
  BuildCompositeObstacle();
  time_keeper.RecordTime("UpdateCompositeCost");
  UpdateCompositeCost();

  time_keeper.RecordTime("END");
  ADEBUG << time_keeper.DumpString();
  grid_map_.setTimestamp(TimeKeeper::TimeNowSec() * 1e9);

  return true;
}

void CostMapProduction::SetGlobalReferencePath(const Path2d &global_ref_path) {
  global_ref_path_ = global_ref_path;
}

void CostMapProduction::ExtractLocalReferencePath() {
  local_ref_path_.clear();
  if (global_ref_path_.poses().empty()) {
    return;
  }

  auto main_bridge = planning_entry_->communication_entry_ptr()->MainBridge();
  const auto &odom = main_bridge->odometry();
  Pose2d start_pose;
  start_pose.position.x() = odom->pose.position.x();
  start_pose.position.y() = odom->pose.position.y();
  start_pose.theta = odom->pose.euler().z();

  auto local_poses = global_ref_path_.pose_subset(
      start_pose.position, config_.reference_path_length());

  Transform3d tf;
  if (!TfBuffer::Instance()->get("map", "base_link", tf)) {
    AWARN << "Failed to get map->base_link transform";
    return;
  }

  for (const auto &pose : local_poses) {
    auto base_pt =
        tf.inverse() * Eigen::Vector3d(pose.position.x(), pose.position.y(), 0);
    auto base_theta = normalize_angle(pose.theta - odom->pose.euler().z());
    local_ref_path_.emplace_back(
        Eigen::Vector3d(base_pt.x(), base_pt.y(), base_theta));
  }
}

void CostMapProduction::UpdateObstacleDistance() {
  if (use_cuda_) {
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary =
        grid_map_.get(layers_[Layer::OBSTACLE]).cast<unsigned char>();
    int rows = binary.rows();
    int cols = binary.cols();
    cuda_pipeline_.ProcObstacle(
        binary.data(), rows, cols, config_.safe_margin(),
        config_.inflation_margin(), config_.max_obstacle_dist(),
        config_.resolution());
    cuda_pipeline_.DownloadDistanceField(
        grid_map_.get(layers_[Layer::OBSTACLE_DISTANCE]).data(), rows * cols);
    return;
  }

  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary =
      grid_map_.get(layers_[Layer::OBSTACLE]).cast<unsigned char>();
  cv::distanceTransform(
      eigen2cv(binary),
      eigen2cv(grid_map_.get(layers_[Layer::OBSTACLE_DISTANCE])), cv::DIST_L2,
      cv::DIST_MASK_PRECISE);
  grid_map_.get(layers_[Layer::OBSTACLE_DISTANCE]) *= config_.resolution();
}

void CostMapProduction::BuildCompositeObstacle() {
  if (use_cuda_) {
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> obstacle =
        grid_map_.get(layers_[Layer::OBSTACLE]).cast<unsigned char>();
    int rows = obstacle.rows();
    int cols = obstacle.cols();
    const auto &obstacle_distance =
        grid_map_.get(layers_[Layer::OBSTACLE_DISTANCE]);
    const auto &ref_offset =
        grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST]);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>
        composite_obstacle(rows, cols);
    if (cuda_pipeline_.ProcCompositeObstacle(
            obstacle.data(), obstacle_distance.data(), ref_offset.data(),
            composite_obstacle.data(), rows, cols,
            config_.safe_margin() + config_.inflation_margin(),
            config_.max_reference_offset(), (local_ref_path_.size() > 1))) {
      grid_map_.get(layers_[Layer::COMPOSITE_OBSTACLE]) =
          composite_obstacle.cast<float>();
      return;
    }
    AWARN << "CUDA ProcCompositeObstacle failed, fallback to CPU";
  }

  auto &composite_obstacle_layer =
      grid_map_.get(layers_[Layer::COMPOSITE_OBSTACLE]);
  composite_obstacle_layer = grid_map_.get(layers_[Layer::OBSTACLE]);
  const double safe_margin = config_.safe_margin();
  const double inflation_margin = config_.inflation_margin();
  const double inflation_safe_margin = safe_margin + inflation_margin;
  const auto &obstacle_distance =
      grid_map_.get(layers_[Layer::OBSTACLE_DISTANCE]);
  composite_obstacle_layer =
      (composite_obstacle_layer.array() < 0.5 ||
       obstacle_distance.array() <= inflation_safe_margin)
          .select(0.0f, composite_obstacle_layer);
  if (!local_ref_path_.empty()) {
    const auto &ref_offset =
        grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST]);
    composite_obstacle_layer =
        (composite_obstacle_layer.array() < 0.5 ||
         ref_offset.array() >= config_.max_reference_offset())
            .select(0.0f, composite_obstacle_layer);
  }
}

void CostMapProduction::UpdateReferenceOffsetDist() {
  if (local_ref_path_.empty()) {
    grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST])
        .setConstant(config_.max_reference_offset());
    return;
  }

  auto &path_layer = grid_map_.get(layers_[Layer::REFERENCE_PATH]);
  path_layer.setConstant(255.0);
  for (int i = 0; i < static_cast<int>(local_ref_path_.size()) - 1; ++i) {
    grid_map::Position start(local_ref_path_[i].x(), local_ref_path_[i].y());
    grid_map::Position end(local_ref_path_[i + 1].x(),
                           local_ref_path_[i + 1].y());
    if (!grid_map_.isInside(start) && !grid_map_.isInside(end)) {
      continue;
    }
    try {
      for (grid_map::LineIterator iterator(grid_map_, start, end);
           !iterator.isPastEnd(); ++iterator) {
        grid_map_.at(layers_[Layer::REFERENCE_PATH], *iterator) = 0.0;
      }
    } catch (const std::exception &e) {
      continue;
    }
  }

  if (use_cuda_) {
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary =
        grid_map_.get(layers_[Layer::REFERENCE_PATH]).cast<unsigned char>();
    int rows = binary.rows();
    int cols = binary.cols();

    if (cuda_pipeline_.ProcReference(binary.data(), rows, cols,
                                     config_.max_reference_offset(),
                                     config_.resolution())) {
      const int size = rows * cols;
      cuda_pipeline_.DownloadRefOffsetDistField(
          grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST]).data(), size);
      return;
    }
  }

  cv::distanceTransform(
      eigen2cv(path_layer.cast<unsigned char>()),
      eigen2cv(grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST])),
      cv::DIST_L2, cv::DIST_MASK_PRECISE);
  grid_map_.get(layers_[Layer::REFERENCE_OFFSET_DIST]) *= config_.resolution();
}

void CostMapProduction::UpdateCompositeCost() {
  if (use_cuda_) {
    int rows = grid_map_.getSize()(0);
    int cols = grid_map_.getSize()(1);
    cuda_pipeline_.ProcComposite(nullptr, rows, cols);
    cuda_pipeline_.DownloadCompositeCostField(
        grid_map_.get(layers_[Layer::COMPOSITE_COST]).data(), rows * cols);
    cuda_pipeline_.DownloadGradientXField(
        grid_map_.get(layers_[Layer::COMPOSITE_GRADIENT_X]).data(),
        rows * cols);
    cuda_pipeline_.DownloadGradientYField(
        grid_map_.get(layers_[Layer::COMPOSITE_GRADIENT_Y]).data(),
        rows * cols);
    return;
  }
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> binary =
      grid_map_.get(layers_[Layer::COMPOSITE_OBSTACLE]).cast<unsigned char>();
  cv::distanceTransform(eigen2cv(binary),
                        eigen2cv(grid_map_.get(layers_[Layer::COMPOSITE_COST])),
                        cv::DIST_L2, cv::DIST_MASK_PRECISE);
  auto &cost_matrix = grid_map_.get(layers_[Layer::COMPOSITE_COST]);
  double max_dist = cost_matrix.maxCoeff();
  cost_matrix = 1.0 - ((cost_matrix.array()) / (max_dist));

  auto &composite_gx = grid_map_.get(layers_[Layer::COMPOSITE_GRADIENT_X]);
  auto &composite_gy = grid_map_.get(layers_[Layer::COMPOSITE_GRADIENT_Y]);
  cv::Sobel(eigen2cv(cost_matrix), eigen2cv(composite_gx), -1, 1, 0, 3);
  cv::Sobel(eigen2cv(cost_matrix), eigen2cv(composite_gy), -1, 0, 1, 3);
  auto mag =
      (composite_gx.array().square() + composite_gy.array().square()).sqrt();
  composite_gx = composite_gx.array() / (mag + 1e-6f);
  composite_gy = composite_gy.array() / (mag + 1e-6f);
}

double CostMapProduction::GetCompositeCost(const double relative_x,
                                           const double relative_y) {
  grid_map::Position pos(relative_x, relative_y);
  if (grid_map_.isInside(pos)) {
    return grid_map_.atPosition(layers_[Layer::COMPOSITE_COST], pos,
                                grid_map::InterpolationMethods::INTER_LINEAR);
  }
  return 1.0;
}

void CostMapProduction::GetCompositeGradient(const double relative_x,
                                             const double relative_y,
                                             double &grad_x, double &grad_y) {
  grid_map::Position pos(relative_x, relative_y);
  if (grid_map_.isInside(pos)) {
    grad_x = grid_map_.atPosition(layers_[Layer::COMPOSITE_GRADIENT_X], pos,
                                  grid_map::InterpolationMethods::INTER_LINEAR);
    grad_y = grid_map_.atPosition(layers_[Layer::COMPOSITE_GRADIENT_Y], pos,
                                  grid_map::InterpolationMethods::INTER_LINEAR);
  } else {
    grad_x = 0.0;
    grad_y = 0.0;
  }
}

bool CostMapProduction::IsCollision(const double &relative_x,
                                    const double &relative_y,
                                    const double &relative_theta) {
  auto footprint =
      ConfigHelper::GetBoundingCircle(relative_x, relative_y, relative_theta);
  grid_map::Position pos(footprint.center_x, footprint.center_y);
  if (grid_map_.isInside(pos)) {
    double clearance = GetObstacleDistance(pos);
    if (clearance < footprint.radius) {
      return (IsCollisionExact(relative_x, relative_y, relative_theta));
    } else {
      return false;
    }
  } else {
    return false;
  }
}

double CostMapProduction::GetObstacleDistance(const double relative_x,
                                              const double relative_y) {
  grid_map::Position pos(relative_x, relative_y);
  return GetObstacleDistance(pos);
}

double CostMapProduction::GetObstacleDistance(const grid_map::Position &pos) {
  if (grid_map_.isInside(pos)) {
    return grid_map_.atPosition(layers_[Layer::OBSTACLE_DISTANCE], pos,
                                grid_map::InterpolationMethods::INTER_LINEAR);
  } else {
    return 1e6;
  }
}
bool CostMapProduction::IsCollisionExact(const double &relative_x,
                                         const double &relative_y,
                                         const double &relative_theta) {
  auto footprint =
      ConfigHelper::GetBoundingCircles(relative_x, relative_y, relative_theta);
  for (auto &circle_itr : footprint) {
    grid_map::Position pos(circle_itr.center_x, circle_itr.center_y);
    if (grid_map_.isInside(pos)) {
      double clearance = GetObstacleDistance(pos);
      if (clearance < circle_itr.radius) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

void CostMapProduction::AddObstacleFromMap() {
  auto &global_map = planning_entry_->preprocess_entry_ptr()
                         ->global_map_manager()
                         ->global_map();
  auto odom =
      planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
  const double odom_x = odom->pose.position.x();
  const double odom_y = odom->pose.position.y();
  const double yaw = odom->pose.euler()[2];
  const double cos_yaw = cos(yaw);
  const double sin_yaw = sin(yaw);

  const auto local_size = grid_map_.getSize();
  const double local_res = grid_map_.getResolution();
  const auto local_pos = grid_map_.getPosition();
  const auto local_len = grid_map_.getLength();
  const auto local_start = grid_map_.getStartIndex();
  const double local_origin_x =
      local_pos.x() + local_len.x() * 0.5 - local_res * 0.5;
  const double local_origin_y =
      local_pos.y() + local_len.y() * 0.5 - local_res * 0.5;

  const double global_res = global_map.getResolution();
  const auto global_size = global_map.getSize();
  const auto global_pos = global_map.getPosition();
  const auto global_len = global_map.getLength();
  const auto global_start = global_map.getStartIndex();
  const double global_origin_x = global_pos.x() + global_len.x() * 0.5;
  const double global_origin_y = global_pos.y() + global_len.y() * 0.5;

  const double scale = local_res / global_res;
  const double A = cos_yaw * scale;
  const double B = -sin_yaw * scale;
  const double C = (global_origin_x - odom_x - cos_yaw * local_origin_x +
                    sin_yaw * local_origin_y) /
                   global_res;
  const double D = sin_yaw * scale;
  const double E = cos_yaw * scale;
  const double F = (global_origin_y - odom_y - sin_yaw * local_origin_x -
                    cos_yaw * local_origin_y) /
                   global_res;

  const std::string layer_name = layers_[Layer::OBSTACLE];
  const std::string pgm_layer = "pgm_obstacle";
  auto &obstacle_data = grid_map_[layer_name];
  obstacle_data.setConstant(0.0);

  const int gsx = static_cast<int>(global_size(0));
  const int gsy = static_cast<int>(global_size(1));
  const int lsx = static_cast<int>(local_size(0));
  const int lsy = static_cast<int>(local_size(1));
  const int global_sx = global_start(0);
  const int global_sy = global_start(1);
  const int local_sx = local_start(0);
  const int local_sy = local_start(1);

  for (int i = 0; i < lsx; ++i) {
    for (int j = 0; j < lsy; ++j) {
      int gi_raw = static_cast<int>(A * i + B * j + C);
      int gj_raw = static_cast<int>(D * i + E * j + F);

      if (gi_raw >= 0 && gi_raw < gsx && gj_raw >= 0 && gj_raw < gsy) {
        int gi = (gi_raw + global_sx) % gsx;
        if (gi < 0)
          gi += gsx;
        int gj = (gj_raw + global_sy) % gsy;
        if (gj < 0)
          gj += gsy;
        int li = (i + local_sx) % lsx;
        if (li < 0)
          li += lsx;
        int lj = (j + local_sy) % lsy;
        if (lj < 0)
          lj += lsy;
        obstacle_data(li, lj) =
            global_map.at(pgm_layer, grid_map::Index(gi, gj));
      }
    }
  }
}

void CostMapProduction::AddObstacleFromPerception() {
  const auto &occupied_map_2d = planning_entry_->communication_entry_ptr()
                                    ->MainBridge()
                                    ->occupied_map_2d();
  if (!occupied_map_2d->exists("obstacle")) {
    return;
  }
  const auto occ_size = occupied_map_2d->getSize();
  const double occ_res_x = occupied_map_2d->getResolution();
  const double occ_res_y = occ_res_x;
  const auto occ_pos = occupied_map_2d->getPosition();
  const auto occ_len = occupied_map_2d->getLength();
  const auto occ_start = occupied_map_2d->getStartIndex();

  const auto grid_size = grid_map_.getSize();
  const double grid_res = grid_map_.getResolution();
  const auto grid_pos = grid_map_.getPosition();
  const auto grid_len = grid_map_.getLength();
  const auto grid_start = grid_map_.getStartIndex();

  const double C_i = (grid_pos.x() + grid_len.x() * 0.5 - occ_pos.x() -
                      occ_len.x() * 0.5 + occ_res_x * 0.5) /
                     grid_res;
  const double C_j = (grid_pos.y() + grid_len.y() * 0.5 - occ_pos.y() -
                      occ_len.y() * 0.5 + occ_res_y * 0.5) /
                     grid_res;
  const double scale_x = occ_res_x / grid_res;
  const double scale_y = occ_res_y / grid_res;

  const std::string layer_name = layers_[Layer::OBSTACLE];
  const auto &occ_data = (*occupied_map_2d)[layer_name];
  auto &grid_data = grid_map_[layer_name];

  const int occ_sx = static_cast<int>(occ_size(0));
  const int occ_sy = static_cast<int>(occ_size(1));
  const int grid_sx = static_cast<int>(grid_size(0));
  const int grid_sy = static_cast<int>(grid_size(1));
  const int occ_start_x = occ_start(0);
  const int occ_start_y = occ_start(1);
  const int grid_start_x = grid_start(0);
  const int grid_start_y = grid_start(1);

  for (int i = 0; i < occ_sx; ++i) {
    const int grid_i_raw = static_cast<int>(C_i + scale_x * i);
    if (grid_i_raw < 0 || grid_i_raw >= grid_sx)
      continue;

    int gi = (grid_i_raw + grid_start_x) % grid_sx;
    if (gi < 0)
      gi += grid_sx;

    int oi = (i + occ_start_x) % occ_sx;
    if (oi < 0)
      oi += occ_sx;

    for (int j = 0; j < occ_sy; ++j) {
      const int grid_j_raw = static_cast<int>(C_j + scale_y * j);
      if (grid_j_raw < 0 || grid_j_raw >= grid_sy)
        continue;

      int oj = (j + occ_start_y) % occ_sy;
      if (oj < 0)
        oj += occ_sy;

      if (occ_data(oi, oj) < 0.01f) {
        int gj = (grid_j_raw + grid_start_y) % grid_sy;
        if (gj < 0)
          gj += grid_sy;
        grid_data(gi, gj) = 0.0f;
      }
    }
  }
}

double CostMapProduction::GetReferenceOffsetDist(const double relative_x,
                                                 const double relative_y) {
  grid_map::Position pos(relative_x, relative_y);
  if (grid_map_.isInside(pos)) {
    return grid_map_.atPosition(layers_[Layer::REFERENCE_OFFSET_DIST], pos,
                                grid_map::InterpolationMethods::INTER_LINEAR);
  }
  return 1.0;
}

double CostMapProduction::GetReferenceOffset(const double relative_x,
                                             const double relative_y) {
  grid_map::Position pos(relative_x, relative_y);
  if (grid_map_.isInside(pos)) {
    return grid_map_.atPosition(layers_[Layer::REFERENCE_OFFSET_DIST], pos,
                                grid_map::InterpolationMethods::INTER_LINEAR);
  }
  return 1e6;
}

} // namespace preprocess
} // namespace planning
