#pragma once
#include "src/common/util/log.h"
#include "teb_planner_conf.pb.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace teb_planner {

class CoarsePathSearch {
  typedef struct Node {
    Eigen::Vector3d coord;
    Eigen::Vector3i index;
    double g;
    double h;
    int father;
    double res_;
    double theta_res_;
    int theta_max_index_;
    double min_x_;
    double max_x_;
    double min_y_;
    double max_y_;

    Node(const Eigen::Vector3d &coord, double res, double theta_res,
         double min_x, double max_x, double min_y, double max_y)
        : res_(res), theta_res_(theta_res),
          theta_max_index_(static_cast<int>(M_PI / theta_res_)), min_x_(min_x),
          max_x_(max_x), min_y_(min_y), max_y_(max_y) {
      this->coord = coord;
      this->index = getIndex(coord);
      this->g = 0;
      this->h = 0;
      this->father = -1;
    }
    Node(const Eigen::Vector3i &index, double res, double theta_res,
         double min_x, double max_x, double min_y, double max_y)
        : res_(res), theta_res_(theta_res),
          theta_max_index_(static_cast<int>(M_PI / theta_res_)), min_x_(min_x),
          max_x_(max_x), min_y_(min_y), max_y_(max_y) {
      this->index = index;
      this->coord = getCoord(index);
      this->g = 0;
      this->h = 0;
      this->father = -1;
    }
    Node shifted(int dx, int dy, int dtheta) const {
      Eigen::Vector3i next_index = index + Eigen::Vector3i(dx, dy, dtheta);
      if (next_index(2) > theta_max_index_) {
        next_index(2) -= 2 * theta_max_index_;
      } else if (next_index(2) < -theta_max_index_) {
        next_index(2) += 2 * theta_max_index_;
      }
      return Node(next_index, res_, theta_res_, min_x_, max_x_, min_y_, max_y_);
    }
    bool operator==(const Node &n) const { return this->index == n.index; }
    bool operator!=(const Node &n) const { return this->index != n.index; }

    struct hash {
      size_t operator()(Node const &p) const noexcept {
        size_t seed = 0;
        hashCombine(seed, p.index(0));
        hashCombine(seed, p.index(1));
        hashCombine(seed, p.index(2));
        return seed;
      }

    private:
      static void hashCombine(size_t &seed, int value) noexcept {
        seed ^=
            std::hash<int>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
    };

    bool isInside() const {
      return this->coord(0) >= min_x_ && this->coord(0) <= max_x_ &&
             this->coord(1) >= min_y_ && this->coord(1) <= max_y_;
    }
    Eigen::Vector3i getIndex(const Eigen::Vector3d &coord) const {
      return Eigen::Vector3i(coord(0) / res_, coord(1) / res_,
                             coord(2) / theta_res_);
    }

    Eigen::Vector3d getCoord(const Eigen::Vector3i &index) const {
      return Eigen::Vector3d(index(0) * res_, index(1) * res_,
                             index(2) * theta_res_);
    }
  } Node;

public:
  CoarsePathSearch(
      std::function<bool(const double, const double, const double)>
          is_collision_func,
      std::function<double(const double, const double)> composite_cost_func,
      const planning::config::TebPlannerConf_CoarseSearchConf &conf)
      : is_collision_func_(is_collision_func),
        composite_cost_func_(composite_cost_func), conf_(conf) {}

  const std::vector<Eigen::Vector3d> &getPath() { return path_; }

  void clearReferencePath() {
    reference_path_.clear();
    ref_grid_index_.clear();
  }
  void setReferencePath(const std::vector<Eigen::Vector3d> &ref_path) {
    reference_path_ = ref_path;
    if (reference_path_.empty()) {
      ref_grid_index_.clear();
      return;
    }
    buildReferenceGridIndex();
  }

  void
  setLastSuccessPath(const std::vector<Eigen::Vector3d> &last_success_path) {
    last_success_path_ = last_success_path;
    if (last_success_path_.empty()) {
      last_path_grid_index_.clear();
      return;
    }
    buildLastSuccessPathGridIndex();
  }

  bool plan(const Eigen::Vector3d &start, const Eigen::Vector3d &end,
            bool constant_final_theta = false) {
    if (is_collision_func_(start(0), start(1), start(2))) {
      AERROR << "start pose is in collision";
      return false;
    }

    bool end_in_collision = false;
    if (is_collision_func_(end(0), end(1), end(2))) {
      AERROR << "end pose is in collision";
      end_in_collision = true;
      if (constant_final_theta) {
        return false;
      }
    }

    Node start_node(start, conf_.xy_resolution(), conf_.theta_resolution(),
                    conf_.min_x(), conf_.max_x(), conf_.min_y(), conf_.max_y());
    Node end_node(end, conf_.xy_resolution(), conf_.theta_resolution(),
                  conf_.min_x(), conf_.max_x(), conf_.min_y(), conf_.max_y());
    use_last_success_path_ = shouldUseLastSuccessPath(start, end);
    AINFO << "start node: " << start_node.index.transpose()
          << " coord: " << start_node.coord.transpose()
          << ", end node: " << end_node.index.transpose()
          << " coord: " << end_node.coord.transpose() << ", search bounds: x["
          << conf_.min_x() << "," << conf_.max_x() << "] y[" << conf_.min_y()
          << "," << conf_.max_y() << "]"
          << ", constant_final_theta: " << constant_final_theta
          << ", reference path size: " << reference_path_.size()
          << ", use last success path: " << use_last_success_path_;

    if (!end_node.isInside()) {
      AERROR << "End node is outside search bounds, coord: "
             << end_node.coord.transpose();
    }

    int max_search_nodes = constant_final_theta
                               ? conf_.max_search_nodes_with_theta()
                               : conf_.max_search_nodes_no_theta();
    int total_expanded = 0;
    int total_rejected_isinside = 0;
    int total_rejected_collision = 0;
    int total_rejected_visited = 0;

    std::vector<Node> open_list_vec;
    open_list_vec.reserve(1024);
    auto cmp = [](const Node &a, const Node &b) {
      return (a.g + a.h) > (b.g + b.h);
    };
    std::make_heap(open_list_vec.begin(), open_list_vec.end(), cmp);
    std::vector<Node> close_list;
    close_list.reserve(2048);
    std::unordered_set<Node, Node::hash> visited;
    visited.reserve(4096);
    std::unordered_map<Node, double, Node::hash> best_g;
    best_g.reserve(4096);
    start_node.g = 0.0;
    best_g[start_node] = start_node.g;
    open_list_vec.push_back(start_node);
    std::push_heap(open_list_vec.begin(), open_list_vec.end(), cmp);

    while (!open_list_vec.empty() &&
           static_cast<int>(close_list.size()) < max_search_nodes) {
      std::pop_heap(open_list_vec.begin(), open_list_vec.end(), cmp);
      Node current_node = open_list_vec.back();
      open_list_vec.pop_back();
      auto best_current_it = best_g.find(current_node);
      if (best_current_it != best_g.end() &&
          current_node.g > best_current_it->second + 1e-9) {
        continue;
      }
      if (visited.find(current_node) != visited.end()) {
        continue;
      }
      close_list.emplace_back(current_node);
      visited.insert(current_node);
      double dist_to_end_current =
          (current_node.coord.head<2>() - end_node.coord.head<2>()).norm();
      if ((current_node.index == end_node.index) ||
          (!constant_final_theta && dist_to_end_current < 0.5) ||
          (end_in_collision && dist_to_end_current < 2.5)) {
        AINFO << "find the path, close size:" << close_list.size()
              << ", expanded: " << total_expanded
              << ", rejected: isInside=" << total_rejected_isinside
              << " collision=" << total_rejected_collision
              << " visited=" << total_rejected_visited;
        path_.clear();
        if (constant_final_theta) {
          path_.push_back(end);
        }
        int father_idx = current_node.father;
        while (father_idx >= 0) {
          path_.push_back(close_list[father_idx].coord);
          father_idx = close_list[father_idx].father;
        }
        path_.push_back(start);
        std::reverse(path_.begin(), path_.end());
        return true;
      }
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          for (int k = -1; k <= 1; ++k) {
            if (i == 0 && j == 0 && k == 0) {
              continue;
            }
            Node neighbor = current_node.shifted(i, j, k);
            if (!neighbor.isInside()) {
              total_rejected_isinside++;
              continue;
            }
            if (is_collision_func_(neighbor.coord(0), neighbor.coord(1),
                                   neighbor.coord(2))) {
              total_rejected_collision++;
              continue;
            }
            double node_cost =
                composite_cost_func_(neighbor.coord(0), neighbor.coord(1));
            if (node_cost >= 1.0) {
              total_rejected_collision++;
              continue;
            }
            if (visited.find(neighbor) != visited.end()) {
              total_rejected_visited++;
              continue;
            }
            total_expanded++;
            neighbor.g = current_node.g +
                         calculateTransitionCost(current_node, neighbor);
            auto best_it = best_g.find(neighbor);
            if (best_it != best_g.end() &&
                neighbor.g >= best_it->second - 1e-9) {
              continue;
            }
            best_g[neighbor] = neighbor.g;
            Eigen::Vector2d to_goal =
                end_node.coord.head<2>() - neighbor.coord.head<2>();
            double dist_to_end = to_goal.norm();
            if (constant_final_theta) {
              double theta_err =
                  abs(normalize_angle(neighbor.coord(2) - end_node.coord(2)));
              neighbor.h = dist_to_end + theta_err;
            } else {
              double ideal_theta = std::atan2(to_goal(1), to_goal(0));
              double theta_err_with_ideal =
                  abs(normalize_angle(neighbor.coord(2) - ideal_theta));
              neighbor.h = dist_to_end + theta_err_with_ideal;
            }
            neighbor.h *= 2.0;
            neighbor.father = close_list.size() - 1;
            open_list_vec.push_back(neighbor);
            std::push_heap(open_list_vec.begin(), open_list_vec.end(), cmp);
          }
        }
      }
    }
    AERROR << "No path found, close size: " << close_list.size()
           << ", expanded: " << total_expanded
           << ", rejected: isInside=" << total_rejected_isinside
           << " collision=" << total_rejected_collision
           << " visited=" << total_rejected_visited;
    return false;
  }

private:
  static double normalize_angle(double angle) {
    double a = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (a < 0.0) {
      a += (2.0 * M_PI);
    }
    return a - M_PI;
  }

  static int64_t makeGridKey(int gx, int gy) {
    uint64_t ux = static_cast<uint32_t>(gx);
    uint64_t uy = static_cast<uint32_t>(gy);
    return static_cast<int64_t>((ux << 32) | uy);
  }

  static double squaredDistance(double x, double y, const Eigen::Vector3d &p) {
    double dx = x - p(0);
    double dy = y - p(1);
    return dx * dx + dy * dy;
  }

  double calculateTransitionCost(const Node &from, const Node &to) const {
    double step_dist = (from.coord.head<2>() - to.coord.head<2>()).norm();
    double cost = step_dist;
    double angle_diff = std::abs(normalize_angle(from.coord(2) - to.coord(2)));
    cost += angle_diff;
    if (!reference_path_.empty()) {
      double ref_dist = findClosestReferenceDistance(to.coord(0), to.coord(1));
      cost += conf_.weight_reference_path() * ref_dist *
              std::max(step_dist, to.res_);
    }
    if (use_last_success_path_) {
      double last_path_dist =
          findClosestLastSuccessPathDistance(to.coord(0), to.coord(1));
      cost += conf_.weight_last_path_consistency() * last_path_dist *
              std::max(step_dist, to.res_);
    }
    return cost;
  }

  bool shouldUseLastSuccessPath(const Eigen::Vector3d &start,
                                const Eigen::Vector3d &end) const {
    if (conf_.weight_last_path_consistency() <= 1e-6 ||
        last_success_path_.empty()) {
      return false;
    }
    double start_dist = findClosestLastSuccessPathDistance(start(0), start(1));
    double end_dist = findClosestLastSuccessPathDistance(end(0), end(1));
    constexpr double max_start_dist = 2.0;
    return start_dist <= max_start_dist && end_dist <= max_start_dist;
  }

  double findClosestPathDistance(const std::vector<Eigen::Vector3d> &path,
                                 double x, double y) const {
    if (path.empty()) {
      return 0.0;
    }
    double min_dist_sq = std::numeric_limits<double>::max();
    for (const auto &p : path) {
      min_dist_sq = std::min(min_dist_sq, squaredDistance(x, y, p));
    }
    return std::sqrt(min_dist_sq);
  }

  void buildReferenceGridIndex() {
    ref_grid_resolution_ = conf_.ref_grid_resolution();
    ref_min_x_ = std::numeric_limits<double>::max();
    ref_min_y_ = std::numeric_limits<double>::max();
    for (const auto &p : reference_path_) {
      ref_min_x_ = std::min(ref_min_x_, p(0));
      ref_min_y_ = std::min(ref_min_y_, p(1));
    }
    ref_grid_index_.clear();
    for (int i = 0; i < static_cast<int>(reference_path_.size()); ++i) {
      int gx = static_cast<int>(std::floor(
          (reference_path_[i](0) - ref_min_x_) / ref_grid_resolution_));
      int gy = static_cast<int>(std::floor(
          (reference_path_[i](1) - ref_min_y_) / ref_grid_resolution_));
      ref_grid_index_[makeGridKey(gx, gy)].push_back(i);
    }
  }

  void buildLastSuccessPathGridIndex() {
    last_path_grid_resolution_ = conf_.ref_grid_resolution();
    last_path_min_x_ = std::numeric_limits<double>::max();
    last_path_min_y_ = std::numeric_limits<double>::max();
    for (const auto &p : last_success_path_) {
      last_path_min_x_ = std::min(last_path_min_x_, p(0));
      last_path_min_y_ = std::min(last_path_min_y_, p(1));
    }
    last_path_grid_index_.clear();
    for (int i = 0; i < static_cast<int>(last_success_path_.size()); ++i) {
      int gx = static_cast<int>(
          std::floor((last_success_path_[i](0) - last_path_min_x_) /
                     last_path_grid_resolution_));
      int gy = static_cast<int>(
          std::floor((last_success_path_[i](1) - last_path_min_y_) /
                     last_path_grid_resolution_));
      last_path_grid_index_[makeGridKey(gx, gy)].push_back(i);
    }
  }

  double findClosestReferenceDistance(double x, double y) const {
    return findClosestIndexedPathDistance(reference_path_, ref_grid_index_,
                                          ref_min_x_, ref_min_y_,
                                          ref_grid_resolution_, x, y);
  }

  double findClosestLastSuccessPathDistance(double x, double y) const {
    return findClosestIndexedPathDistance(
        last_success_path_, last_path_grid_index_, last_path_min_x_,
        last_path_min_y_, last_path_grid_resolution_, x, y);
  }

  double findClosestIndexedPathDistance(
      const std::vector<Eigen::Vector3d> &path,
      const std::unordered_map<int64_t, std::vector<int>> &grid_index,
      double min_x, double min_y, double grid_resolution, double x,
      double y) const {
    if (path.empty()) {
      return 0.0;
    }
    if (grid_index.empty() || grid_resolution <= 0.0) {
      return findClosestPathDistance(path, x, y);
    }
    int gx = static_cast<int>(std::floor((x - min_x) / grid_resolution));
    int gy = static_cast<int>(std::floor((y - min_y) / grid_resolution));
    double min_dist_sq = std::numeric_limits<double>::max();
    for (int dx = -2; dx <= 2; ++dx) {
      for (int dy = -2; dy <= 2; ++dy) {
        auto it = grid_index.find(makeGridKey(gx + dx, gy + dy));
        if (it == grid_index.end()) {
          continue;
        }
        for (int idx : it->second) {
          min_dist_sq = std::min(min_dist_sq, squaredDistance(x, y, path[idx]));
        }
      }
    }
    if (min_dist_sq == std::numeric_limits<double>::max()) {
      return findClosestPathDistance(path, x, y);
    }
    return std::sqrt(min_dist_sq);
  }

private:
  planning::config::TebPlannerConf_CoarseSearchConf conf_;
  std::vector<Eigen::Vector3d> path_;
  std::vector<Eigen::Vector3d> reference_path_;
  std::vector<Eigen::Vector3d> last_success_path_;
  bool use_last_success_path_{false};
  std::unordered_map<int64_t, std::vector<int>> ref_grid_index_;
  double ref_grid_resolution_{0.5};
  double ref_min_x_{0.0};
  double ref_min_y_{0.0};
  std::unordered_map<int64_t, std::vector<int>> last_path_grid_index_;
  double last_path_grid_resolution_{0.5};
  double last_path_min_x_{0.0};
  double last_path_min_y_{0.0};
  std::function<bool(const double, const double, const double)>
      is_collision_func_;
  std::function<double(const double, const double)> composite_cost_func_;
};

} // namespace teb_planner
