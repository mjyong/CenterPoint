#pragma once
#include "src/common/structures.h"
#include "src/common/util/log.h"
#include <grid_map_core/grid_map_core.hpp>
namespace planning {
namespace algorithm {
class PlannerBase {
protected:
  typedef struct Node {
    int x;
    int y;
    double g;
    double h;
    int id;
    int parent_id;
    Node(int _x = 0, int _y = 0, double _g = 0, double _h = 0, int _id = -1,
         int _parent_id = -1)
        : x(_x), y(_y), g(_g), h(_h), id(_id), parent_id(_parent_id) {}

    Node operator+(const Node &n) const { return Node(x + n.x, y + n.y); };
    Node operator-(const Node &n) const { return Node(x - n.x, y - n.y); };
    bool operator==(const Node &n) const { return x == n.x && y == n.y; };
    bool operator!=(const Node &n) const { return !operator==(n); };

    struct compare_cost {
      bool operator()(const Node &n1, const Node &n2) const {
        return (n1.g + n1.h > n2.g + n2.h) ||
               ((n1.g + n1.h == n2.g + n2.h) && (n1.h > n2.h));
      };
    };
    struct compare_coordinates {
      bool operator()(const Node &n1, const Node &n2) const {
        return (n1.x == n2.x) && (n1.y == n2.y);
      };
    };
  } Node;
  const std::vector<Node> motions_ = {
      {0, 1, 1.0},           {1, 0, 1.0},
      {0, -1, 1.0},          {-1, 0, 1.0},
      {1, 1, std::sqrt(2)},  {1, -1, std::sqrt(2)},
      {-1, 1, std::sqrt(2)}, {-1, -1, std::sqrt(2)},
  }; // 8 directions
  std::vector<Eigen::Vector2d> expand_;

  double heuristic(const Eigen::Vector2d &source, const Eigen::Vector2d &target,
                   int heuristic_type = 2) {
    double h = 0.0;
    switch (heuristic_type) {
    case 0:
      h = manhattan(source, target);
      break;
    case 1:
      h = euclidean(source, target);
      break;
    case 2:
      h = diagonal(source, target);
      break;
    case 3:
      h = dijkstra();
    default:
      h = diagonal(source, target);
      break;
    }
    return h;
  }

public:
  virtual bool plan(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
                    const grid_map::GridMap &map,
                    std::vector<Eigen::Vector2d> &path,
                    double far_away_obstacle_dist = 2.5,
                    int heuristic_type = 2) = 0;

private:
  Eigen::Vector2d getDelta(const Eigen::Vector2d &source,
                           const Eigen::Vector2d &target) {
    return {abs(source.x() - target.x()), abs(source.y() - target.y())};
  }
  double manhattan(const Eigen::Vector2d &source,
                   const Eigen::Vector2d &target) {
    return getDelta(source, target).sum();
  }
  double euclidean(const Eigen::Vector2d &source,
                   const Eigen::Vector2d &target) {
    return getDelta(source, target).norm();
  }
  double diagonal(const Eigen::Vector2d &source,
                  const Eigen::Vector2d &target) {
    return getDelta(source, target).maxCoeff();
  }
  double dijkstra() { return 0.0; }
};
} // namespace algorithm
} // namespace planning