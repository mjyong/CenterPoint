#pragma once
#include "src/algorithm/grid_path_planner/dynamicvoronoi/dynamicvoronoi.h"
#include "src/algorithm/grid_path_planner/planner_base.h"
#include <limits>
#include <queue>
#include <unordered_map>

namespace planning {
namespace algorithm {

class VoronoiAstarSearch : public PlannerBase {
public:
  VoronoiAstarSearch(DynamicVoronoi *voronoi_map) : voronoi_(voronoi_map) {}
  bool plan(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
            const grid_map::GridMap &map, std::vector<Eigen::Vector2d> &path,
            double far_away_obstacle_dist = 2.5,
            int heuristic_type = 2) override {
    std::cout << "VoronoiAstarSearch plan start!" << std::endl;
    auto is_pos_collision = [&](const Eigen::Vector2d &point) {
      if (!map.isInside(point)) {
        AERROR << "Point is out of map!";
        return false;
      }
      return map.atPosition("distance_cost", point) > 0.99;
    };

    if (is_pos_collision(start)) {
      AERROR << "Start point is in collision!";
      return false;
    }
    if (is_pos_collision(end)) {
      AERROR << "End point is in collision!";
      return false;
    }

    grid_map::Index start_index;
    if (!map.getIndex(start, start_index)) {
      AERROR << "Start point is out of map!";
      return false;
    }
    grid_map::Index end_index;
    if (!map.getIndex(end, end_index)) {
      AERROR << "End point is out of map!";
      return false;
    }

    Node start_node(start_index.x(), start_index.y());
    Node goal_node(end_index.x(), end_index.y());
    auto xy_to_id = [&](const Node &node) {
      return node.x + node.y * map.getSize()(0);
    };
    auto id_to_xy = [&](const int id) {
      return std::make_pair(id % map.getSize()(0), id / map.getSize()(0));
    };
    start_node.id = xy_to_id(start_node);
    goal_node.id = xy_to_id(goal_node);
    path.clear();

    grid_map::Index voronoi_start_idx =
        findNearestVoronoiNode(start_index, map);
    grid_map::Index voronoi_end_idx = findNearestVoronoiNode(end_index, map);

    if (voronoi_start_idx.x() == -1 || voronoi_end_idx.x() == -1) {
      AERROR << "Could not find nearest Voronoi node for start or end point!";
      return false;
    }

    Node voronoi_start_node(voronoi_start_idx.x(), voronoi_start_idx.y());
    Node voronoi_goal_node(voronoi_end_idx.x(), voronoi_end_idx.y());
    voronoi_start_node.id = xy_to_id(voronoi_start_node);
    voronoi_goal_node.id = xy_to_id(voronoi_goal_node);

    std::priority_queue<Node, std::vector<Node>, Node::compare_cost> open_list;
    const int map_width = map.getSize()(0);
    const int map_height = map.getSize()(1);
    std::vector<Node> closed_list(map_width * map_height, Node());
    std::vector<bool> visited(map_width * map_height, false);

    open_list.push(voronoi_start_node);

    while (!open_list.empty()) {
      Node current = open_list.top();
      open_list.pop();

      if (visited[current.id]) {
        continue;
      }
      visited[current.id] = true;
      closed_list[current.id] = current;

      if (current.id == voronoi_goal_node.id) {
        std::vector<Eigen::Vector2d> voronoi_path_segment;
        Node temp = current;
        while (temp.id != voronoi_start_node.id) {
          grid_map::Position position;
          if (map.getPosition(grid_map::Index(temp.x, temp.y), position)) {
            voronoi_path_segment.emplace_back(position.x(), position.y());
          }
          temp = closed_list[temp.parent_id];
        }
        grid_map::Position position;
        if (map.getPosition(
                grid_map::Index(voronoi_start_node.x, voronoi_start_node.y),
                position)) {
          voronoi_path_segment.push_back(position);
        }
        std::reverse(voronoi_path_segment.begin(), voronoi_path_segment.end());

        path.push_back(start);
        path.insert(path.end(), voronoi_path_segment.begin(),
                    voronoi_path_segment.end());
        path.push_back(end);

        // path = voronoi_path_segment;
        return true;
      }

      for (const auto &m : motions_) {
        int next_x = current.x + m.x;
        int next_y = current.y + m.y;

        if (next_x < 0 || next_y < 0 || next_x >= map_width ||
            next_y >= map_height) {
          continue;
        }

        int next_id = xy_to_id(Node(next_x, next_y));
        if (visited[next_id]) {
          continue;
        }

        if (!voronoi_->isVoronoi(next_x, next_y)) {
          continue;
        }
        grid_map::Position point;
        map.getPosition(grid_map::Index(next_x, next_y), point);
        double new_g = current.g + m.g +
                       std::exp(far_away_obstacle_dist -
                                map.atPosition("distance", point));
        Node &neighbor = closed_list[next_id];

        if (new_g < neighbor.g || neighbor.g == 0.0) {
          neighbor.g = new_g;
          neighbor.h = heuristic(
              Eigen::Vector2d(next_x, next_y),
              Eigen::Vector2d(voronoi_goal_node.x, voronoi_goal_node.y),
              heuristic_type);
          neighbor.x = next_x;
          neighbor.y = next_y;
          neighbor.id = next_id;
          neighbor.parent_id = current.id;
          open_list.push(neighbor);
        }
      }
    }

    return false;
  }

private:
  grid_map::Index findNearestVoronoiNode(const grid_map::Index &start_idx,
                                         const grid_map::GridMap &map) {
    double min_dist_sq = std::numeric_limits<double>::max();
    grid_map::Index nearest_voronoi_idx(-1, -1);

    for (int x = 0; x < map.getSize()(0); ++x) {
      for (int y = 0; y < map.getSize()(1); ++y) {
        if (voronoi_->isVoronoi(x, y)) {
          double dist_sq = (x - start_idx.x()) * (x - start_idx.x()) +
                           (y - start_idx.y()) * (y - start_idx.y());
          if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_voronoi_idx = grid_map::Index(x, y);
          }
        }
      }
    }
    return nearest_voronoi_idx;
  }

private:
  DynamicVoronoi *voronoi_;
};

} // namespace algorithm
} // namespace planning