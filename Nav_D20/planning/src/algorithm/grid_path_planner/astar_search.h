#pragma once
#include "src/algorithm/grid_path_planner/planner_base.h"
#include <limits>

namespace planning {
namespace algorithm {
class AstarSearch : public PlannerBase {
public:
  bool plan(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
            const grid_map::GridMap &map, std::vector<Eigen::Vector2d> &path,
            double far_away_obstacle_dist = 2.5,
            int heuristic_type = 2) override {
    std::cout << "Optimized AstarSearch plan start!" << std::endl;
    auto is_pos_collision = [&](const Eigen::Vector2d &point) {
      if (!map.isInside(point)) {
        AERROR << "Point is out of map!";
        return false;
      }
      return map.atPosition("distance_cost", point) > 0.99;
    };
    auto is_collision = [&](const grid_map::Index &index) {
      grid_map::Position point;
      if (!map.getPosition(index, point)) {
        AERROR << "Index is out of map!";
      }
      return map.atPosition("distance_cost", point) > 0.99;
    };
    auto is_collision_node = [&](const Node &node) {
      return is_collision(grid_map::Index(node.x, node.y));
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
    std::priority_queue<Node, std::vector<Node>, Node::compare_cost> open_list;
    // std::unordered_map<int, Node> closed_list;
    const int map_width = map.getSize()(0);
    const int map_height = map.getSize()(1);
    std::vector<Node> closed_list(map_width * map_height, Node());
    std::vector<bool> visited(map_width * map_height, false);

    open_list.push(start_node);

    while (!open_list.empty()) {
      Node current = open_list.top();
      open_list.pop();
      if (visited[current.id]) {
        continue;
      }
      visited[current.id] = true;
      closed_list[current.id] = current;

      if (current.id == goal_node.id) {
        Node temp = current;
        while (temp.id != start_node.id) {
          grid_map::Position position;
          if (map.getPosition(grid_map::Index(temp.x, temp.y), position)) {
            path.emplace_back(position.x(), position.y());
          }
          temp = closed_list[temp.parent_id];
        }
        path.push_back(start);
        if (path.size() > 1) {
          std::reverse(path.begin(), path.end());
        }
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

        grid_map::Position point;
        map.getPosition(grid_map::Index(next_x, next_y), point);
        if (map.atPosition("distance_cost", point) > 0.9) {
          continue;
        }

        double new_g = current.g + m.g +
                       std::exp(far_away_obstacle_dist -
                                map.atPosition("distance", point));
        Node &neighbor = closed_list[next_id];
        if (new_g < neighbor.g || neighbor.g == 0.0) {
          neighbor.g = new_g;
          neighbor.h = heuristic(Eigen::Vector2d(next_x, next_y),
                                 Eigen::Vector2d(goal_node.x, goal_node.y),
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
};

} // namespace algorithm
} // namespace planning