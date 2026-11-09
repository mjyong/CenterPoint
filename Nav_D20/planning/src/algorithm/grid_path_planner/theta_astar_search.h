#pragma once
#include "src/algorithm/grid_path_planner/planner_base.h"
namespace planning {
namespace algorithm {
class ThetaAstarSearch : public PlannerBase {
public:
  bool plan(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
            const grid_map::GridMap &map, std::vector<Eigen::Vector2d> &path,
            double far_away_obstacle_dist = 2.5,
            int heuristic_type = 2) override {
    std::cout << "ThetaAstarSearch plan start!" << std::endl;
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
    auto is_collision_between_nodes = [&](const Node &node1,
                                          const Node &node2) {
      grid_map::LineIterator iterator(map, grid_map::Index(node1.x, node1.y),
                                      grid_map::Index(node2.x, node2.y));
      for (; !iterator.isPastEnd(); ++iterator) {
        if (is_collision(*iterator)) {
          return true;
        }
      }
      return false;
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
    std::unordered_map<int, Node> closed_list;
    open_list.push(start_node);

    while (!open_list.empty()) {
      auto current = open_list.top();
      open_list.pop();
      if (current.g >= std::numeric_limits<double>::max())
        continue;
      if (closed_list.find(current.id) != closed_list.end()) {
        continue;
      }

      // update_parent(current);
      closed_list.insert(std::make_pair(current.id, current));

      // goal found
      if (current == goal_node) {
        auto close_list_to_path = [&](std::vector<Node> &node_path) {
          auto current = closed_list.find(goal_node.id);
          while (current->second != start_node) {
            node_path.emplace_back(current->second.x, current->second.y);
            auto it = closed_list.find(current->second.parent_id);
            if (it != closed_list.end())
              current = it;
            else
              return false;
          }
          node_path.push_back(start_node);
          return true;
        };
        std::vector<Node> backtrace;
        if (close_list_to_path(backtrace)) {
          for (auto iter = backtrace.rbegin(); iter != backtrace.rend();
               ++iter) {
            grid_map::Position position;
            if (map.getPosition(grid_map::Index(iter->x, iter->y), position)) {
              path.emplace_back(position.x(), position.y());
            }
          }
        }
        return true;
      }

      // explore neighbor
      for (const auto &m : motions_) {
        auto node_new = current + m;
        grid_map::Position point_new;
        map.getPosition(grid_map::Index(node_new.x, node_new.y), point_new);
        double obstacle_penalty = std::exp(
            far_away_obstacle_dist - map.atPosition("distance", point_new));
        node_new.parent_id = current.id;
        node_new.g = current.g + m.g + obstacle_penalty;

        node_new.h = heuristic(Eigen::Vector2d(node_new.x, node_new.y),
                               Eigen::Vector2d(goal_node.x, goal_node.y),
                               heuristic_type);
        node_new.id = xy_to_id(node_new);
        node_new.parent_id = current.id;
        if (closed_list.find(node_new.id) != closed_list.end()) {
          continue;
        }
        if (node_new.x < 0 || node_new.y < 0 ||
            node_new.x >= map.getSize()(0) || node_new.y >= map.getSize()(1)) {
          continue;
        }
        if (is_collision_node(node_new)) {
          continue;
        }
        Node parent;
        parent.id = current.parent_id;
        auto find_xy = id_to_xy(parent.id);
        parent.x = find_xy.first;
        parent.y = find_xy.second;
        auto find_parent = closed_list.find(parent.id);
        if (find_parent != closed_list.end()) {
          parent = find_parent->second;
          const double dist =
              heuristic(Eigen::Vector2d(node_new.x, node_new.y),
                        Eigen::Vector2d(parent.x, parent.y), heuristic_type);
          if (parent.g + dist < node_new.g) {
            if (!is_collision_between_nodes(parent, node_new)) {
              node_new.g = parent.g + dist + obstacle_penalty;
              node_new.parent_id = parent.id;
            }
          }
        }
        open_list.push(node_new);
      }
    }
    return false;
  }
};

} // namespace algorithm
} // namespace planning