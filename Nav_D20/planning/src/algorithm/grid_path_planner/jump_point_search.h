#pragma once
#include "src/algorithm/grid_path_planner/planner_base.h"
namespace planning {
namespace algorithm {
class JumpPointSearch : public PlannerBase {
public:
  bool plan(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
            const grid_map::GridMap &map, std::vector<Eigen::Vector2d> &path,
            double far_away_obstacle_dist = 2.5,
            int heuristic_type = 2) override {
    std::cout << "JPS plan start!" << std::endl;
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
    expand_.clear();
    std::priority_queue<Node, std::vector<Node>, Node::compare_cost> open_list;
    std::unordered_map<int, Node> closed_list;
    open_list.push(start_node);

    while (!open_list.empty()) {
      auto current = open_list.top();
      open_list.pop();

      if (closed_list.find(current.id) != closed_list.end())
        continue;

      closed_list.insert(std::make_pair(current.id, current));
      expand_.emplace_back(current.x, current.y);

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
      auto is_inside = [&](const grid_map::Index &index) -> bool {
        return index.x() >= 0 && index.x() < map.getSize()(0) &&
               index.y() >= 0 && index.y() < map.getSize()(1);
      };

      auto detect_force_neighbor = [&](const Node &point,
                                       const Node &motion) -> bool {
        int x = point.x;
        int y = point.y;
        int x_dir = motion.x;
        int y_dir = motion.y;
        // horizontal
        if (x_dir && !y_dir) {
          grid_map::Index node1(x, y + 1);
          grid_map::Index node2(x + x_dir, y + 1);
          grid_map::Index node3(x, y - 1);
          grid_map::Index node4(x + x_dir, y - 1);
          if (!is_inside(node1) || !is_inside(node2)) {
            return false;
          }
          if (is_collision(node1) && !is_collision(node2)) {
            return true;
          }

          if (!is_inside(node3) || !is_inside(node4)) {
            return false;
          }
          if (is_collision(node3) && !is_collision(node4)) {
            return true;
          }
        }
        // vertical
        if (!x_dir && y_dir) {
          grid_map::Index node1(x + 1, y);
          grid_map::Index node2(x + 1, y + y_dir);
          grid_map::Index node3(x - 1, y);
          grid_map::Index node4(x - 1, y + y_dir);
          if (!is_inside(node1) || !is_inside(node2)) {
            return false;
          }
          if (is_collision(node1) && !is_collision(node2)) {
            return true;
          }

          if (!is_inside(node3) || !is_inside(node4)) {
            return false;
          }
          if (is_collision(node3) && !is_collision(node4)) {
            return true;
          }
        }
        // diagonal
        if (x_dir && y_dir) {
          grid_map::Index node1(x - x_dir, y);
          grid_map::Index node2(x - x_dir, y + y_dir);
          grid_map::Index node3(x, y - y_dir);
          grid_map::Index node4(x + x_dir, y - y_dir);
          if (!is_inside(node1) || !is_inside(node2)) {
            return false;
          }
          if (is_collision(node1) && !is_collision(node2)) {
            return true;
          }

          if (!is_inside(node3) || !is_inside(node4)) {
            return false;
          }
          if (is_collision(node3) && !is_collision(node4)) {
            return true;
          }
        }
        return false;
      };

      std::function<Node(Node, Node)> jump = [&](const Node &point,
                                                 const Node &motion) -> Node {
        auto new_node = point + motion;
        grid_map::Position point_new;
        if (!map.getPosition(grid_map::Index(new_node.x, new_node.y),
                             point_new)) {
          return Node();
        }
        double obstacle_penalty = std::exp(
            far_away_obstacle_dist - map.atPosition("distance", point_new));
        new_node.g = point.g + motion.g + obstacle_penalty;

        new_node.id = xy_to_id(new_node);
        new_node.parent_id = point.id;
        new_node.h = heuristic(Eigen::Vector2d(new_node.x, new_node.y),
                               Eigen::Vector2d(goal_node.x, goal_node.y),
                               heuristic_type);

        if (new_node.x < 0 || new_node.y < 0 ||
            new_node.x >= map.getSize()(0) || new_node.y >= map.getSize()(1)) {
          return Node();
        }
        if (is_collision(grid_map::Index(new_node.x, new_node.y))) {
          return Node();
        }
        if (new_node == goal_node) {
          return new_node;
        }
        if (motion.x && motion.y) {
          Node x_dir(motion.x, 0, 1, 0, 0, 0);
          Node y_dir(0, motion.y, 1, 0, 0, 0);
          if (jump(new_node, x_dir).id != -1 || jump(new_node, y_dir).id != -1)
            return new_node;
        }
        if (detect_force_neighbor(new_node, motion))
          return new_node;
        else
          return jump(new_node, motion);
      };

      for (const auto &motion : motions_) {
        auto node_new = jump(current, motion);
        // node_new not exits or in closed list
        if (node_new.id == -1 ||
            closed_list.find(node_new.id) != closed_list.end())
          continue;

        node_new.parent_id = current.id;
        open_list.push(node_new);
      }
    }
    return false;
  }
};
} // namespace algorithm
} // namespace planning
