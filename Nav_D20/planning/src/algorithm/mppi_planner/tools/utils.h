// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). ROS message / tf2 / costmap_2d helpers
// are removed; pose helpers use the POD mppi::models::Pose type.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "src/algorithm/mppi_planner/models/control_sequence.h"
#include "src/algorithm/mppi_planner/models/optimizer_settings.h"
#include "src/algorithm/mppi_planner/models/path.h"
#include "src/algorithm/mppi_planner/models/state.h"
#include "src/algorithm/mppi_planner/critic_data.h"

#define M_PIF 3.141592653589793238462643383279502884e+00F
#define M_PIF_2 1.5707963267948966e+00F

namespace mppi {
namespace utils {

/**
 * @brief normalize an angle to (-M_PIF, M_PIF]
 */
inline float normalize_angle(float angle) {
  float remainder = std::fmod(angle + M_PIF, 2.0f * M_PIF);
  if (remainder < 0.0f) {
    remainder += 2.0f * M_PIF;
  }
  return remainder - M_PIF;
}

/**
 * @brief shortest signed angular distance from `from` to `to`
 */
inline float shortest_angular_distance(float from, float to) {
  return normalize_angle(to - from);
}

/**
 * @brief normalize the angles of an Eigen array to (-M_PIF, M_PIF]
 */
template <typename T>
auto normalize_angles(const T &angles) {
  return angles.unaryExpr(
      [](const float x) { return normalize_angle(x); });
}

/**
 * @brief shortest signed angular distance between two Eigen arrays
 */
template <typename F, typename T>
auto shortest_angular_distance(const F &from, const T &to) {
  return normalize_angles(to - from);
}

/**
 * @brief Check if the robot pose is within tolerance to the goal
 */
inline bool withinPositionGoalTolerance(
    float pose_tolerance, const models::Pose &robot, const models::Pose &goal) {
  const float dx = goal.x - robot.x;
  const float dy = goal.y - robot.y;
  const float dist_sq = dx * dx + dy * dy;
  return dist_sq < pose_tolerance * pose_tolerance;
}

/**
 * @brief Evaluate furthest point idx of data.path which is nearest to some
 * trajectory in data.trajectories
 */
inline size_t findPathFurthestReachedPoint(const CriticData &data) {
  int traj_cols = data.trajectories.x.cols();
  const auto traj_x = data.trajectories.x.col(traj_cols - 1);
  const auto traj_y = data.trajectories.y.col(traj_cols - 1);

  const auto dx =
      (data.path.x.transpose()).replicate(traj_x.rows(), 1).colwise() - traj_x;
  const auto dy =
      (data.path.y.transpose()).replicate(traj_y.rows(), 1).colwise() - traj_y;

  const auto dists = dx * dx + dy * dy;

  int max_id_by_trajectories = 0, min_id_by_path = 0;
  float min_distance_by_path = std::numeric_limits<float>::max();
  size_t n_rows = dists.rows();
  size_t n_cols = dists.cols();
  for (size_t i = 0; i != n_rows; i++) {
    min_id_by_path = 0;
    min_distance_by_path = std::numeric_limits<float>::max();
    for (size_t j = max_id_by_trajectories; j != n_cols; j++) {
      const float cur_dist = dists(i, j);
      if (cur_dist < min_distance_by_path) {
        min_distance_by_path = cur_dist;
        min_id_by_path = static_cast<int>(j);
      }
    }
    max_id_by_trajectories = std::max(max_id_by_trajectories, min_id_by_path);
  }
  return static_cast<size_t>(max_id_by_trajectories);
}

/**
 * @brief evaluate path furthest point if it is not set
 */
inline void setPathFurthestPointIfNotSet(CriticData &data) {
  if (!data.furthest_reached_path_point) {
    data.furthest_reached_path_point = findPathFurthestReachedPoint(data);
  }
}

/**
 * @brief evaluate path point validity (not on an obstacle) if not set.
 * Validity is queried via the obstacle predicate in the critic context.
 */
inline void setPathCostsIfNotSet(CriticData &data) {
  if (data.path_pts_valid) {
    return;
  }
  const size_t path_segments_count = data.path.x.size() - 1;
  data.path_pts_valid = std::vector<bool>(path_segments_count, false);
  for (size_t idx = 0; idx < path_segments_count; idx++) {
    const float x = data.path.x(idx);
    const float y = data.path.y(idx);
    (*data.path_pts_valid)[idx] = !data.ctx->is_obstacle_func(x, y);
  }
}

/**
 * @brief evaluate angle from pose (has angle) to point (no angle)
 * @param pose pose
 * @param point_x Point to find angle relative to X axis
 * @param point_y Point to find angle relative to Y axis
 * @param forward_preference If reversing direction is valid
 * @return Angle between two points
 */
inline float posePointAngle(
    const models::Pose &pose, float point_x, float point_y,
    bool forward_preference) {
  const float yaw = atan2f(point_y - pose.y, point_x - pose.x);

  // If no preference for forward, return smallest angle either in heading or
  // 180 of heading
  if (!forward_preference) {
    return std::min(fabs(shortest_angular_distance(yaw, pose.yaw)),
                    fabs(shortest_angular_distance(
                        yaw, normalize_angle(pose.yaw + M_PIF))));
  }

  return fabs(shortest_angular_distance(yaw, pose.yaw));
}

/**
 * @brief evaluate angle from pose (has angle) to point (has angle)
 */
inline float posePointAngle(
    const models::Pose &pose, float point_x, float point_y, float point_yaw) {
  float yaw = atan2f(point_y - pose.y, point_x - pose.x);

  if (fabs(shortest_angular_distance(yaw, point_yaw)) > M_PIF_2) {
    yaw = normalize_angle(yaw + M_PIF);
  }

  return fabs(shortest_angular_distance(yaw, pose.yaw));
}

/**
 * @brief Apply Savitzky-Golay filter to optimal trajectory
 */
inline void savitskyGolayFilter(
    models::ControlSequence &control_sequence,
    std::array<mppi::models::Control, 4> &control_history,
    const models::OptimizerSettings &settings) {
  // Savitzky-Golay Quadratic, 9-point Coefficients
  Eigen::Array<float, 9, 1> filter;
  filter << -21.0f, 14.0f, 39.0f, 54.0f, 59.0f, 54.0f, 39.0f, 14.0f, -21.0f;
  filter /= 231.0f;

  // Too short to smooth meaningfully
  const unsigned int num_sequences = control_sequence.vx.size() - 1;
  if (num_sequences < 20) {
    return;
  }

  auto applyFilter = [&](const Eigen::Array<float, 9, 1> &data) -> float {
    return (data * filter).eval().sum();
  };

  auto applyFilterOverAxis =
      [&](Eigen::ArrayXf &sequence, const Eigen::ArrayXf &initial_sequence,
          const float hist_0, const float hist_1, const float hist_2,
          const float hist_3) -> void {
    float pt_m4 = hist_0;
    float pt_m3 = hist_1;
    float pt_m2 = hist_2;
    float pt_m1 = hist_3;
    float pt = initial_sequence(0);
    float pt_p1 = initial_sequence(1);
    float pt_p2 = initial_sequence(2);
    float pt_p3 = initial_sequence(3);
    float pt_p4 = initial_sequence(4);

    for (unsigned int idx = 0; idx != num_sequences; idx++) {
      Eigen::Array<float, 9, 1> window;
      window << pt_m4, pt_m3, pt_m2, pt_m1, pt, pt_p1, pt_p2, pt_p3, pt_p4;
      sequence(idx) = applyFilter(window);
      pt_m4 = pt_m3;
      pt_m3 = pt_m2;
      pt_m2 = pt_m1;
      pt_m1 = pt;
      pt = pt_p1;
      pt_p1 = pt_p2;
      pt_p2 = pt_p3;
      pt_p3 = pt_p4;

      if (idx + 5 < num_sequences) {
        pt_p4 = initial_sequence(idx + 5);
      } else {
        // Return the last point
        pt_p4 = initial_sequence(num_sequences);
      }
    }
  };

  // Filter trajectories
  const models::ControlSequence initial_control_sequence = control_sequence;
  applyFilterOverAxis(
      control_sequence.vx, initial_control_sequence.vx, control_history[0].vx,
      control_history[1].vx, control_history[2].vx, control_history[3].vx);
  applyFilterOverAxis(
      control_sequence.vy, initial_control_sequence.vy, control_history[0].vy,
      control_history[1].vy, control_history[2].vy, control_history[3].vy);
  applyFilterOverAxis(
      control_sequence.wz, initial_control_sequence.wz, control_history[0].wz,
      control_history[1].wz, control_history[2].wz, control_history[3].wz);

  // Update control history
  unsigned int offset = settings.shift_control_sequence ? 1 : 0;
  control_history[0] = control_history[1];
  control_history[1] = control_history[2];
  control_history[2] = control_history[3];
  control_history[3] = {
      control_sequence.vx(offset), control_sequence.vy(offset),
      control_sequence.wz(offset)};
}

/**
 * @brief Compare trajectory points to find closest path point along
 * integrated distances
 * @param vec Vector of integrated distances to check
 * @param dist Distance to look for
 * @param init Starting index to search from
 */
inline unsigned int findClosestPathPt(
    const std::vector<float> &vec, const float dist, const unsigned int init = 0u) {
  float distim1 = init != 0u ? vec[init] : 0.0f; // First is 0, no accumulated distance yet
  float disti = 0.0f;
  const unsigned int size = vec.size();
  for (unsigned int i = init + 1; i != size; i++) {
    disti = vec[i];
    if (disti > dist) {
      if (i > 0 && dist - distim1 < disti - dist) {
        return i - 1;
      }
      return i;
    }
    distim1 = disti;
  }
  return size - 1;
}

// A struct to hold pose data in floating point resolution
struct Pose2D {
  float x, y, theta;
};

/**
 * @brief Shift the columns of a 2D Eigen Array or scalar values of
 *    1D Eigen Array by 1 place.
 * @param e Eigen Array
 * @param direction direction in which Array will be shifted.
 *     1 for shift in right direction and -1 for left direction.
 */
inline void shiftColumnsByOnePlace(Eigen::Ref<Eigen::ArrayXXf> e, int direction) {
  int size = e.size();
  if (size == 1) {
    return;
  }

  if ((e.cols() == 1 || e.rows() == 1) && size > 1) {
    auto start_ptr = direction == 1 ? e.data() + size - 2 : e.data() + 1;
    auto end_ptr = direction == 1 ? e.data() : e.data() + size - 1;
    while (start_ptr != end_ptr) {
      *(start_ptr + direction) = *start_ptr;
      start_ptr -= direction;
    }
    *(start_ptr + direction) = *start_ptr;
  } else {
    auto start_ptr =
        direction == 1 ? e.data() + size - 2 * e.rows() : e.data() + e.rows();
    auto end_ptr =
        direction == 1 ? e.data() : e.data() + size - e.rows();
    auto span = e.rows();
    while (start_ptr != end_ptr) {
      std::copy(start_ptr, start_ptr + span, start_ptr + direction * span);
      start_ptr -= (direction * span);
    }
    std::copy(start_ptr, start_ptr + span, start_ptr + direction * span);
  }
}

/**
 * @brief Normalize the yaws between points on the basis of final yaw angle
 *    of the trajectory.
 */
inline auto normalize_yaws_between_points(
    const Eigen::Ref<const Eigen::ArrayXf> &last_yaws,
    const Eigen::Ref<const Eigen::ArrayXf> &yaw_between_points) {
  Eigen::ArrayXf yaws =
      utils::shortest_angular_distance(last_yaws, yaw_between_points).abs();
  int size = yaws.size();
  Eigen::ArrayXf yaws_between_points_corrected(size);
  for (int i = 0; i != size; i++) {
    const float &yaw_between_point = yaw_between_points[i];
    yaws_between_points_corrected[i] =
        yaws[i] < M_PIF_2 ? yaw_between_point
                          : normalize_angle(yaw_between_point + M_PIF);
  }
  return yaws_between_points_corrected;
}

/**
 * @brief Normalize the yaws between points on the basis of goal angle.
 */
inline auto normalize_yaws_between_points(
    const float goal_yaw, const Eigen::Ref<const Eigen::ArrayXf> &yaw_between_points) {
  int size = yaw_between_points.size();
  Eigen::ArrayXf yaws_between_points_corrected(size);
  for (int i = 0; i != size; i++) {
    const float &yaw_between_point = yaw_between_points[i];
    yaws_between_points_corrected[i] =
        fabs(normalize_angle(yaw_between_point - goal_yaw)) < M_PIF_2
            ? yaw_between_point
            : normalize_angle(yaw_between_point + M_PIF);
  }
  return yaws_between_points_corrected;
}

/**
 * @brief Clamps the input between the given lower and upper bounds.
 */
inline float clamp(const float lower_bound, const float upper_bound, const float input) {
  return std::min(upper_bound, std::max(input, lower_bound));
}

} // namespace utils
} // namespace mppi
