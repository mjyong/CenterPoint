// Device-side parameter blocks for the CUDA MPPI backend.
// Static parameters are uploaded once at backend init; cycle parameters are
// uploaded every optimization step. All values are float: proto doubles are
// narrowed exactly like the CPU optimizer does when filling
// models::OptimizerSettings.

#pragma once

#include <algorithm>
#include <cmath>
#include <exception>

#include "mppi_planner_conf.pb.h"
#include "src/algorithm/mppi_planner/models/optimizer_settings.h"
#include "src/common/util/config_helper.h"

namespace mppi {
namespace cuda {

// Motion model discriminators used on the device (matches
// MppiPlannerConf::MotionModel numbering).
enum DeviceMotionModel : int {
  kMotionDiffDrive = 0,
  kMotionOmni = 1,
  kMotionAckermann = 2,
};

// Fixed device-side capacity; configurations exceeding it fall back to CPU.
constexpr int kMaxPathPoints = 512;
constexpr int kMaxFootprintCircles = 8;

// Critic slots addressed by the device kernels (stable ids, independent of
// the configured execution order which is uploaded separately).
enum DeviceCriticId : int {
  kCriticConstraint = 0,
  kCriticGoal,
  kCriticGoalAngle,
  kCriticObstacles,
  kCriticPathAlign,
  kCriticPathAngle,
  kCriticPathFollow,
  kCriticPreferForward,
  kCriticReferenceVelocity,
  kCriticTwirling,
  kCriticVelocityDeadband,
  kNumDeviceCritics
};

struct MppiDeviceParams {
  // Rollout propagation / integration (models::ControlConstraints semantics).
  float model_dt;
  float vx_max, vx_min, vy_max, wz_max;
  float ax_max, ax_min, ay_max, az_max;

  int batch_size;
  int time_steps;
  int motion_model;   // DeviceMotionModel
  int lateral_coupling; // 0 = plain omni, 1 = lateral coupling enabled
  float lateral_blend_vx_start;
  float lateral_blend_vx_end;

  // Per-axis first-order actuation response (0 = instant response). The
  // kernels use the same response factor dt/(tau+dt) as the CPU
  // MotionModel::initialize, computed in float for exact equivalence.
  float response_tau_vx;
  float response_tau_vy;
  float response_tau_wz;

  // Static rollout parameters derived once at init.
  static MppiDeviceParams fromSettings(
      const planning::config::MppiPlannerConf &conf,
      const mppi::models::OptimizerSettings &s) {
    MppiDeviceParams p{};
    p.model_dt = s.model_dt;
    p.vx_max = s.constraints.vx_max;
    p.vx_min = s.constraints.vx_min;
    p.vy_max = s.constraints.vy;
    p.wz_max = s.constraints.wz;
    p.ax_max = s.constraints.ax_max;
    p.ax_min = s.constraints.ax_min;
    p.ay_max = s.constraints.ay_max;
    p.az_max = s.constraints.az_max;
    p.batch_size = static_cast<int>(s.batch_size);
    p.time_steps = static_cast<int>(s.time_steps);
    p.motion_model = conf.motion_model();
    p.lateral_coupling =
        (conf.motion_model() == planning::config::MppiPlannerConf::Omni &&
         conf.dynamic_constraints().lateral_coupling_enabled())
            ? 1
            : 0;
    p.lateral_blend_vx_start =
        static_cast<float>(conf.dynamic_constraints().lateral_blend_vx_start());
    p.lateral_blend_vx_end =
        static_cast<float>(conf.dynamic_constraints().lateral_blend_vx_end());
    p.response_tau_vx =
        static_cast<float>(conf.dynamic_constraints().response_tau_vx());
    p.response_tau_vy =
        static_cast<float>(conf.dynamic_constraints().response_tau_vy());
    p.response_tau_wz =
        static_cast<float>(conf.dynamic_constraints().response_tau_wz());

    // ---- Critic scoring parameters (mirrors of the CriticFunction
    // initialize() members; threshold_to_consider gates are evaluated on the
    // host every cycle and uploaded as MppiCycleParams::critic_active bits).
    p.critic_order_count = 0;
    for (int i = 0; i < kNumDeviceCritics; ++i) {
      p.critic_weight[i] = 0.0f;
      p.critic_power[i] = 1;
    }
    // Sub-message accessors return the proto default instance when unset,
    // exactly like the CPU CriticFunction::initialize() reads.
    p.critic_weight[kCriticGoal] =
        static_cast<float>(conf.goal_critic().weight());
    p.critic_power[kCriticGoal] = conf.goal_critic().power();
    p.critic_weight[kCriticGoalAngle] =
        static_cast<float>(conf.goal_angle_critic().weight());
    p.critic_power[kCriticGoalAngle] = conf.goal_angle_critic().power();
    p.critic_weight[kCriticConstraint] =
        static_cast<float>(conf.constraint_critic().weight());
    p.critic_power[kCriticConstraint] = conf.constraint_critic().power();
    p.critic_weight[kCriticPreferForward] =
        static_cast<float>(conf.prefer_forward_critic().weight());
    p.critic_power[kCriticPreferForward] =
        conf.prefer_forward_critic().power();
    {
      const auto &c = conf.reference_velocity_critic();
      p.critic_weight[kCriticReferenceVelocity] = static_cast<float>(c.weight());
      p.critic_power[kCriticReferenceVelocity] = c.power();
      p.decel_start_dist = static_cast<float>(c.decel_start_dist());
      p.reference_velocity_config = static_cast<float>(c.reference_velocity());
    }
    {
      const auto &c = conf.twirling_critic();
      p.critic_weight[kCriticTwirling] = static_cast<float>(c.weight());
      p.critic_power[kCriticTwirling] = c.power();
    }
    {
      const auto &c = conf.velocity_deadband_critic();
      p.critic_weight[kCriticVelocityDeadband] = static_cast<float>(c.weight());
      p.critic_power[kCriticVelocityDeadband] = c.power();
      p.deadband_vx = static_cast<float>(c.vx_deadband());
      p.deadband_vy = static_cast<float>(c.vy_deadband());
      p.deadband_wz = static_cast<float>(c.wz_deadband());
    }
    {
      const auto &c = conf.path_align_critic();
      p.critic_weight[kCriticPathAlign] = static_cast<float>(c.weight());
      p.critic_power[kCriticPathAlign] = c.power();
      p.max_path_occupancy_ratio =
          static_cast<float>(c.max_path_occupancy_ratio());
      p.offset_from_furthest_align = c.offset_from_furthest();
      p.trajectory_point_step = c.trajectory_point_step();
      p.use_path_orientations = c.use_path_orientations() ? 1 : 0;
    }
    {
      const auto &c = conf.path_follow_critic();
      p.critic_weight[kCriticPathFollow] = static_cast<float>(c.weight());
      p.critic_power[kCriticPathFollow] = c.power();
      p.offset_from_furthest_follow = c.offset_from_furthest();
    }
    {
      const auto &c = conf.path_angle_critic();
      p.critic_weight[kCriticPathAngle] = static_cast<float>(c.weight());
      p.critic_power[kCriticPathAngle] = c.power();
      p.max_angle_to_furthest = static_cast<float>(c.max_angle_to_furthest());
      p.offset_from_furthest_angle = c.offset_from_furthest();
      p.path_angle_mode = c.mode();
      const float vx_min_cfg = static_cast<float>(conf.dynamic_constraints().min_vx());
      const bool reversing_allowed =
          (fabsf(vx_min_cfg) >= 1e-6f) && (vx_min_cfg < 0.0f);
      if (!reversing_allowed &&
          p.path_angle_mode == 1 /* NO_DIRECTIONAL_PREFERENCE */) {
        p.path_angle_mode = 0; // FORWARD_PREFERENCE (mirrors the CPU critic)
      }
    }
    {
      const auto &c = conf.obstacles_critic();
      p.critic_weight[kCriticObstacles] =
          static_cast<float>(c.repulsion_weight());
      p.critic_power[kCriticObstacles] = c.power();
      p.critical_weight = static_cast<float>(c.critical_weight());
      p.collision_cost = static_cast<float>(c.collision_cost());
      p.collision_margin_distance =
          static_cast<float>(c.collision_margin_distance());
      p.near_goal_distance = static_cast<float>(c.near_goal_distance());
      p.repulsion_radius = static_cast<float>(c.repulsion_radius());
    }

    // ConstraintCritic::initialize speed limits (dynamic constraints).
    {
      const auto &dc = conf.dynamic_constraints();
      const float vx_max = static_cast<float>(dc.max_vx());
      const float vy_max = static_cast<float>(dc.max_vy());
      const float vx_min = static_cast<float>(dc.min_vx());
      const float min_sgn = vx_min > 0.0f ? 1.0f : -1.0f;
      p.max_vel = sqrtf(vx_max * vx_max + vy_max * vy_max);
      p.min_vel = min_sgn * sqrtf(vx_min * vx_min + vy_max * vy_max);
    }

    // Multi-circle footprint at identity pose (ObstaclesCritic circles_).
    p.fp_count = 0;
    try {
      const auto circles =
          planning::common::util::ConfigHelper::GetBoundingCircles(0.0, 0.0, 0.0);
      p.fp_count = static_cast<int>(
          std::min(circles.size(), static_cast<size_t>(kMaxFootprintCircles)));
      for (int i = 0; i < p.fp_count; ++i) {
        p.fp_x[i] = static_cast<float>(circles[i].center_x);
        p.fp_y[i] = static_cast<float>(circles[i].center_y);
        p.fp_r[i] = static_cast<float>(circles[i].radius);
      }
    } catch (const std::exception &) {
      p.fp_count = 0;
    }
    return p;
  }

  // ---- Critic scoring parameters (see fromSettings above). ----
  // Execution order of the enabled GPU critics (DeviceCriticId values in the
  // configured critic-list order) and the collision-fail blocking mask: bit k
  // set = critic k sits after ObstaclesCritic in the list and is skipped when
  // every rollout collides (CriticManager::evalTrajectoriesScores breaks on
  // data.fail_flag).
  int critic_order[kNumDeviceCritics];
  int critic_order_count;
  unsigned int post_obstacles_mask;

  float critic_weight[kNumDeviceCritics];
  int critic_power[kNumDeviceCritics];

  // ConstraintCritic
  float max_vel, min_vel;
  // ObstaclesCritic (weight slot holds repulsion_weight)
  float critical_weight, collision_cost;
  float collision_margin_distance, near_goal_distance, repulsion_radius;
  // PathAlignCritic
  float max_path_occupancy_ratio;
  int offset_from_furthest_align, trajectory_point_step;
  int use_path_orientations;
  // PathFollowCritic / PathAngleCritic
  int offset_from_furthest_follow, offset_from_furthest_angle;
  // PathAngleCritic (mode adjusted for reversing at init, 0/1/2)
  int path_angle_mode;
  float max_angle_to_furthest;
  // ReferenceVelocityCritic
  float decel_start_dist, reference_velocity_config;
  // VelocityDeadbandCritic: per-axis deadband magnitudes (fabs is applied
  // before comparison, like the CPU critic).
  float deadband_vx, deadband_vy, deadband_wz;

  // Multi-circle footprint (robot frame, identity pose)
  float fp_x[kMaxFootprintCircles];
  float fp_y[kMaxFootprintCircles];
  float fp_r[kMaxFootprintCircles];
  int fp_count;
};

// Per-cycle dynamic inputs (pose/speed/goal change every step).
struct MppiCycleParams {
  float pose_x, pose_y, pose_yaw;
  float speed_vx, speed_vy, speed_wz;
  float goal_x, goal_y, goal_yaw;
  // Externally injected reference speed (0 = use the configured value);
  // resolved to the effective value by the host and exposed via v_ref.
  float runtime_reference_velocity;
  int path_count;

  // ---- Host-evaluated critic gating and scalars (per cycle). ----
  // Bit k (DeviceCriticId) set = critic enabled and its gates passed.
  unsigned int critic_active;
  int obstacles_near_goal; // withinPositionGoalTolerance(near_goal_distance)
  float obstacles_collision_threshold; // obstacleCollisionThreshold(start footprint clearance), 0 when clear
  float v_ref;             // effective reference velocity (>0), else 0
  float goal_yaw_from_path;            // path.yaws(path_count - 1)
  float path_angle_goal_x, path_angle_goal_y, path_angle_goal_yaw;
  float path_follow_goal_x, path_follow_goal_y;
  int path_align_segments; // furthest reached path point (segment count)

  // ---- Obstacle distance field (device pointers / grid_map geometry). ----
  // Column-major linearization mirroring the grid_map layer matrix:
  // linear = j * rows + i with i the x-buffer index (x decreases as i grows)
  // and j the y-buffer index. Geometry values are doubles: the CPU sampling
  // path (GridMap::atPosition INTER_LINEAR) computes everything in double.
  const float *distance_field;
  int map_rows, map_cols;
  double map_resolution;
  double map_position_x, map_position_y; // grid map center position
  double map_length_x, map_length_y;
};

} // namespace cuda
} // namespace mppi
