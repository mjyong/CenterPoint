#pragma once

#include <memory>

#include "src/runnable/mission/mission_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base.h"
#include "src/runnable/mission/mission_state_machine/mission_fsm_base_ctx.h"

namespace planning {
namespace mission {
class Skill : public MissionBase {
  enum class SkillType {
    kNone = 0,
    kMoving = 1,
    kRotating = 2,
  };
  struct SkillInfo {
    struct MovingSkill {
      double odom_x;
      double odom_y;
      double odom_yaw;
      void reset() {
        odom_x = 0.0;
        odom_y = 0.0;
        odom_yaw = 0.0;
      }
    };
    struct RotatingSkill {
      double requested_rotation;
      double accumulated_yaw;
      double last_yaw;
      void reset() {
        requested_rotation = 0.0;
        accumulated_yaw = 0.0;
        last_yaw = 0.0;
      }
    };
    SkillType skill_type{SkillType::kNone};
    bool complete{false};
    union {
      MovingSkill moving_skill;
      RotatingSkill rotating_skill;
    };
  };

public:
  Skill(const MissionConf::Conf &conf);
  ~Skill();
  bool Init(PlanningEntry *planning_entry) override;
  bool Proc() override;
  const MissionFSMBase::StateType &mission_status() override {
    return mission_state_;
  }
  void set_moving_skill(double delta_yaw, double dist) {
    skill_info_.skill_type = SkillType::kMoving;
    skill_info_.complete = false;
    skill_info_.moving_skill.reset();
    const double dx = dist * std::cos(delta_yaw);
    const double dy = dist * std::sin(delta_yaw);
    const auto *odom =
        planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
    skill_info_.moving_skill.odom_x = odom->pose.position.x() + dx;
    skill_info_.moving_skill.odom_y = odom->pose.position.y() + dy;
    skill_info_.moving_skill.odom_yaw =
        common::util::normalize_angle(odom->pose.euler().z() + delta_yaw);
  }

  void set_rotating_skill(double accumulated_yaw) {
    skill_info_.skill_type = SkillType::kRotating;
    skill_info_.complete = false;
    skill_info_.rotating_skill.reset();
    skill_info_.rotating_skill.requested_rotation = accumulated_yaw;
    const auto *odom =
        planning_entry_->communication_entry_ptr()->MainBridge()->odometry();
    skill_info_.rotating_skill.last_yaw = odom->pose.euler().z();
  }

private:
  Twist ProcMovingSkill();
  Twist ProcRotatingSkill();

  // friend class MissionFSMBaseCtx;
  bool check_e_off_passive() override;
  bool check_e_off_fault() override;
  bool check_e_passive_standby() override;
  bool check_e_passive_fault() override;
  bool check_e_standby_active() override;
  bool check_e_standby_passive() override;
  bool check_e_standby_fault() override;
  bool check_e_standby_off() override;
  bool check_e_active_standstill() override;
  bool check_e_active_standby() override;
  bool check_e_active_fault() override;
  bool check_e_active_off() override;
  bool check_e_standstill_active() override;
  bool check_e_standstill_standby() override;
  bool check_e_standstill_fault() override;
  bool check_e_standstill_off() override;
  bool check_e_fault_off() override;
  bool check_e_fault_passive() override;
  void transition_e_off_passive() override;
  void transition_e_off_fault() override;
  void transition_e_passive_standby() override;
  void transition_e_passive_fault() override;
  void transition_e_standby_active() override;
  void transition_e_standby_passive() override;
  void transition_e_standby_fault() override;
  void transition_e_active_standstill() override;
  void transition_e_active_standby() override;
  void transition_e_active_fault() override;
  void transition_e_active_off() override;
  void transition_e_standstill_active() override;
  void transition_e_standstill_standby() override;
  void transition_e_standstill_fault() override;
  void transition_e_fault_off() override;
  void transition_e_fault_passive() override;
  void spin_s_off() override;
  void spin_s_passive() override;
  void spin_s_standby() override;
  void spin_s_active() override;
  void spin_s_standstill() override;
  void spin_s_fault() override;
  void on_entry_s_off() override;
  void on_entry_s_passive() override;
  void on_entry_s_standby() override;
  void on_entry_s_active() override;
  void on_entry_s_standstill() override;
  void on_entry_s_fault() override;
  void on_exit_s_off() override;
  void on_exit_s_passive() override;
  void on_exit_s_standby() override;
  void on_exit_s_active() override;
  void on_exit_s_standstill() override;
  void on_exit_s_fault() override;

private:
  bool init_flag_{false};
  MissionFSMBaseCtx mission_fsm_ctx_;
  MissionFSMBase::StateType mission_state_;
  SkillInfo skill_info_;
};
} // namespace mission
} // namespace planning