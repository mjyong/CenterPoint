/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-23 13:59:55
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */
#include "src/runnable/mission/recharging/recharging.h"
#include "src/common/util/diagnostic.h"
#include "src/common/util/math.h"
#include "src/runnable/entry/planning_entry.h"
#include <mc_interface/msg/motion_skill.hpp>

using planning::common::util::Diagnostic;

namespace planning {
namespace mission {

Recharging::Recharging(const MissionConf::Conf &conf)
    : mission_fsm_ctx_(this), recharging_fsm_ctx_(this) {
  conf_ = conf;
}

Recharging::~Recharging() {}

bool Recharging::Init(PlanningEntry *planning_entry) {
  planning_entry_ = planning_entry;
  ACHECK(planning_entry_ != nullptr) << "planning_entry_ is nullptr";
  init_flag_ = true;
  return true;
}

bool Recharging::Proc() {
  ready_position_ = ReadyPosition();
  mission_fsm_ctx_.Proc();
  mission_state_ = mission_fsm_ctx_.current_state();
  AINFO_EVERY(20) << "mission_state: " << mission_fsm_ctx_.current_state_name()
                  << ", recharging_state: "
                  << recharging_fsm_ctx_.current_state_name();
  return true;
}

Pose3dStamped Recharging::ReadyPosition() {
  const auto pos = charging_station_.pose.position;
  const double heading = charging_station_.pose.euler()[2];
  const Eigen::Vector2d unit_vec = {std::cos(heading), std::sin(heading)};
  const auto ready_pos =
      pos.head<2>() - conf_.recharging_conf().ready_distance() * unit_vec;

  return Pose3dStamped(Pose3d(Point3d(ready_pos.x(), ready_pos.y(), pos.z()),
                              charging_station_.pose.orientation),
                       charging_station_.timestamp);
}

bool Recharging::IsArrived(const Pose3d &relative_target,
                           double pos_x_tolerance, double pos_y_tolerance,
                           double euler_tolerance, double stable_time) {
  Eigen::Vector2d pos_diff = relative_target.position.head<2>().cwiseAbs();
  bool is_within_position_tolerance =
      pos_diff.x() < pos_x_tolerance && pos_diff.y() < pos_y_tolerance;

  bool is_within_angle_tolerance =
      std::abs(relative_target.euler()[2]) < euler_tolerance;

  bool is_arrived = is_within_position_tolerance && is_within_angle_tolerance;

  static size_t stable_count = 0;
  stable_count = is_arrived ? stable_count + 1 : 0;

  return stable_count * conf_.recharging_conf().period() > stable_time;
}

bool Recharging::IsDetected() {
  return fabs(planning_entry_->timestamp() - charging_station_.timestamp) < 1.0;
}

// ==================== 充电业务反馈与动作指令 ====================
void Recharging::UpdateBatteryState(const BatteryState &state) {
  battery_state_ = state;
}

void Recharging::UpdateMotionStates(const std::vector<MotionState> &states) {
  motion_states_ = states;
}

BatteryState Recharging::battery_state() const { return battery_state_; }

bool Recharging::IsMotionDone(uint8_t motion_id) const {
  for (const auto &ms : motion_states_) {
    if (ms.motion == motion_id) {
      // state: 1=REJECTED, 2=DOING, 3=DONE, 4=ABORTED
      // 仅接受指令排队之后收到的反馈，防上一轮残留 DONE 误判
      bool done = (ms.state == 3) && (ms.timestamp >= last_motion_send_time_);
      AINFO << "[IsMotionDone] motion=" << static_cast<int>(motion_id)
            << " state=" << static_cast<int>(ms.state) << " => "
            << (done ? "DONE" : "NOT_DONE");
      return done;
    }
  }
  return false;
}

void Recharging::QueueMotionSkill(uint8_t motion,
                                  const std::vector<float> &para) {
  if (has_pending_motion_) {
    AWARN << "Pending motion " << static_cast<int>(pending_motion_)
          << " overwritten by " << static_cast<int>(motion);
  }
  pending_motion_ = motion;
  pending_para_ = para;
  has_pending_motion_ = true;
  last_motion_send_time_ = TimeKeeper::TimeNowSec();
  AINFO << "[QueueMotionSkill] motion=" << static_cast<int>(motion);
}

bool Recharging::TakePendingMotion(uint8_t &motion, std::vector<float> &para) {
  if (!has_pending_motion_) {
    return false;
  }
  has_pending_motion_ = false;
  motion = pending_motion_;
  para = std::move(pending_para_);
  return true;
}

double Recharging::InnerStateElapsed() const {
  return TimeKeeper::TimeNowSec() - inner_state_enter_time_;
}

bool Recharging::TakeRetry() {
  if (retry_count_ < max_retry_count_) {
    ++retry_count_;
    AINFO << "Take retry " << retry_count_ << "/" << max_retry_count_;
    return true;
  }
  AERROR << "Retry exhausted (" << retry_count_ << "/" << max_retry_count_
         << "), request fault";
  return false;
}

TwistStamped Recharging::FineTuning(const Pose3d &relative_target) {
  double dx = relative_target.position.x();
  double dy = relative_target.position.y();
  double dtheta = common::util::normalize_angle(relative_target.euler()[2]);

  AINFO << "dx: " << dx << ", dy: " << dy << ", dtheta: " << dtheta;

  double dt = conf_.recharging_conf().period();

  // PID control for vx
  if (std::abs(dx) < conf_.recharging_conf().vx_deadband()) {
    pid_vx_integral_ += dx * dt;
  } else {
    pid_vx_integral_ = 0;
  }
  double vx_derivative = (dx - pid_vx_prev_error_) / dt;
  pid_vx_prev_error_ = dx;
  double vx = conf_.recharging_conf().vx_gain() * dx +
              conf_.recharging_conf().vx_ki() * pid_vx_integral_ +
              conf_.recharging_conf().vx_kd() * vx_derivative;
  vx = std::clamp(vx, conf_.recharging_conf().vx_min(),
                  conf_.recharging_conf().vx_max());

  // PID control for vy
  if (std::abs(dy) < conf_.recharging_conf().vy_deadband()) {
    pid_vy_integral_ += dy * dt;
  }
  double vy_derivative = (dy - pid_vy_prev_error_) / dt;
  pid_vy_prev_error_ = dy;
  double vy = conf_.recharging_conf().vy_gain() * dy +
              conf_.recharging_conf().vy_ki() * pid_vy_integral_ +
              conf_.recharging_conf().vy_kd() * vy_derivative;
  vy = std::clamp(vy, -conf_.recharging_conf().vy_max(),
                  conf_.recharging_conf().vy_max());

  // PID control for yaw_rate
  if (std::abs(dtheta) < conf_.recharging_conf().yaw_rate_deadband()) {
    pid_yaw_integral_ += dtheta * dt;
  }
  double yaw_derivative = (dtheta - pid_yaw_prev_error_) / dt;
  pid_yaw_prev_error_ = dtheta;
  double yaw_rate = conf_.recharging_conf().yaw_rate_gain() * dtheta +
                    conf_.recharging_conf().yaw_rate_ki() * pid_yaw_integral_ +
                    conf_.recharging_conf().yaw_rate_kd() * yaw_derivative;
  yaw_rate = std::clamp(yaw_rate, -conf_.recharging_conf().yaw_rate_max(),
                        conf_.recharging_conf().yaw_rate_max());

  return TwistStamped(
      Twist(Eigen::Vector3d(vx, vy, 0.0), Eigen::Vector3d(0.0, 0.0, yaw_rate)),
      TimeKeeper::TimeNowSec());
}

// ==================== 外层状态机钩子 ====================
// ---- check ----
bool Recharging::check_e_off_passive() { return init_flag_ && need_start(); }
bool Recharging::check_e_passive_standby() {
  return recharging_fsm_ctx_.passive_to_standby();
}
bool Recharging::check_e_passive_fault() {
  return recharging_fsm_ctx_.to_fault();
}
bool Recharging::check_e_passive_off() { return need_stop(); }
bool Recharging::check_e_standby_active() {
  return recharging_fsm_ctx_.standby_to_active();
}
bool Recharging::check_e_standby_fault() {
  return recharging_fsm_ctx_.to_fault();
}
bool Recharging::check_e_standby_off() { return need_stop(); }
bool Recharging::check_e_active_standstill() {
  if (need_pause()) {
    return true;
  }
  // 仅进桩运动阶段因桩数据持续失效暂停
  // if (recharging_fsm_ctx_.current_state() ==
  //     RechargingFSMBase::StateType::inserting) {
  //   if (IsDetected()) {
  //     data_lost_start_ = 0.0;
  //   } else {
  //     const auto now = TimeKeeper::TimeNowSec();
  //     if (data_lost_start_ <= 0.0) {
  //       data_lost_start_ = now;
  //     } else if (now - data_lost_start_ > kDataLostPauseTime) {
  //       return true;
  //     }
  //   }
  // } else {
  //   data_lost_start_ = 0.0;
  // }
  return false;
}
bool Recharging::check_e_active_off() {
  return recharging_fsm_ctx_.active_to_off() || need_stop();
}
bool Recharging::check_e_active_fault() {
  return recharging_fsm_ctx_.to_fault();
}
bool Recharging::check_e_standstill_active() {
  return need_resume() || (pause_by_data_loss_ && IsDetected());
}
bool Recharging::check_e_standstill_off() { return need_stop(); }
bool Recharging::check_e_fault_off() { return need_stop(); }
bool Recharging::check_e_fault_passive() { return need_start(); }

// ---- transition ----
void Recharging::transition_e_off_passive() { reset_trigger(); }
void Recharging::transition_e_passive_standby() {
  recharging_fsm_ctx_.ClearPassiveToStandby();
}
void Recharging::transition_e_passive_fault() { recharging_fsm_ctx_.Reset(); }
void Recharging::transition_e_passive_off() {
  AINFO << "Recharging cancelled in passive, switch to off";
  recharging_fsm_ctx_.Reset();
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Recharging::transition_e_standby_active() {
  recharging_fsm_ctx_.ClearStandbyToActive();
}
void Recharging::transition_e_standby_off() {
  recharging_fsm_ctx_.Reset();
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Recharging::transition_e_standby_fault() { recharging_fsm_ctx_.Reset(); }
void Recharging::transition_e_active_standstill() {
  pause_by_data_loss_ = !need_pause();
  data_lost_start_ = 0.0;
  AINFO << "Active paused"
        << (pause_by_data_loss_ ? " by charging data loss"
                                : " by external trigger");
  reset_trigger();
}
void Recharging::transition_e_active_off() {
  AINFO << "Recharging finished, switch to off";
  recharging_fsm_ctx_.Reset();
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Recharging::transition_e_active_fault() {
  AERROR << "Recharging failed, switch to fault";
  recharging_fsm_ctx_.Reset();
}
void Recharging::transition_e_standstill_active() { reset_trigger(); }
void Recharging::transition_e_standstill_off() {
  recharging_fsm_ctx_.Reset();
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Recharging::transition_e_fault_off() {
  AINFO << "Recharging cancelled in fault, switch to off";
  recharging_fsm_ctx_.Reset();
  reset_trigger();
  planning_entry_->mission_entry_ptr()->SetCurrentMissionType(
      MissionConf::Idle);
}
void Recharging::transition_e_fault_passive() {
  AINFO << "Recharging restart from fault";
  recharging_fsm_ctx_.Reset();
  reset_trigger();
}

// ---- spin ----
void Recharging::spin_s_off() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
}
void Recharging::spin_s_passive() { recharging_fsm_ctx_.Proc(); }
void Recharging::spin_s_standby() { recharging_fsm_ctx_.Proc(); }
void Recharging::spin_s_active() { recharging_fsm_ctx_.Proc(); }
void Recharging::spin_s_standstill() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
}
void Recharging::spin_s_fault() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
  if (searching_timeout_) {
    Diagnostic::Instance()->emplace("0x04040009");
  } else {
    Diagnostic::Instance()->emplace("0x0404000A");
  }
}

// ---- on_entry ----
void Recharging::on_entry_s_passive() {
  AINFO << "Recharging start, flow: "
        << (back_stake_ ? "back stake" : "charge");
  recharging_fsm_ctx_.Start();
}
void Recharging::on_entry_s_active() {
  ResetPid();
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  if (recharging_fsm_ctx_.current_state() ==
      RechargingFSMBase::StateType::idle) {
    recharging_fsm_ctx_.Start();
  }
}
void Recharging::on_entry_s_standstill() {
  AINFO << "Enter standstill (paused)";
}
void Recharging::on_entry_s_fault() {
  pause_by_data_loss_ = false;
  data_lost_start_ = 0.0;
}

// ==================== 内层子状态机钩子 ====================
// ---- check ----
bool Recharging::check_e_detected() { return IsDetected(); }
bool Recharging::check_e_searching_timeout() {
  searching_timeout_ =
      InnerStateElapsed() > conf_.recharging_conf().time_out_threshold();
  return searching_timeout_;
}
bool Recharging::check_e_aligning_arrived() {
  return IsArrived(ready_position_.pose,
                   conf_.recharging_conf().standby_pos_tolerance(),
                   conf_.recharging_conf().standby_pos_tolerance(),
                   conf_.recharging_conf().standby_euler_tolerance(),
                   conf_.recharging_conf().standby_stable_time());
}
bool Recharging::check_e_aligning_timeout() {
  return InnerStateElapsed() > conf_.recharging_conf().time_out_threshold();
}
bool Recharging::check_e_inserting_arrived() {
  return IsArrived(charging_station_.pose,
                   conf_.recharging_conf().standstill_pos_x_tolerance(),
                   conf_.recharging_conf().standstill_pos_y_tolerance(),
                   conf_.recharging_conf().standstill_euler_tolerance(),
                   conf_.recharging_conf().standstill_stable_time());
}
bool Recharging::check_e_inserting_timeout() {
  return InnerStateElapsed() > conf_.recharging_conf().time_out_threshold();
}
bool Recharging::check_e_charge_ok() {
  const auto battery = battery_state();
  return battery.valid && battery.charge_state == 1;
}
bool Recharging::check_e_charge_fail() {
  return InnerStateElapsed() > kChargeWaitTimeout;
}
bool Recharging::check_e_motion_done() {
  return IsMotionDone(mc_interface::msg::MotionSkill::MOTION_GAIT_WALK);
}
bool Recharging::check_e_backoff_arrived() {
  return IsArrived(ready_position_.pose,
                   conf_.recharging_conf().standby_pos_tolerance(),
                   conf_.recharging_conf().standby_pos_tolerance(),
                   conf_.recharging_conf().standby_euler_tolerance(),
                   conf_.recharging_conf().standby_stable_time());
}
bool Recharging::check_e_standing_up_timeout() {
  return InnerStateElapsed() > conf_.recharging_conf().time_out_threshold();
}
bool Recharging::check_e_backoff_timeout() {
  return InnerStateElapsed() > conf_.recharging_conf().time_out_threshold();
}

// ---- transition ----
void Recharging::transition_e_start_charge() {
  AINFO << "Charge flow start: searching charging station";
}
void Recharging::transition_e_start_back_stake() {
  AINFO << "Back stake flow start: standing up";
}
void Recharging::transition_e_detected() {
  AINFO << "Charging station detected";
}
void Recharging::transition_e_arrived_ready() {
  AINFO << "Aligned at ready position";
}
void Recharging::transition_e_arrived_station() {
  AINFO << "Inserted at charging station";
}
void Recharging::transition_e_retry() {
  AINFO << "Inserting timeout, stand up for retry";
}
void Recharging::transition_e_charge_ok() {
  AINFO << "Charge confirmed, charge flow done";
}
void Recharging::transition_e_charge_fail() {
  AINFO << "Charge failed, stand up for retry";
}
void Recharging::transition_e_motion_done() { AINFO << "Stand up done"; }
void Recharging::transition_e_backoff_retry_done() {
  AINFO << "Backed off to ready position, retry aligning";
}
void Recharging::transition_e_backoff_finish() {
  AINFO << "Backed off, back stake flow done";
}
void Recharging::transition_e_reset() {}

// ---- spin ----
void Recharging::spin_s_idle() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
}
void Recharging::spin_s_searching() {
  twist_ = TwistStamped(
      Twist(Eigen::Vector3d::Zero(),
            Eigen::Vector3d(0.0, 0.0,
                            conf_.recharging_conf().detect_spin_rate())),
      TimeKeeper::TimeNowSec());
}
void Recharging::spin_s_aligning() {
  twist_ = FineTuning(ready_position_.pose);
}
void Recharging::spin_s_inserting() {
  twist_ = FineTuning(charging_station_.pose);
}
void Recharging::spin_s_sitting() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
  // 动作序列：先控制身体高度下降（MOTION_BODY_TARGET_POSE），
  // 完成后下发蹲下（MOTION_SQUAT），蹲下完成后下发趴下（MOTION_LIE_BACK）
  if (!squat_cmd_sent_ &&
      IsMotionDone(mc_interface::msg::MotionSkill::MOTION_BODY_TARGET_POSE)) {
    squat_cmd_sent_ = true;
    QueueMotionSkill(mc_interface::msg::MotionSkill::MOTION_SQUAT);
  }
  if (!lie_cmd_sent_ &&
      IsMotionDone(mc_interface::msg::MotionSkill::MOTION_SQUAT)) {
    lie_cmd_sent_ = true;
    QueueMotionSkill(mc_interface::msg::MotionSkill::MOTION_LIE_BACK);
  }
  const auto battery = battery_state();
  if (battery.valid) {
    AINFO << "Battery in sitting: charge_state="
          << static_cast<int>(battery.charge_state) << ", rsoc=" << battery.rsoc
          << "%, voltage=" << battery.voltage
          << "V, current=" << battery.current << "A";
  }
}
void Recharging::spin_s_standing_up() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
}
void Recharging::spin_s_backing_off() {
  twist_ = FineTuning(ready_position_.pose);
}
void Recharging::spin_s_done() {
  twist_ = TwistStamped(Twist(), TimeKeeper::TimeNowSec());
}

// ---- on_entry ----
void Recharging::on_entry_s_idle() {
  retry_count_ = 0;
  pause_by_data_loss_ = false;
  data_lost_start_ = 0.0;
  lie_cmd_sent_ = false;
  squat_cmd_sent_ = false;
  searching_timeout_ = false;
  // 清残留待发指令，避免上一任务停止前的排队指令在下个任务误发
  has_pending_motion_ = false;
  pending_para_.clear();
}
void Recharging::on_entry_s_searching() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
}
void Recharging::on_entry_s_aligning() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  ResetPid();
}
void Recharging::on_entry_s_inserting() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  ResetPid();
}
void Recharging::on_entry_s_sitting() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  // 动作序列起点：先下发身体目标姿态控制（MOTION_BODY_TARGET_POSE）控制高度下降，
  // 完成后由 spin_s_sitting
  // 依次下发蹲下（MOTION_SQUAT）和趴下（MOTION_LIE_BACK）
  lie_cmd_sent_ = false;
  squat_cmd_sent_ = false;
  QueueMotionSkill(
      mc_interface::msg::MotionSkill::MOTION_BODY_TARGET_POSE,
      {0.0f, 0.0f, static_cast<float>(sitting_height()), 0.0f, 0.0f, 0.0f});
  AINFO << "Enter sitting, lower body height by " << sitting_height()
        << "m, wait for charge (timeout " << kChargeWaitTimeout << "s)";
}
void Recharging::on_entry_s_standing_up() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  QueueMotionSkill(mc_interface::msg::MotionSkill::MOTION_STAND);
}
void Recharging::on_entry_s_backing_off() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
  ResetPid();
}
void Recharging::on_entry_s_done() {
  inner_state_enter_time_ = TimeKeeper::TimeNowSec();
}

// ---- on_exit ----
void Recharging::on_exit_s_idle() {}
void Recharging::on_exit_s_searching() {}
void Recharging::on_exit_s_aligning() {}
void Recharging::on_exit_s_inserting() {}
void Recharging::on_exit_s_sitting() {}
void Recharging::on_exit_s_standing_up() {}
void Recharging::on_exit_s_backing_off() {}
void Recharging::on_exit_s_done() {}

} // namespace mission
} // namespace planning
