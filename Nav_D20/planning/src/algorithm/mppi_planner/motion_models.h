// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). The Ackermann model now takes its
// minimum turning radius directly instead of a Parameters struct.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "src/algorithm/mppi_planner/models/constraints.h"
#include "src/algorithm/mppi_planner/models/control_sequence.h"
#include "src/algorithm/mppi_planner/models/state.h"

namespace mppi {

// Forward declaration of utils method, since utils.h can't be included here
// due to recursive inclusion.
namespace utils {
float clamp(const float lower_bound, const float upper_bound, const float input);
} // namespace utils

/**
 * @class mppi::MotionModel
 * @brief Abstract motion model for modeling a vehicle
 */
class MotionModel {
public:
  MotionModel() = default;
  virtual ~MotionModel() = default;

  /**
   * @brief Initialize motion model on bringup and set required variables.
   *        The per-axis response taus model a first-order actuation lag:
   *        actual velocity approaches the clamped command with time constant
   *        tau (0 = perfect response, the command is realized next step).
   */
  void initialize(const models::ControlConstraints &control_constraints,
                  float model_dt, float tau_vx = 0.0f, float tau_vy = 0.0f,
                  float tau_wz = 0.0f) {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
    response_alpha_vx_ = model_dt / (tau_vx + model_dt);
    response_alpha_vy_ = model_dt / (tau_vy + model_dt);
    response_alpha_wz_ = model_dt / (tau_wz + model_dt);
  }

  /**
   * @brief With input velocities, find the vehicle's output velocities
   */
  virtual void predict(models::State &state) {
    const bool is_holo = isHolonomic();
    float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    float min_delta_vx = model_dt_ * control_constraints_.ax_min;
    float max_delta_vy = model_dt_ * control_constraints_.ay_max;
    float max_delta_wz = model_dt_ * control_constraints_.az_max;

    unsigned int n_rows = state.vx.rows();
    unsigned int n_cols = state.vx.cols();

    // Default layout in eigen is column-major, hence accessing elements in
    // column-major fashion to utilize L1 cache as much as possible
    for (unsigned int i = 1; i != n_cols; i++) {
      for (unsigned int j = 0; j != n_rows; j++) {
        float vx_last = state.vx(j, i - 1);
        float &cvx_curr = state.cvx(j, i - 1);
        cvx_curr = utils::clamp(vx_last + min_delta_vx, vx_last + max_delta_vx, cvx_curr);
        // First-order response: v_next = v_last + alpha * (v_cmd - v_last);
        // alpha = 1 when tau = 0 (original instant-response behavior).
        state.vx(j, i) = vx_last + response_alpha_vx_ * (cvx_curr - vx_last);

        float wz_last = state.wz(j, i - 1);
        float &cwz_curr = state.cwz(j, i - 1);
        cwz_curr = utils::clamp(wz_last - max_delta_wz, wz_last + max_delta_wz, cwz_curr);
        state.wz(j, i) = wz_last + response_alpha_wz_ * (cwz_curr - wz_last);

        if (is_holo) {
          float vy_last = state.vy(j, i - 1);
          float &cvy_curr = state.cvy(j, i - 1);
          cvy_curr = utils::clamp(vy_last - max_delta_vy, vy_last + max_delta_vy, cvy_curr);
          state.vy(j, i) = vy_last + response_alpha_vy_ * (cvy_curr - vy_last);
        }
      }
    }
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   */
  virtual bool isHolonomic() = 0;

  /**
   * @brief Constrain the optimized command sequence against dynamic
   *        feasibility, propagating the same first-order response lag as
   *        predict(): acceleration boxes are evaluated around the *predicted
   *        actual* velocity chain starting from the measured speed, and the
   *        lateral coupling cap (omni) is evaluated at the lagged
   *        longitudinal speed. Commands, rollouts and the real robot thus
   *        share one reference velocity chain.
   */
  virtual void constrainCommandSequence(models::ControlSequence &cs,
                                        const models::Velocity &speed) {
    const float dt = model_dt_;
    const float max_dvx = dt * control_constraints_.ax_max;
    const float min_dvx = dt * control_constraints_.ax_min;
    const float max_dvy = dt * control_constraints_.ay_max;
    const float max_dwz = dt * control_constraints_.az_max;

    float vx_actual = speed.vx;
    float wz_actual = speed.wz;
    float vy_actual = speed.vy;
    const bool holo = isHolonomic();

    for (int i = 0; i < cs.vx.size(); ++i) {
      // Longitudinal: speed box, then acceleration box around the predicted
      // actual velocity (same clamp semantics as predict()).
      float vx_cmd = utils::clamp(control_constraints_.vx_min,
                                  control_constraints_.vx_max, cs.vx(i));
      vx_cmd = utils::clamp(vx_actual + min_dvx, vx_actual + max_dvx, vx_cmd);
      cs.vx(i) = vx_cmd;
      vx_actual += response_alpha_vx_ * (vx_cmd - vx_actual);

      float wz_cmd = utils::clamp(-control_constraints_.wz,
                                  control_constraints_.wz, cs.wz(i));
      wz_cmd = utils::clamp(wz_actual - max_dwz, wz_actual + max_dwz, wz_cmd);
      cs.wz(i) = wz_cmd;
      wz_actual += response_alpha_wz_ * (wz_cmd - wz_actual);

      if (holo) {
        float vy_cmd = utils::clamp(-control_constraints_.vy,
                                    control_constraints_.vy, cs.vy(i));
        vy_cmd = clampVyCmd(vy_cmd, vy_actual, max_dvy, vx_actual);
        cs.vy(i) = vy_cmd;
        vy_actual += response_alpha_vy_ * (vy_cmd - vy_actual);
      }
    }

    // Model-specific hard constraints on the commands themselves (e.g.
    // Ackermann curvature); commands are in the executable domain here.
    applyConstraints(cs);
  }

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   */
  virtual void applyConstraints(models::ControlSequence & /*control_sequence*/) {}

protected:
  /**
   * @brief Lateral command clamp for one step of constrainCommandSequence:
   *        acceleration box around the predicted actual lateral velocity.
   *        Omni with lateral coupling overrides this with the joint box
   *        capped at the lagged longitudinal speed.
   */
  virtual float clampVyCmd(float vy_cmd, float vy_last, float max_dvy,
                           float /*vx_actual*/) {
    return utils::clamp(vy_last - max_dvy, vy_last + max_dvy, vy_cmd);
  }

  float model_dt_{0.0};
  models::ControlConstraints control_constraints_{
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  // Per-axis first-order response factors dt/(tau+dt); 1.0 = instant response.
  float response_alpha_vx_{1.0f};
  float response_alpha_vy_{1.0f};
  float response_alpha_wz_{1.0f};
};

/**
 * @class mppi::AckermannMotionModel
 * @brief Ackermann motion model
 */
class AckermannMotionModel : public MotionModel {
public:
  explicit AckermannMotionModel(float min_turning_r) : min_turning_r_(min_turning_r) {}

  bool isHolonomic() override { return false; }

  void applyConstraints(models::ControlSequence &control_sequence) override {
    const auto vx_ptr = control_sequence.vx.data();
    auto wz_ptr = control_sequence.wz.data();
    int steps = control_sequence.vx.size();
    for (int i = 0; i < steps; i++) {
      float wz_constrained = fabs(*(vx_ptr + i) / min_turning_r_);
      float &wz_curr = *(wz_ptr + i);
      wz_curr = utils::clamp(-1 * wz_constrained, wz_constrained, wz_curr);
    }
  }

  float getMinTurningRadius() { return min_turning_r_; }

private:
  float min_turning_r_{0};
};

/**
 * @class mppi::DiffDriveMotionModel
 * @brief Differential drive motion model
 */
class DiffDriveMotionModel : public MotionModel {
public:
  DiffDriveMotionModel() = default;

  bool isHolonomic() override { return false; }
};

/**
 * @class mppi::OmniMotionModel
 * @brief Omnidirectional motion model, optionally with lateral coupling:
 *        the lateral authority |vy| decays linearly with |vx|, degenerating
 *        into a diff-drive model at speed. This keeps the full omni
 *        maneuverability for slow operations (docking, narrow passages) while
 *        forcing turn-to-drive behavior at cruise speed.
 */
class OmniMotionModel : public MotionModel {
public:
  OmniMotionModel() = default;

  explicit OmniMotionModel(
      float lateral_blend_vx_start, float lateral_blend_vx_end)
      : lateral_coupling_enabled_(true),
        lateral_blend_vx_start_(lateral_blend_vx_start),
        lateral_blend_vx_end_(lateral_blend_vx_end) {}

  bool isHolonomic() override { return true; }

  void predict(models::State &state) override {
    const bool coupled = lateral_coupling_enabled_ &&
                         lateral_blend_vx_end_ > lateral_blend_vx_start_;
    float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    float min_delta_vx = model_dt_ * control_constraints_.ax_min;
    float max_delta_vy = model_dt_ * control_constraints_.ay_max;
    float max_delta_wz = model_dt_ * control_constraints_.az_max;

    unsigned int n_rows = state.vx.rows();
    unsigned int n_cols = state.vx.cols();

    for (unsigned int i = 1; i != n_cols; i++) {
      for (unsigned int j = 0; j != n_rows; j++) {
        float vx_last = state.vx(j, i - 1);
        float &cvx_curr = state.cvx(j, i - 1);
        cvx_curr = utils::clamp(vx_last + min_delta_vx, vx_last + max_delta_vx, cvx_curr);
        state.vx(j, i) = vx_last + response_alpha_vx_ * (cvx_curr - vx_last);

        float wz_last = state.wz(j, i - 1);
        float &cwz_curr = state.cwz(j, i - 1);
        cwz_curr = utils::clamp(wz_last - max_delta_wz, wz_last + max_delta_wz, cwz_curr);
        state.wz(j, i) = wz_last + response_alpha_wz_ * (cwz_curr - wz_last);

        float vy_last = state.vy(j, i - 1);
        float &cvy_curr = state.cvy(j, i - 1);
        if (coupled) {
          // Joint box: acceleration limit AND the coupling cap, so rollouts
          // never contain samples violating the coupling — the scored
          // trajectories stay identical to the executable ones.
          const float vy_cap = lateralVyCap(std::fabs(state.vx(j, i)));
          float vy_hi = utils::clamp(0.0f, vy_cap, vy_last + max_delta_vy);
          float vy_lo = utils::clamp(-vy_cap, 0.0f, vy_last - max_delta_vy);
          cvy_curr = utils::clamp(vy_lo, vy_hi, cvy_curr);
        } else {
          // Identical to the base model: acceleration limit only (rollouts
          // may exceed the vy box; ConstraintCritic penalizes that softly).
          cvy_curr =
              utils::clamp(vy_last - max_delta_vy, vy_last + max_delta_vy, cvy_curr);
        }
        state.vy(j, i) = vy_last + response_alpha_vy_ * (cvy_curr - vy_last);
      }
    }
  }

  float clampVyCmd(float vy_cmd, float vy_last, float max_dvy,
                   float vx_actual) override {
    if (!lateral_coupling_enabled_ ||
        lateral_blend_vx_end_ <= lateral_blend_vx_start_) {
      return utils::clamp(vy_last - max_dvy, vy_last + max_dvy, vy_cmd);
    }
    // Joint box mirroring predict(): acceleration limit AND the coupling cap
    // evaluated at the lagged longitudinal speed, so the output commands
    // never claim more lateral authority than the rollouts assumed.
    const float vy_cap = lateralVyCap(std::fabs(vx_actual));
    const float vy_hi = utils::clamp(0.0f, vy_cap, vy_last + max_dvy);
    const float vy_lo = utils::clamp(-vy_cap, 0.0f, vy_last - max_dvy);
    return utils::clamp(vy_lo, vy_hi, vy_cmd);
  }

private:
  /**
   * @brief Lateral velocity authority at the given longitudinal speed:
   *        full control_constraints_.vy below blend start, zero above blend
   *        end, linear in between.
   */
  float lateralVyCap(float abs_vx) const {
    if (abs_vx <= lateral_blend_vx_start_) {
      return control_constraints_.vy;
    }
    if (abs_vx >= lateral_blend_vx_end_) {
      return 0.0f;
    }
    const float t =
        (abs_vx - lateral_blend_vx_start_) /
        (lateral_blend_vx_end_ - lateral_blend_vx_start_);
    return control_constraints_.vy * (1.0f - t);
  }

  bool lateral_coupling_enabled_{false};
  float lateral_blend_vx_start_{0.3f};
  float lateral_blend_vx_end_{1.2f};
};

} // namespace mppi
