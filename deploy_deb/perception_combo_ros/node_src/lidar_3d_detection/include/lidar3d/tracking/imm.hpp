// IMM filter with constant-velocity and coordinated-turn models (tracking/imm.py).
//
// Shared state x = [px, py, vx, vy, omega] (world frame). The filter is the tracker's state estimator and
// the tier-1 predictor (rollout).
#pragma once

#include <Eigen/Dense>
#include <vector>

#include "lidar3d/config.hpp"

namespace lidar3d {

constexpr int kStateDim = 5;
using StateVec = Eigen::Matrix<double, kStateDim, 1>;
using StateMat = Eigen::Matrix<double, kStateDim, kStateDim>;

struct MotionModel {
  enum Kind { CV, CT } kind = CV;
  double q_acc = 1.0;
  double q_omega = 1e-3;
  static MotionModel from_spec(const MotionModelSpec& s);
  void predict(const StateVec& x, const StateMat& P, double dt, StateVec& x_out, StateMat& P_out) const;
};

struct Rollout {
  // per model on its own: means[j][k], covs[j][k] at times[k]
  std::vector<std::vector<StateVec>> means;
  std::vector<std::vector<StateMat>> covs;
  std::vector<double> mode_probs;
  std::vector<double> times;
};

class IMM {
 public:
  IMM() = default;
  IMM(std::vector<MotionModel> models, double stay_prob = 0.95, double dt_ref = 0.1);
  void initialize(const StateVec& x0, const StateMat& P0);
  Eigen::MatrixXd transition(double dt) const;
  void predict(double dt);
  // measurement z = H x + noise(R); returns the log likelihood
  double update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
  void state(StateVec& x, StateMat& P) const;    // moment-matched mixture
  StateVec mean() const;
  Rollout rollout(double horizon, double step) const;   // per-mode open-loop prediction (mixed=False)

  int num_models() const { return static_cast<int>(models_.size()); }
  const std::vector<double>& mu() const { return mu_; }
  const std::vector<StateVec>& x() const { return x_; }
  const std::vector<StateMat>& P() const { return P_; }

 private:
  std::vector<MotionModel> models_;
  Eigen::MatrixXd Pi_ref_;
  double dt_ref_ = 0.1;
  std::vector<double> mu_;
  std::vector<StateVec> x_;
  std::vector<StateMat> P_;
};

}  // namespace lidar3d
