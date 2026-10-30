#include "lidar3d/tracking/imm.hpp"

#include <algorithm>
#include <cmath>

namespace lidar3d {

namespace {

StateMat pv_noise(double dt, double q_acc) {
  const double q = std::pow(q_acc, 2.0);
  const double b00 = q * (std::pow(dt, 4.0) / 4), b01 = q * (std::pow(dt, 3.0) / 2), b11 = q * std::pow(dt, 2.0);
  StateMat Q = StateMat::Zero();
  for (int i = 0; i < 2; ++i) {     // x: (0, 2), y: (1, 3)
    Q(i, i) = b00;
    Q(i, i + 2) = b01;
    Q(i + 2, i) = b01;
    Q(i + 2, i + 2) = b11;
  }
  return Q;
}

StateVec ct_f(const StateVec& x, double dt) {
  const double px = x[0], py = x[1], vx = x[2], vy = x[3], w = x[4];
  StateVec o;
  if (std::abs(w) < 1e-4) {
    const double dt2 = std::pow(dt, 2.0);
    o << px + vx * dt - vy * w * dt2 / 2, py + vy * dt + vx * w * dt2 / 2, vx - vy * w * dt, vy + vx * w * dt, w;
    return o;
  }
  const double s = std::sin(w * dt), c = std::cos(w * dt);
  o << px + (s * vx - (1 - c) * vy) / w, py + ((1 - c) * vx + s * vy) / w, c * vx - s * vy, s * vx + c * vy, w;
  return o;
}

StateMat ct_F(const StateVec& x, double dt) {
  const double vx = x[2], vy = x[3], w = x[4];
  StateMat F = StateMat::Identity();
  if (std::abs(w) < 1e-4) {
    const double dt2 = std::pow(dt, 2.0);
    F(0, 2) = F(1, 3) = dt;
    F(0, 4) = -vy * dt2 / 2;
    F(1, 4) = vx * dt2 / 2;
    F(2, 4) = -vy * dt;
    F(3, 4) = vx * dt;
    return F;
  }
  const double s = std::sin(w * dt), c = std::cos(w * dt), w2 = std::pow(w, 2.0);
  F(0, 2) = s / w;
  F(0, 3) = -(1 - c) / w;
  F(1, 2) = (1 - c) / w;
  F(1, 3) = s / w;
  F(2, 2) = c;
  F(2, 3) = -s;
  F(3, 2) = s;
  F(3, 3) = c;
  const double a = (dt * c * w - s) / w2;          // d(s/w)/dw
  const double b = (dt * s * w - (1 - c)) / w2;    // d((1-c)/w)/dw
  F(0, 4) = a * vx - b * vy;
  F(1, 4) = b * vx + a * vy;
  F(2, 4) = -dt * s * vx - dt * c * vy;
  F(3, 4) = dt * c * vx - dt * s * vy;
  return F;
}

}  // namespace

MotionModel MotionModel::from_spec(const MotionModelSpec& s) {
  MotionModel m;
  m.kind = s.kind == "ct" ? CT : CV;
  m.q_acc = s.q_acc;
  m.q_omega = s.q_omega;
  return m;
}

void MotionModel::predict(const StateVec& x, const StateMat& P, double dt, StateVec& xo, StateMat& Po) const {
  StateMat Q = pv_noise(dt, q_acc);
  StateMat F;
  if (kind == CV) {
    F = StateMat::Identity();
    F(0, 2) = F(1, 3) = dt;
    F(4, 4) = 0.0;
    Q(4, 4) = std::pow(q_omega, 2.0);
    xo = F * x;
  } else {
    F = ct_F(x, dt);
    Q(4, 4) = std::pow(q_omega, 2.0) * dt;
    xo = ct_f(x, dt);
  }
  Po = (F * P) * F.transpose() + Q;
}

IMM::IMM(std::vector<MotionModel> models, double stay_prob, double dt_ref)
    : models_(std::move(models)), dt_ref_(dt_ref) {
  const int M = num_models();
  if (M == 1) {
    Pi_ref_ = Eigen::MatrixXd::Ones(1, 1);
  } else {
    Pi_ref_ = Eigen::MatrixXd::Constant(M, M, (1 - stay_prob) / (M - 1));
    Pi_ref_.diagonal().setConstant(stay_prob);
  }
  mu_.assign(M, 1.0 / M);
  x_.assign(M, StateVec::Zero());
  P_.assign(M, StateMat::Identity());
}

void IMM::initialize(const StateVec& x0, const StateMat& P0) {
  const int M = num_models();
  x_.assign(M, x0);
  P_.assign(M, P0);
  mu_.assign(M, 1.0 / M);
}

Eigen::MatrixXd IMM::transition(double dt) const {
  const int M = num_models();
  if (M == 1) return Pi_ref_;
  Eigen::MatrixXd Pi = Pi_ref_;
  for (int i = 0; i < M; ++i) {
    const double d = std::pow(Pi_ref_(i, i), std::max(dt, 1e-3) / dt_ref_);
    double off_sum = 0.0;
    for (int j = 0; j < M; ++j)
      if (j != i) off_sum += Pi_ref_(i, j);
    for (int j = 0; j < M; ++j) Pi(i, j) = j == i ? 0.0 : Pi_ref_(i, j) / std::max(off_sum, 1e-12) * (1 - d);
    Pi(i, i) = d;
  }
  return Pi;
}

void IMM::predict(double dt) {
  if (dt <= 0) return;
  const int M = num_models();
  const Eigen::MatrixXd Pi = transition(dt);
  std::vector<double> c(M, 0.0);
  for (int j = 0; j < M; ++j)
    for (int i = 0; i < M; ++i) c[j] += Pi(i, j) * mu_[i];
  Eigen::MatrixXd W(M, M);   // W(i, j) = P(i | j)
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j) W(i, j) = (Pi(i, j) * mu_[i]) / std::max(c[j], 1e-12);
  std::vector<StateVec> x_mix(M, StateVec::Zero());
  for (int j = 0; j < M; ++j)
    for (int i = 0; i < M; ++i) x_mix[j] += W(i, j) * x_[i];
  std::vector<StateMat> P_mix(M, StateMat::Zero());
  for (int j = 0; j < M; ++j)
    for (int i = 0; i < M; ++i) {
      const StateVec d = x_[i] - x_mix[j];
      P_mix[j] += W(i, j) * (P_[i] + d * d.transpose());
    }
  for (int j = 0; j < M; ++j) models_[j].predict(x_mix[j], P_mix[j], dt, x_[j], P_[j]);
  double cs = 0.0;
  for (double v : c) cs += v;
  for (int j = 0; j < M; ++j) mu_[j] = c[j] / cs;
}

double IMM::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  const int M = num_models();
  std::vector<double> lik(M);
  const StateMat I = StateMat::Identity();
  for (int j = 0; j < M; ++j) {
    const StateVec x = x_[j];
    const StateMat P = P_[j];
    const Eigen::VectorXd y = z - H * x;
    const Eigen::MatrixXd S = (H * P) * H.transpose() + R;
    const Eigen::MatrixXd S_inv = S.inverse();
    const Eigen::MatrixXd K = (P * H.transpose()) * S_inv;
    x_[j] = x + K * y;
    const StateMat A = I - K * H;
    P_[j] = (A * P) * A.transpose() + (K * R) * K.transpose();
    const double logdet = std::log((2 * kPi * S).determinant());
    lik[j] = -0.5 * (y.dot(S_inv * y) + logdet);
  }
  const double m = *std::max_element(lik.begin(), lik.end());
  std::vector<double> w(M);
  double ws = 0.0;
  for (int j = 0; j < M; ++j) {
    w[j] = mu_[j] * std::exp(lik[j] - m);
    ws += w[j];
  }
  double s = 0.0;
  for (int j = 0; j < M; ++j) {
    mu_[j] = std::max(w[j] / ws, 1e-4);
    s += mu_[j];
  }
  for (int j = 0; j < M; ++j) mu_[j] /= s;
  return m + std::log(ws);
}

void IMM::state(StateVec& x, StateMat& P) const {
  x = StateVec::Zero();
  for (int j = 0; j < num_models(); ++j) x += mu_[j] * x_[j];
  P = StateMat::Zero();
  for (int j = 0; j < num_models(); ++j) {
    const StateVec d = x_[j] - x;
    P += mu_[j] * (P_[j] + d * d.transpose());
  }
}

StateVec IMM::mean() const {
  StateVec x = StateVec::Zero();
  for (int j = 0; j < num_models(); ++j) x += mu_[j] * x_[j];
  return x;
}

Rollout IMM::rollout(double horizon, double step) const {
  const int n = static_cast<int>(std::nearbyint(horizon / step));
  Rollout r;
  r.mode_probs = mu_;
  for (int k = 1; k <= n; ++k) r.times.push_back(step * k);
  for (int j = 0; j < num_models(); ++j) {
    StateVec x = x_[j];
    StateMat P = P_[j];
    std::vector<StateVec> ms;
    std::vector<StateMat> cs;
    for (int k = 0; k < n; ++k) {
      StateVec xn;
      StateMat Pn;
      models_[j].predict(x, P, step, xn, Pn);
      x = xn;
      P = Pn;
      ms.push_back(x);
      cs.push_back(P);
    }
    r.means.push_back(ms);
    r.covs.push_back(cs);
  }
  return r;
}

}  // namespace lidar3d
