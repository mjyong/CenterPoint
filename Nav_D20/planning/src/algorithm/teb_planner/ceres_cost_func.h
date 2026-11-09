#pragma once
#include "src/common/util/config_helper.h"
#include <algorithm>
#include <array>
#include <ceres/ceres.h>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace teb_planner {

template <typename T> T normalize_angle(T angle) {
  T two_pi = T(2.0 * M_PI);
  return angle - two_pi * ceres::floor((angle + T(M_PI)) / two_pi);
}

#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)
class AngleLocalParameterization : public ceres::Manifold {
public:
  virtual ~AngleLocalParameterization() {}

  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    x_plus_delta[0] = normalize_angle(x[0] + delta[0]);
    return true;
  }

  virtual bool PlusJacobian(const double *x, double *jacobian) const override {
    jacobian[0] = 1.0;
    return true;
  }

  virtual bool Minus(const double *y, const double *x,
                     double *y_minus_x) const override {
    y_minus_x[0] = normalize_angle(y[0] - x[0]);
    return true;
  }

  virtual bool MinusJacobian(const double *x, double *jacobian) const override {
    jacobian[0] = 1.0;
    return true;
  }

  virtual int AmbientSize() const override { return 1; }
  virtual int TangentSize() const override { return 1; }
};
#else
class AngleLocalParameterization : public ceres::LocalParameterization {
public:
  virtual ~AngleLocalParameterization() {}

  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const {
    x_plus_delta[0] = normalize_angle(x[0] + delta[0]);
    return true;
  }

  virtual bool ComputeJacobian(const double *x, double *jacobian) const {
    jacobian[0] = 1.0;
    return true;
  }

  virtual int GlobalSize() const { return 1; }
  virtual int LocalSize() const { return 1; }
};
#endif
template <typename T>
void computeRobotVelocity(const T *x1, const T *y1, const T *theta1,
                          const T *x2, const T *y2, const T *theta2,
                          const T *dt, T &vx_robot, T &vy_robot, T &omega) {
  T dx = x2[0] - x1[0];
  T dy = y2[0] - y1[0];
  T dtheta = normalize_angle(theta2[0] - theta1[0]);

  T safe_dt = ceres::fmax(dt[0], T(1e-3));
  T vx_global = dx / safe_dt;
  T vy_global = dy / safe_dt;
  omega = dtheta / safe_dt;

  T cos_theta = ceres::cos(theta1[0]);
  T sin_theta = ceres::sin(theta1[0]);
  vx_robot = vx_global * cos_theta + vy_global * sin_theta;
  vy_robot = -vx_global * sin_theta + vy_global * cos_theta;
}

struct PathFollowingCost {
  PathFollowingCost(const double &reference_x, const double &reference_y,
                    const double &reference_theta, double weight)
      : ref_x_(reference_x), ref_y_(reference_y), ref_theta_(reference_theta),
        weight_(weight) {}

  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const theta,
                  T *residual) const {
    residual[0] = T(weight_) * ceres::abs(x[0] - T(ref_x_));
    residual[1] = T(weight_) * ceres::abs(y[0] - T(ref_y_));

    T angle_diff = normalize_angle(theta[0] - T(ref_theta_));
    residual[2] = T(weight_) * ceres::abs(angle_diff);

    return true;
  }

private:
  double ref_x_, ref_y_, ref_theta_;
  double weight_;
};

struct InitialVelocityCost {
  InitialVelocityCost(const double &initial_vx, const double &initial_vy,
                      const double &initial_omega, double weight)
      : initial_vx_(initial_vx), initial_vy_(initial_vy),
        initial_omega_(initial_omega), weight_(weight) {}

  template <typename T>
  bool operator()(const T *const x1, const T *const y1, const T *const theta1,
                  const T *const x2, const T *const y2, const T *const theta2,
                  const T *const dt, T *residual) const {
    T vx_robot, vy_robot, omega;
    computeRobotVelocity(x1, y1, theta1, x2, y2, theta2, dt, vx_robot, vy_robot,
                         omega);

    residual[0] = T(weight_) * ceres::abs(vx_robot - T(initial_vx_));
    residual[1] = T(weight_) * ceres::abs(vy_robot - T(initial_vy_));
    residual[2] = T(weight_) * ceres::abs(omega - T(initial_omega_));

    return true;
  }

private:
  double initial_vx_, initial_vy_, initial_omega_;
  double weight_;
};

struct FinalVelocityCost {
  FinalVelocityCost(double weight) : weight_(weight) {}

  template <typename T>
  bool operator()(const T *const x1, const T *const y1, const T *const theta1,
                  const T *const x2, const T *const y2, const T *const theta2,
                  const T *const dt, T *residual) const {
    T vx_robot, vy_robot, omega;
    computeRobotVelocity(x1, y1, theta1, x2, y2, theta2, dt, vx_robot, vy_robot,
                         omega);

    residual[0] = T(weight_) * ceres::abs(vx_robot);
    residual[1] = T(weight_) * ceres::abs(vy_robot);
    residual[2] = T(weight_) * ceres::abs(omega);

    return true;
  }

private:
  double weight_;
};

inline double Sign(double value) {
  if (value > 0.0) {
    return 1.0;
  }
  if (value < 0.0) {
    return -1.0;
  }
  return 0.0;
}

inline void ComputeRobotVelocity7(double const *const *parameters,
                                  double *vx_robot, double *vy_robot,
                                  double *omega,
                                  std::array<double, 7> *d_vx_robot,
                                  std::array<double, 7> *d_vy_robot,
                                  std::array<double, 7> *d_omega) {
  const double x1 = parameters[0][0];
  const double y1 = parameters[1][0];
  const double theta1 = parameters[2][0];
  const double x2 = parameters[3][0];
  const double y2 = parameters[4][0];
  const double theta2 = parameters[5][0];
  const double dt = parameters[6][0];

  const double dx = x2 - x1;
  const double dy = y2 - y1;
  const double dtheta = normalize_angle(theta2 - theta1);
  const bool dt_active = dt > 1e-3;
  const double safe_dt = std::max(dt, 1e-3);
  const double inv_dt = 1.0 / safe_dt;
  const double inv_dt_sq = inv_dt * inv_dt;

  const double vx_global = dx * inv_dt;
  const double vy_global = dy * inv_dt;
  *omega = dtheta * inv_dt;

  const double cos_theta = std::cos(theta1);
  const double sin_theta = std::sin(theta1);
  *vx_robot = vx_global * cos_theta + vy_global * sin_theta;
  *vy_robot = -vx_global * sin_theta + vy_global * cos_theta;

  std::array<double, 7> d_vx_global{};
  std::array<double, 7> d_vy_global{};
  d_vx_global[0] = -inv_dt;
  d_vx_global[3] = inv_dt;
  d_vy_global[1] = -inv_dt;
  d_vy_global[4] = inv_dt;
  (*d_omega)[2] = -inv_dt;
  (*d_omega)[5] = inv_dt;
  if (dt_active) {
    d_vx_global[6] = -dx * inv_dt_sq;
    d_vy_global[6] = -dy * inv_dt_sq;
    (*d_omega)[6] = -dtheta * inv_dt_sq;
  }

  for (int i = 0; i < 7; ++i) {
    (*d_vx_robot)[i] = cos_theta * d_vx_global[i] + sin_theta * d_vy_global[i];
    (*d_vy_robot)[i] = -sin_theta * d_vx_global[i] + cos_theta * d_vy_global[i];
  }

  (*d_vx_robot)[2] += -vx_global * sin_theta + vy_global * cos_theta;
  (*d_vy_robot)[2] += -vx_global * cos_theta - vy_global * sin_theta;
}

inline void FillJacobianRow(double **jacobians, int residual_idx,
                            double residual_scale,
                            const std::array<double, 7> &derivatives) {
  for (int block_idx = 0; block_idx < 7; ++block_idx) {
    if (jacobians[block_idx] != nullptr) {
      jacobians[block_idx][residual_idx] =
          residual_scale * derivatives[block_idx];
    }
  }
}

class AccelerationLimitAnalyticCost
    : public ceres::SizedCostFunction<3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1> {
public:
  AccelerationLimitAnalyticCost(double max_accel_vx, double max_accel_vy,
                                double max_accel_omega, double weight_smooth,
                                double weight_limit)
      : max_accel_vx_(max_accel_vx), max_accel_vy_(max_accel_vy),
        max_accel_omega_(max_accel_omega), weight_smooth_(weight_smooth),
        weight_limit_(weight_limit) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    double vx1 = 0.0, vy1 = 0.0, omega1 = 0.0;
    double vx2 = 0.0, vy2 = 0.0, omega2 = 0.0;
    std::array<double, 7> d_vx1{}, d_vy1{}, d_omega1{};
    std::array<double, 7> d_vx2{}, d_vy2{}, d_omega2{};

    const double *params1[] = {parameters[0], parameters[1], parameters[2],
                               parameters[3], parameters[4], parameters[5],
                               parameters[6]};
    const double *params2[] = {parameters[3], parameters[4], parameters[5],
                               parameters[7], parameters[8], parameters[9],
                               parameters[10]};
    ComputeRobotVelocity7(params1, &vx1, &vy1, &omega1, &d_vx1, &d_vy1,
                          &d_omega1);
    ComputeRobotVelocity7(params2, &vx2, &vy2, &omega2, &d_vx2, &d_vy2,
                          &d_omega2);

    const double dt1 = parameters[6][0];
    const double dt2 = parameters[10][0];
    const double safe_dt = std::max((dt1 + dt2) / 2.0, 1e-3);
    const double inv_dt = 1.0 / safe_dt;

    const double accel_vx = (vx2 - vx1) * inv_dt;
    const double accel_vy = (vy2 - vy1) * inv_dt;
    const double accel_omega = (omega2 - omega1) * inv_dt;

    residuals[0] =
        weight_smooth_ * std::abs(accel_vx) +
        weight_limit_ * std::max(0.0, std::abs(accel_vx) - max_accel_vx_);
    residuals[1] =
        weight_smooth_ * std::abs(accel_vy) +
        weight_limit_ * std::max(0.0, std::abs(accel_vy) - max_accel_vy_);
    residuals[2] =
        weight_smooth_ * std::abs(accel_omega) +
        weight_limit_ * std::max(0.0, std::abs(accel_omega) - max_accel_omega_);

    if (jacobians == nullptr) {
      return true;
    }

    double s_accel_vx = weight_smooth_ * Sign(accel_vx);
    double s_accel_vy = weight_smooth_ * Sign(accel_vy);
    double s_accel_omega = weight_smooth_ * Sign(accel_omega);
    if (std::abs(accel_vx) > max_accel_vx_) {
      s_accel_vx += weight_limit_ * Sign(accel_vx);
    }
    if (std::abs(accel_vy) > max_accel_vy_) {
      s_accel_vy += weight_limit_ * Sign(accel_vy);
    }
    if (std::abs(accel_omega) > max_accel_omega_) {
      s_accel_omega += weight_limit_ * Sign(accel_omega);
    }

    for (int i = 0; i < 11; ++i) {
      if (jacobians[i] == nullptr) {
        continue;
      }
      for (int r = 0; r < 3; ++r) {
        jacobians[i][r] = 0.0;
      }
    }

    const int params1_block_map[] = {0, 1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 7; ++i) {
      int block1 = params1_block_map[i];
      if (jacobians[block1] != nullptr) {
        jacobians[block1][0] = -s_accel_vx * inv_dt * d_vx1[i];
        jacobians[block1][1] = -s_accel_vy * inv_dt * d_vy1[i];
        jacobians[block1][2] = -s_accel_omega * inv_dt * d_omega1[i];
      }
    }

    const int params2_block_map[] = {3, 4, 5, 7, 8, 9, 10};
    for (int i = 0; i < 7; ++i) {
      int block2 = params2_block_map[i];
      if (jacobians[block2] != nullptr) {
        jacobians[block2][0] += s_accel_vx * inv_dt * d_vx2[i];
        jacobians[block2][1] += s_accel_vy * inv_dt * d_vy2[i];
        jacobians[block2][2] += s_accel_omega * inv_dt * d_omega2[i];
      }
    }

    return true;
  }

private:
  double max_accel_vx_, max_accel_vy_, max_accel_omega_;
  double weight_smooth_, weight_limit_;
};

class VelocityLimitAnalyticCost
    : public ceres::SizedCostFunction<3, 1, 1, 1, 1, 1, 1, 1> {
public:
  VelocityLimitAnalyticCost(double max_vx, double max_vy, double max_omega,
                            double weight)
      : max_vx_(max_vx), max_vy_(max_vy), max_omega_(max_omega),
        weight_(weight) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    double vx_robot = 0.0;
    double vy_robot = 0.0;
    double omega = 0.0;
    std::array<double, 7> d_vx_robot{};
    std::array<double, 7> d_vy_robot{};
    std::array<double, 7> d_omega{};
    ComputeRobotVelocity7(parameters, &vx_robot, &vy_robot, &omega, &d_vx_robot,
                          &d_vy_robot, &d_omega);

    residuals[0] = weight_ * std::max(0.0, std::abs(vx_robot) - max_vx_);
    residuals[1] = weight_ * std::max(0.0, std::abs(vy_robot) - max_vy_);
    residuals[2] = weight_ * std::max(0.0, std::abs(omega) - max_omega_);

    if (jacobians == nullptr) {
      return true;
    }

    FillJacobianRow(jacobians, 0, LimitDerivativeScale(vx_robot, max_vx_),
                    d_vx_robot);
    FillJacobianRow(jacobians, 1, LimitDerivativeScale(vy_robot, max_vy_),
                    d_vy_robot);
    FillJacobianRow(jacobians, 2, LimitDerivativeScale(omega, max_omega_),
                    d_omega);
    return true;
  }

  double LimitDerivativeScale(double value, double limit) const {
    return std::abs(value) > limit ? weight_ * Sign(value) : 0.0;
  }

private:
  double max_vx_, max_vy_, max_omega_;
  double weight_;
};

struct ReferenceVelocityCost {
  ReferenceVelocityCost(double reference_velocity, double weight)
      : reference_velocity_(reference_velocity), weight_(weight) {}

  template <typename T>
  bool operator()(const T *const x1, const T *const y1, const T *const theta1,
                  const T *const x2, const T *const y2, const T *const theta2,
                  const T *const dt, T *residual) const {
    T dx = x2[0] - x1[0];
    T dy = y2[0] - y1[0];
    T distance_sq = dx * dx + dy * dy;

    T safe_dt = ceres::fmax(dt[0], T(1e-3));
    T velocity = ceres::sqrt(distance_sq + T(1e-6)) / safe_dt;

    residual[0] = T(weight_) * (velocity - T(reference_velocity_));
    return true;
  }

private:
  double reference_velocity_;
  double weight_;
};

class LateralVelocityAnalyticCost
    : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1, 1, 1> {
public:
  LateralVelocityAnalyticCost(double weight, double ref_vx)
      : weight_(weight), ref_vx_(ref_vx) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    double vx_robot = 0.0;
    double vy_robot = 0.0;
    double omega = 0.0;
    std::array<double, 7> d_vx_robot{};
    std::array<double, 7> d_vy_robot{};
    std::array<double, 7> d_omega{};
    ComputeRobotVelocity7(parameters, &vx_robot, &vy_robot, &omega, &d_vx_robot,
                          &d_vy_robot, &d_omega);

    double abs_vx = std::abs(vx_robot);
    double sigmoid = 1.0 / (1.0 + std::exp(-20.0 * (abs_vx - 0.3)));
    double scale = 0.5 + 1.5 * sigmoid;
    double abs_vy = std::abs(vy_robot);
    double sign_vy = Sign(vy_robot);

    residuals[0] = weight_ * scale * abs_vy;

    if (jacobians == nullptr) {
      return true;
    }

    double d_scale_d_abs_vx = 1.5 * 20.0 * sigmoid * (1.0 - sigmoid);
    double d_abs_vx_d_vx = Sign(vx_robot);
    double d_scale_d_vx = d_scale_d_abs_vx * d_abs_vx_d_vx;

    for (int i = 0; i < 7; ++i) {
      if (jacobians[i] != nullptr) {
        double d_res_d_param =
            weight_ * (d_scale_d_vx * d_vx_robot[i] * abs_vy +
                       scale * sign_vy * d_vy_robot[i]);
        jacobians[i][0] = d_res_d_param;
      }
    }
    return true;
  }

private:
  double weight_;
  double ref_vx_;
};

struct NonHolonomicCost : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1> {
  explicit NonHolonomicCost(double weight) : weight_(weight) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double x1 = parameters[0][0];
    const double y1 = parameters[1][0];
    const double theta1 = parameters[2][0];
    const double x2 = parameters[3][0];
    const double y2 = parameters[4][0];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double sin_theta = std::sin(theta1);
    const double cos_theta = std::cos(theta1);

    residuals[0] = weight_ * (sin_theta * dx - cos_theta * dy);

    if (jacobians == nullptr) {
      return true;
    }
    if (jacobians[0] != nullptr) {
      jacobians[0][0] = -weight_ * sin_theta;
    }
    if (jacobians[1] != nullptr) {
      jacobians[1][0] = weight_ * cos_theta;
    }
    if (jacobians[2] != nullptr) {
      jacobians[2][0] = weight_ * (cos_theta * dx + sin_theta * dy);
    }
    if (jacobians[3] != nullptr) {
      jacobians[3][0] = weight_ * sin_theta;
    }
    if (jacobians[4] != nullptr) {
      jacobians[4][0] = -weight_ * cos_theta;
    }
    return true;
  }

private:
  double weight_;
};

struct PreferForwardCost : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1> {
  explicit PreferForwardCost(double weight) : weight_(weight) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double x1 = parameters[0][0];
    const double y1 = parameters[1][0];
    const double theta1 = parameters[2][0];
    const double x2 = parameters[3][0];
    const double y2 = parameters[4][0];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double sin_theta = std::sin(theta1);
    const double cos_theta = std::cos(theta1);
    const double forward = dx * cos_theta + dy * sin_theta;

    residuals[0] = weight_ * std::max(0.0, -forward);

    if (jacobians == nullptr) {
      return true;
    }

    const double residual_scale = forward < 0.0 ? -weight_ : 0.0;
    if (jacobians[0] != nullptr) {
      jacobians[0][0] = residual_scale * -cos_theta;
    }
    if (jacobians[1] != nullptr) {
      jacobians[1][0] = residual_scale * -sin_theta;
    }
    if (jacobians[2] != nullptr) {
      jacobians[2][0] = residual_scale * (-dx * sin_theta + dy * cos_theta);
    }
    if (jacobians[3] != nullptr) {
      jacobians[3][0] = residual_scale * cos_theta;
    }
    if (jacobians[4] != nullptr) {
      jacobians[4][0] = residual_scale * sin_theta;
    }
    return true;
  }

private:
  double weight_;
};

class ObstacleCost : public ceres::SizedCostFunction<2, 1, 1, 1> {
public:
  ObstacleCost(
      std::function<double(const double, const double)> obstacle_distance_func,
      std::function<bool(const double, const double, const double)>
          is_collision_func,
      double weight_warning, double weight_danger, double d1, double d2,
      double steepness = 20.0)
      : obstacle_distance_func_(obstacle_distance_func),
        is_collision_func_(is_collision_func), weight_warning_(weight_warning),
        weight_danger_(weight_danger), d1_(d1), d2_(d2), steepness_(steepness) {
    auto circles =
        planning::common::util::ConfigHelper::Instance()->GetBoundingCircles(
            0, 0, 0);
    for (const auto &c : circles) {
      circle_offsets_.emplace_back(c.center_x, c.center_y);
      circle_radii_.push_back(c.radius);
    }
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double x = parameters[0][0];
    const double y = parameters[1][0];
    const double theta = parameters[2][0];
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);

    double min_clearance = std::numeric_limits<double>::max();
    size_t j_star = 0;
    for (size_t i = 0; i < circle_offsets_.size(); ++i) {
      const double ox = circle_offsets_[i].x();
      const double oy = circle_offsets_[i].y();
      const double cx = x + ox * cos_t - oy * sin_t;
      const double cy = y + ox * sin_t + oy * cos_t;
      const double dist = obstacle_distance_func_(cx, cy);
      const double clearance = dist - circle_radii_[i];
      if (clearance < min_clearance) {
        min_clearance = clearance;
        j_star = i;
      }
    }

    const double k = steepness_;
    const double arg1 = k * (d1_ - min_clearance);
    const double arg2 = k * (d2_ - min_clearance);
    auto softplus = [](double v) {
      return std::max(v, 0.0) + std::log(1.0 + std::exp(-std::abs(v)));
    };
    residuals[0] =
        (weight_warning_ * softplus(arg1) + weight_danger_ * softplus(arg2)) /
        k;
    residuals[1] = is_collision_func_(x, y, theta) ? 1e3 : 0.0;

    if (jacobians == nullptr) {
      return true;
    }

    for (int b = 0; b < 3; ++b) {
      if (jacobians[b] != nullptr) {
        jacobians[b][0] = 0.0;
        jacobians[b][1] = 0.0;
      }
    }

    const double sig1 = 1.0 / (1.0 + std::exp(-arg1));
    const double sig2 = 1.0 / (1.0 + std::exp(-arg2));
    const double dres0_dd = -weight_warning_ * sig1 - weight_danger_ * sig2;

    const double ox = circle_offsets_[j_star].x();
    const double oy = circle_offsets_[j_star].y();
    const double cx_star = x + ox * cos_t - oy * sin_t;
    const double cy_star = y + ox * sin_t + oy * cos_t;
    const double dist0 = obstacle_distance_func_(cx_star, cy_star);
    const double gx =
        (obstacle_distance_func_(cx_star + kGradEps, cy_star) - dist0) /
        kGradEps;
    const double gy =
        (obstacle_distance_func_(cx_star, cy_star + kGradEps) - dist0) /
        kGradEps;

    const double ddist_dx = gx;
    const double ddist_dy = gy;
    const double ddist_dtheta =
        gx * (-ox * sin_t - oy * cos_t) + gy * (ox * cos_t - oy * sin_t);

    if (jacobians[0] != nullptr) {
      jacobians[0][0] = dres0_dd * ddist_dx;
    }
    if (jacobians[1] != nullptr) {
      jacobians[1][0] = dres0_dd * ddist_dy;
    }
    if (jacobians[2] != nullptr) {
      jacobians[2][0] = dres0_dd * ddist_dtheta;
    }
    return true;
  }

private:
  std::function<double(double, double)> obstacle_distance_func_;
  std::function<bool(double, double, double)> is_collision_func_;
  double weight_warning_;
  double weight_danger_;
  double d1_;
  double d2_;
  double steepness_;
  std::vector<Eigen::Vector2d> circle_offsets_;
  std::vector<double> circle_radii_;

  static constexpr double kGradEps = 0.05;
};

struct TimeOptimalCost {
  TimeOptimalCost(double weight) : weight_(weight) {}

  template <typename T> bool operator()(const T *const dt, T *residual) const {
    residual[0] = T(weight_) * ceres::fmax(dt[0], T(1e-3));
    return true;
  }

private:
  double weight_;
};

struct PathSmoothnessCost {
  PathSmoothnessCost(double weight) : weight_(weight) {}

  template <typename T>
  bool operator()(const T *const x1, const T *const y1, const T *const theta1,
                  const T *const x2, const T *const y2, const T *const theta2,
                  T *residual) const {
    residual[0] = T(weight_) * ceres::pow(x2[0] - x1[0], T(2.0));
    residual[1] = T(weight_) * ceres::pow(y2[0] - y1[0], T(2.0));

    T angle_diff = normalize_angle(theta2[0] - theta1[0]);
    residual[2] = T(weight_) * ceres::pow(angle_diff, T(2.0));

    return true;
  }

private:
  double weight_;
};

} // namespace teb_planner