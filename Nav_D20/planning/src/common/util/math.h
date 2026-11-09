#pragma once
#include <cmath>
namespace planning {
namespace common {
namespace util {

static double deg2rad(double deg) { return deg * M_PI / 180.0f; }
static double rad2deg(double rad) { return rad * 180.0f / M_PI; }
static double normalize_angle(double angle) {
  double a = std::fmod(angle + M_PI, 2.0 * M_PI);
  if (a < 0.0) {
    a += (2.0 * M_PI);
  }
  return a - M_PI;
}

static double wrap_angle(const double angle) {
  const double new_angle = std::fmod(angle, M_PI * 2.0);
  return new_angle < 0 ? new_angle + M_PI * 2.0 : new_angle;
}

static double lerp(const double &x0, const double t0, const double &x1,
                   const double t1, const double t) {
  if (std::abs(t1 - t0) <= 1.0e-6) {
    return x0;
  }
  const double r = (t - t0) / (t1 - t0);
  const double x = x0 + r * (x1 - x0);
  return x;
}

static double slerp(const double a0, const double t0, const double a1,
                    const double t1, const double t) {
  if (std::abs(t1 - t0) <= 1e-10) {
    return normalize_angle(a0);
  }
  const double a0_n = normalize_angle(a0);
  const double a1_n = normalize_angle(a1);
  double d = a1_n - a0_n;
  if (d > M_PI) {
    d = d - 2 * M_PI;
  } else if (d < -M_PI) {
    d = d + 2 * M_PI;
  }

  const double r = (t - t0) / (t1 - t0);
  const double a = a0_n + d * r;
  return normalize_angle(a);
}

} // namespace util
} // namespace common
} // namespace planning