#include "lidar3d/geometry.hpp"

#include <cmath>

namespace lidar3d {

Mat4 make_T(const Mat3& R, const Vec3& t) {
  Mat4 T = Mat4::Identity();
  T.topLeftCorner<3, 3>() = R;
  T.topRightCorner<3, 1>() = t;
  return T;
}

Mat4 inv_T(const Mat4& T) {
  const Mat3 Rt = T.topLeftCorner<3, 3>().transpose();
  Mat4 Ti = Mat4::Identity();
  Ti.topLeftCorner<3, 3>() = Rt;
  Ti.topRightCorner<3, 1>() = -(Rt * T.topRightCorner<3, 1>());
  return Ti;
}

Mat3 rot_z(double yaw) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  Mat3 R;
  R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return R;
}

Mat3 rot_zyx(double yaw, double pitch, double roll) {
  Mat3 Ry, Rx;
  const double cp = std::cos(pitch), sp = std::sin(pitch), cr = std::cos(roll), sr = std::sin(roll);
  Ry << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;
  Rx << 1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr;
  return rot_z(yaw) * Ry * Rx;
}

double yaw_of(const Mat3& R) { return std::atan2(R(1, 0), R(0, 0)); }

double py_mod(double a, double b) {
  double mod = std::fmod(a, b);
  if (mod != 0.0) {
    if ((b < 0) != (mod < 0)) mod += b;
  } else {
    mod = std::copysign(0.0, b);
  }
  return mod;
}

double wrap_angle(double a) { return py_mod(a + kPi, 2 * kPi) - kPi; }

namespace {

Eigen::Vector4d normalized(Eigen::Vector4d q) { return q / q.norm(); }

// scipy's canonical quaternion: w >= 0, and with w == 0 the first non-zero of x, y, z positive
Eigen::Vector4d canonical(Eigen::Vector4d q) {
  bool flip = q[3] < 0;
  if (q[3] == 0) {
    if (q[0] != 0) flip = q[0] < 0;
    else if (q[1] != 0) flip = q[1] < 0;
    else flip = q[2] < 0;
  }
  return flip ? Eigen::Vector4d(-q) : q;
}

}  // namespace

Mat3 quat_to_R(double x, double y, double z, double w) {
  Eigen::Vector4d q = normalized(Eigen::Vector4d(x, y, z, w));
  x = q[0], y = q[1], z = q[2], w = q[3];
  const double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
  const double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
  Mat3 R;
  R << x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
       2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw),
       2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2;
  return R;
}

Eigen::Vector4d R_to_quat(const Mat3& m) {
  // the matrices here are products of rotations, orthogonal to far better than scipy's 1e-12 check,
  // so its SVD re-orthogonalisation never runs
  const double trace = m(0, 0) + m(1, 1) + m(2, 2);
  const double decision[4] = {m(0, 0), m(1, 1), m(2, 2), trace};
  int choice = 0;
  for (int i = 1; i < 4; ++i)
    if (decision[i] > decision[choice]) choice = i;   // argmax: first maximum
  Eigen::Vector4d q;
  if (choice == 0) {
    q << 1 - trace + 2 * m(0, 0), m(1, 0) + m(0, 1), m(2, 0) + m(0, 2), m(2, 1) - m(1, 2);
  } else if (choice == 1) {
    q << m(1, 0) + m(0, 1), 1 - trace + 2 * m(1, 1), m(2, 1) + m(1, 2), m(0, 2) - m(2, 0);
  } else if (choice == 2) {
    q << m(2, 0) + m(0, 2), m(2, 1) + m(1, 2), 1 - trace + 2 * m(2, 2), m(1, 0) - m(0, 1);
  } else {
    q << m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1), 1 + trace;
  }
  return normalized(q);
}

Vec3 so3_log(const Mat3& R) {
  const Eigen::Vector4d q = canonical(R_to_quat(R));
  const double ax = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
  const double angle = 2 * std::atan2(ax, q[3]);
  double scale;
  if (angle <= 1e-3) {
    const double a2 = angle * angle;
    scale = 2 + a2 / 12 + 7 * (a2 * a2) / 2880;
  } else {
    scale = angle / std::sin(angle / 2.0);
  }
  return Vec3(scale * q[0], scale * q[1], scale * q[2]);
}

Mat3 so3_exp(const Vec3& rv) {
  const double theta = rv.norm();
  Mat3 K;
  if (theta < 1e-9) {   // first order: I + [rv]x
    K << 0.0, -rv[2], rv[1], rv[2], 0.0, -rv[0], -rv[1], rv[0], 0.0;
    return Mat3::Identity() + K;
  }
  const Vec3 k = rv / theta;
  K << 0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0;
  return Mat3::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * (K * K);
}

}  // namespace lidar3d
