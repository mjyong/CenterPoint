// SE(3) / SO(3) helpers (dog_perception/geometry.py).
//
// Poses are 4x4 T_A_B mapping points of frame B into frame A; quaternions are [x, y, z, w] (ROS order);
// yaw is counter-clockwise about +z from +x. The rotation conversions follow scipy's Rotation (the
// Python reference uses it), so both round the same way.
#pragma once

#include <Eigen/Dense>

namespace dogperc {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;

constexpr double kPi = 3.14159265358979323846;

Mat4 make_T(const Mat3& R, const Vec3& t);
Mat4 inv_T(const Mat4& T);
Mat3 rot_z(double yaw);
Mat3 rot_zyx(double yaw, double pitch, double roll);   // scipy from_euler("ZYX", [yaw, pitch, roll])
double yaw_of(const Mat3& R);                            // heading of the body x axis on the ground plane
// (a + pi) mod 2 pi - pi with NumPy's floor-modulo (result in [-pi, pi))
double wrap_angle(double a);
double py_mod(double a, double b);                       // NumPy np.remainder for floats

Mat3 quat_to_R(double x, double y, double z, double w);  // scipy from_quat(...).as_matrix()
Eigen::Vector4d R_to_quat(const Mat3& R);                // scipy from_matrix(...).as_quat()
Vec3 so3_log(const Mat3& R);                             // scipy from_matrix(R).as_rotvec()
Mat3 so3_exp(const Vec3& rotvec);                        // geometry.so3_exp (Rodrigues)

}  // namespace dogperc
