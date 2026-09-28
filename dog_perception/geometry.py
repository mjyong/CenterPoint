"""Small SE(3)/SO(3) helpers shared by every stage.

Conventions
-----------
* Rotations are 3x3 matrices, poses are 4x4 homogeneous matrices ``T_A_B``
  mapping points expressed in frame B into frame A (``x_A = T_A_B @ x_B``).
* Quaternions are ``[x, y, z, w]`` (ROS order).
* Yaw is counter-clockwise around +z, measured from +x.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T):
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def transform_points(T, xyz):
    """Apply a 4x4 transform to (N, 3) points."""
    return xyz @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_zyx(yaw, pitch, roll):
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def yaw_of(R):
    """Heading of the body x-axis projected onto the horizontal plane.

    Unlike reading the ZYX Euler yaw this stays well defined under the large
    pitch/roll excursions of a legged robot.
    """
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def quat_to_R(q_xyzw):
    return Rotation.from_quat(q_xyzw).as_matrix()


def R_to_quat(R):
    return Rotation.from_matrix(R).as_quat()


def so3_log(R):
    """(..., 3, 3) rotation matrices -> (..., 3) rotation vectors."""
    R = np.asarray(R)
    flat = R.reshape(-1, 3, 3)
    rv = Rotation.from_matrix(flat).as_rotvec()
    return rv.reshape(R.shape[:-2] + (3,))


def so3_exp(rotvec):
    """(..., 3) rotation vectors -> (..., 3, 3) matrices (vectorised Rodrigues)."""
    rv = np.asarray(rotvec, dtype=np.float64)
    shape = rv.shape[:-1]
    rv = rv.reshape(-1, 3)
    theta = np.linalg.norm(rv, axis=1)
    small = theta < 1e-9
    safe = np.where(small, 1.0, theta)
    k = rv / safe[:, None]
    K = np.zeros((rv.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(theta)[:, None, None]
    c = (1 - np.cos(theta))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    if np.any(small):
        # first-order expansion for tiny angles: I + [rv]x
        Ks = np.zeros((int(small.sum()), 3, 3))
        v = rv[small]
        Ks[:, 0, 1], Ks[:, 0, 2] = -v[:, 2], v[:, 1]
        Ks[:, 1, 0], Ks[:, 1, 2] = v[:, 2], -v[:, 0]
        Ks[:, 2, 0], Ks[:, 2, 1] = -v[:, 1], v[:, 0]
        R[small] = np.eye(3)[None] + Ks
    return R.reshape(shape + (3, 3))


def yaw_to_rot2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])
