"""Per-point motion compensation (deskew).

Every point is moved from the body pose at its own firing time to the body
pose at the scan reference time (the scan end, which is also the timestamp
the detections carry):

    x_B(t_ref) = T_WB(t_ref)^-1 * T_WB(t_i) * T_BL * x_L
"""
import numpy as np

from ..geometry import inv_T


def deskew_points(xyz_lidar, point_times, pose_buffer, t_ref, T_body_lidar, time_resolution=1e-4):
    """Motion-compensate one sweep.

    Args:
        xyz_lidar: (N, 3) points in the lidar frame.
        point_times: (N,) absolute firing time of every point [s].
        pose_buffer: ``PoseBuffer`` holding IMU-rate ``T_world_body``.
        t_ref: reference time the output is expressed at.
        T_body_lidar: 4x4 lidar extrinsic.
        time_resolution: points are grouped into bins of this width and share
            one interpolated pose. XT32 fires a 32-channel block every ~50 us,
            so 0.1 ms bins are effectively exact and keep this O(#bins) instead
            of O(#points) in pose interpolation.

    Returns:
        (N, 3) points in the body frame at ``t_ref``.
    """
    xyz_lidar = np.asarray(xyz_lidar, dtype=np.float64)
    point_times = np.asarray(point_times, dtype=np.float64)

    if time_resolution and time_resolution > 0:
        bins = np.round((point_times - t_ref) / time_resolution).astype(np.int64)
        ubins, inv = np.unique(bins, return_inverse=True)
        query_t = t_ref + ubins * time_resolution
    else:
        query_t, inv = point_times, np.arange(len(point_times))

    R_i, p_i = pose_buffer.interpolate(query_t)
    T_ref_inv = inv_T(pose_buffer.pose_at(t_ref))

    # compose T_ref^-1 * T_WB(t_i) * T_BL for every bin
    R_rel = np.einsum("ij,njk->nik", T_ref_inv[:3, :3], R_i)
    t_rel = p_i @ T_ref_inv[:3, :3].T + T_ref_inv[:3, 3]
    R_full = np.einsum("nij,jk->nik", R_rel, T_body_lidar[:3, :3])
    t_full = np.einsum("nij,j->ni", R_rel, T_body_lidar[:3, 3]) + t_rel

    return np.einsum("nij,nj->ni", R_full[inv], xyz_lidar) + t_full[inv]
