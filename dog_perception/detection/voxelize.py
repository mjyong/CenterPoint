"""Pillar / voxel generation in NumPy, the same voxels as det3d's ``points_to_voxel`` without det3d.

The deployed detectors (``exported.py``: ONNX Runtime or TensorRT engines) run on the Orin without
PyTorch and det3d, so their voxelization cannot use det3d's numba kernel. The result here is
identical to it (``tests/dog/test_detection.py``):

* the cell of a point is ``floor((p - range_min) / voxel_size)`` in float32, the kernel's dtype;
  points outside the grid are dropped;
* voxels are numbered in the order of their first point; once ``max_voxels`` exist, points of new
  cells are dropped;
* a voxel keeps its first ``max_points`` points, in input order;
* coordinates are (z, y, x).
"""
import numpy as np


def voxelize(points, voxel_size, pc_range, max_points, max_voxels):
    """points (N, C) float32. Returns voxels (V, max_points, C) float32, coords (V, 3) int32 (z, y, x),
    num_points (V,) int32."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    vs = np.asarray(voxel_size, dtype=np.float32)
    lo = np.asarray(pc_range, dtype=np.float32)[:3]
    grid = np.round((np.asarray(pc_range, dtype=np.float32)[3:] - lo) / vs).astype(np.int64)
    C = pts.shape[1]
    empty = (np.zeros((0, max_points, C), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int32))
    if len(pts) == 0:
        return empty
    cell = np.floor((pts[:, :3] - lo) / vs)                      # float32, like the numba kernel
    ok = np.all((cell >= 0) & (cell < grid), axis=1)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return empty
    c = cell[idx].astype(np.int64)
    lin = (c[:, 2] * grid[1] + c[:, 1]) * grid[0] + c[:, 0]
    uniq, first, inv = np.unique(lin, return_index=True, return_inverse=True)
    rank = np.empty(len(uniq), np.int64)                        # voxel number = order of the first point
    rank[np.argsort(first, kind="stable")] = np.arange(len(uniq))
    vid = rank[inv.reshape(-1)]
    keep = vid < max_voxels
    idx, vid, c = idx[keep], vid[keep], c[keep]
    n_vox = int(min(len(uniq), max_voxels))
    order = np.argsort(vid, kind="stable")                       # by voxel, then input order
    vs_sorted = vid[order]
    start = np.searchsorted(vs_sorted, np.arange(n_vox))
    slot = np.empty(len(order), np.int64)
    slot[order] = np.arange(len(order)) - start[vs_sorted]
    take = slot < max_points
    voxels = np.zeros((n_vox, max_points, C), np.float32)
    voxels[vid[take], slot[take]] = pts[idx[take]]
    num = np.minimum(np.bincount(vid, minlength=n_vox), max_points).astype(np.int32)
    coords = np.zeros((n_vox, 3), np.int32)
    coords[vid, 0], coords[vid, 1], coords[vid, 2] = c[:, 2], c[:, 1], c[:, 0]
    return voxels, coords, num


def grid_size(voxel_size, pc_range):
    """(nx, ny, nz) of the grid, as det3d's VoxelGenerator computes it."""
    vs = np.asarray(voxel_size, dtype=np.float32)
    r = np.asarray(pc_range, dtype=np.float32)
    return np.round((r[3:] - r[:3]) / vs).astype(np.int64)
