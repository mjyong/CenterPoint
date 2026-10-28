"""Host side of the deployed CenterPoint-Pillar, in NumPy and without PyTorch (the Orin has none).

Pillar decoration, padding to the PFN engine's static size, scatter into the BEV canvas and the
split of the RPN engine's outputs per task: the reference for the CUDA kernels of deploy/cpp.
``pillar_export.py`` holds the PyTorch side (the graphs that are exported).
"""
import numpy as np


def decorate_pillars(voxels, num_points, coords, voxel_size, pc_range):
    """NumPy port of ``PillarFeatureNet`` feature decoration.

    voxels (P, N, C), num_points (P,), coords (P, 3) as (z, y, x).
    Returns (P, N, C + 5): [raw, xyz - pillar mean, xy - pillar center].
    """
    voxels = voxels.astype(np.float32)
    P, N, _ = voxels.shape
    mean = voxels[:, :, :3].sum(axis=1, keepdims=True) / np.maximum(num_points, 1)[:, None, None]
    f_cluster = voxels[:, :, :3] - mean
    cx = coords[:, 2].astype(np.float32) * voxel_size[0] + voxel_size[0] / 2 + pc_range[0]
    cy = coords[:, 1].astype(np.float32) * voxel_size[1] + voxel_size[1] / 2 + pc_range[1]
    f_center = np.stack([voxels[:, :, 0] - cx[:, None], voxels[:, :, 1] - cy[:, None]], axis=-1)
    feats = np.concatenate([voxels, f_cluster, f_center], axis=-1)
    mask = np.arange(N)[None, :] < num_points[:, None]
    return (feats * mask[..., None]).astype(np.float32)


def pad_pillars(features, coords, max_pillars):
    """Pad / truncate to a static pillar count. Returns (1, C, P, N) and coords."""
    P, N, C = features.shape
    out = np.zeros((max_pillars, N, C), dtype=np.float32)
    n = min(P, max_pillars)
    out[:n] = features[:n]
    c = np.zeros((max_pillars, 3), dtype=np.int64)
    c[:n] = coords[:n]
    return out.transpose(2, 0, 1)[None], c, n


def scatter_pillars(pillar_features, coords, num_valid, grid_size):
    """(C, P) features -> (1, C, ny, nx) BEV canvas."""
    nx, ny = int(grid_size[0]), int(grid_size[1])
    C = pillar_features.shape[0]
    canvas = np.zeros((C, ny * nx), dtype=np.float32)
    idx = coords[:num_valid, 1] * nx + coords[:num_valid, 2]
    canvas[:, idx] = pillar_features[:, :num_valid]
    return canvas.reshape(1, C, ny, nx)


def split_head_outputs(outputs, head_keys, num_tasks):
    """Flat tuple of (1, C, H, W) maps -> list of per-task dicts of (C, H, W)."""
    k = len(head_keys)
    return [{key: np.asarray(outputs[t * k + i])[0] for i, key in enumerate(head_keys)}
            for t in range(num_tasks)]
