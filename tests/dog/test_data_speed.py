"""Training data pipeline speed-ups keep the results the same (docs/SERVER_TASKS.md T3 speed)."""
import numpy as np

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel


def test_voxel_lookup_buffer_reuse_matches_a_fresh_one():
    """VoxelGenerator keeps its grid-sized lookup between calls (189 MB for the dog voxel grid, made and
    filled per sample before) and must leave it clean: same voxels as a fresh lookup, call after call."""
    rng = np.random.default_rng(0)
    vg = VoxelGenerator([0.075, 0.075, 0.2], [-40.8, -40.8, -5.0, 40.8, 40.8, 3.0], 10, 2000)
    for n in (5000, 300, 8000):               # the cap (2000 voxels) is hit by the larger clouds
        pts = np.c_[rng.uniform(-45, 45, (n, 2)), rng.uniform(-6, 4, n), rng.uniform(0, 1, (n, 2))]
        pts = pts.astype(np.float32)
        pts[: n // 2, :3] = pts[0, :3] + rng.normal(0, 0.05, (n // 2, 3))    # a dense blob: full voxels
        got = vg.generate(pts)
        want = points_to_voxel(pts, vg.voxel_size, vg.point_cloud_range, 10, True, 2000)
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
        assert (vg._coor_to_voxelidx == -1).all()


def test_sweep_jitter_draws_from_the_worker_seeded_generator():
    """Without a seed SweepPoseJitter uses numpy's global generator, which PyTorch seeds per DataLoader
    worker; a Generator made in __init__ was copied into every worker with the same state."""
    from det3d.datasets.pipelines.dog_aug import SweepPoseJitter

    def run(aug, np_seed):
        np.random.seed(np_seed)
        pts = np.c_[np.ones((10, 3)), np.zeros(10), np.full(10, 0.2)].astype(np.float32)
        res = {"mode": "train", "lidar": {"points": pts.copy()}}
        return aug(res, {})[0]["lidar"]["points"]

    aug = SweepPoseJitter(prob=1.0)
    assert not np.allclose(run(aug, 1), run(aug, 2)) and np.allclose(run(aug, 1), run(aug, 1))
    seeded = [SweepPoseJitter(prob=1.0, seed=3) for _ in range(2)]
    assert np.allclose(run(seeded[0], 1), run(seeded[1], 2))


def test_point_shuffle_is_a_permutation():
    from det3d.datasets.pipelines.preprocess import Preprocess
    from det3d.torchie import Config

    pts = np.random.default_rng(1).normal(size=(1000, 5)).astype(np.float32)
    res = {"type": "DogDataset", "lidar": {"points": pts, "combined": pts}}
    out = Preprocess(cfg=Config(dict(mode="val", shuffle_points=True)))(res, {})[0]["lidar"]["points"]
    assert not np.array_equal(out, pts)
    assert np.array_equal(out[np.lexsort(out.T)], pts[np.lexsort(pts.T)])
