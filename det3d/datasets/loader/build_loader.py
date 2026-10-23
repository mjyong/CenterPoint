import platform
from functools import partial

from det3d.torchie.parallel import collate, collate_kitti
from det3d.torchie.trainer import get_dist_info
import torch
from torch.utils.data import DataLoader

from .sampler import (
    DistributedGroupSampler,
    DistributedSampler,
    DistributedSamplerV2,
    GroupSampler,
)

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def _single_thread_worker(worker_id):
    """Each DataLoader worker computes on one core. Otherwise every worker's numpy (OpenBLAS, e.g. the
    points @ rotation in the augmentation) and torch start as many threads as the machine has cores,
    and 8 workers per training, several trainings per server, fight over them."""
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def build_dataloader(
    dataset, batch_size, workers_per_gpu, num_gpus=1, dist=True, **kwargs
):
    shuffle = kwargs.get("shuffle", True)
    if dist:
        rank, world_size = get_dist_info()
        # sampler = DistributedSamplerV2(dataset,
        #                      num_replicas=world_size,
        #                      rank=rank,
        #                      shuffle=shuffle)
        if shuffle:
            sampler = DistributedGroupSampler(dataset, batch_size, world_size, rank)
        else:
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        batch_size = batch_size
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, batch_size) if shuffle else None
        sampler = None
        batch_size = num_gpus * batch_size
        num_workers = num_gpus * workers_per_gpu

    extra = {}
    if num_workers > 0:
        # persistent: workers (their numba-compiled voxelization, their voxel lookup buffer) are kept
        # across epochs instead of being started again
        extra = dict(worker_init_fn=_single_thread_worker, persistent_workers=True)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_kitti,
        # a voxel batch is ~80 MB; from pinned memory it reaches the GPU without a staging copy
        pin_memory=torch.cuda.is_available(),
        **extra
    )

    return data_loader
