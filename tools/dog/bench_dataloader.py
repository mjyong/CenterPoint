"""Can the training data pipeline keep the GPU busy? Three measurements, no GPU needed.

1. Per stage, in this process on one core: milliseconds per sample of each pipeline transform, and
   of the parts of Preprocess / Voxelization that used to dominate (point shuffle, voxelization).
2. Disk: reading frames the pipeline has not read yet (``--read-files`` of them, 8 threads, MB/s).
   On a cold page cache this is what the workers see in the first epoch, or in every epoch when the
   dataset does not fit in memory.
3. The real DataLoader (the config's batch size and workers): batches per second.

Compare 3 with the training's own step time without data loading (``time - data_time`` in the log,
about 0.19 s for the Voxel and 0.15 s for the Pillar vehicle model on an RTX 4090): when the
DataLoader makes batches faster than that, the GPU is the limit.

    python tools/dog/bench_dataloader.py configs/dog/dog_centerpoint_voxel_xt32_veh.py
"""
import argparse
import functools
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _timed(tot, name, f):
    @functools.wraps(f)
    def w(*a, **k):
        t0 = time.perf_counter()
        r = f(*a, **k)
        tot[name] += time.perf_counter() - t0
        return r
    return w


def stages(ds, n):
    """ms per sample of each transform (and of the voxel generator), after 3 warm-up samples."""
    tot = defaultdict(float)
    for t in ds.pipeline.transforms:
        if type(t).__name__ == "Voxelization":
            t.voxel_generator.generate = _timed(tot, "  voxel_generator.generate", t.voxel_generator.generate)
    for k in range(n + 3):
        if k == 3:
            tot.clear()
        info = ds._infos[(k * 7919) % len(ds)]
        res = {"lidar": {"type": "lidar", "points": None, "nsweeps": ds.nsweeps, "annotations": None},
               "metadata": {"image_prefix": ds._root_path, "num_point_features": ds._num_point_features,
                            "token": info["token"]},
               "calib": None, "cam": {}, "mode": "train", "virtual": False}
        for t in ds.pipeline.transforms:
            t0 = time.perf_counter()
            res, info = t(res, info)
            tot[type(t).__name__] += time.perf_counter() - t0
    return {k: 1e3 * v / n for k, v in tot.items()}


def read_speed(ds, n_files, threads=8):
    paths = sorted({os.path.join(str(ds._root_path), i["lidar_path"]) for i in ds._infos})
    rng = np.random.default_rng(int(time.time()))
    pick = [paths[i] for i in rng.choice(len(paths), min(n_files, len(paths)), replace=False)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        nbytes = sum(ex.map(lambda p: np.load(p).nbytes, pick))
    dt = time.perf_counter() - t0
    return len(pick), nbytes / dt / 1e6, nbytes / len(pick) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--samples", type=int, default=20, help="samples for the per-stage timing")
    ap.add_argument("--read-files", type=int, default=200, help="frames read for the disk speed (0: skip)")
    ap.add_argument("--batches", type=int, default=100, help="DataLoader batches timed (after 10 warm-up)")
    ap.add_argument("--workers", type=int, default=None, help="default: the config's workers_per_gpu")
    args = ap.parse_args()

    from det3d.datasets import build_dataloader, build_dataset
    from det3d.torchie import Config

    cfg = Config.fromfile(args.config)
    ds = build_dataset(cfg.data.train)
    print("train set: %d samples" % len(ds))

    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            st = stages(ds, args.samples)
    except ImportError:
        st = stages(ds, args.samples)
    print("1. per stage, one core (ms per sample):")
    for k, v in st.items():
        print("   %-28s %7.1f" % (k, v))
    print("   %-28s %7.1f" % ("total", sum(v for k, v in st.items() if not k.startswith(" "))))

    if args.read_files:
        n, mbs, size = read_speed(ds, args.read_files)
        print("2. disk: %d unread frames (%.1f MB each), 8 threads: %.0f MB/s = %.0f frames/s"
              % (n, size, mbs, mbs / size))

    workers = cfg.data.workers_per_gpu if args.workers is None else args.workers
    loader = build_dataloader(ds, cfg.data.samples_per_gpu, workers, dist=False)

    def batches():
        while True:                     # small datasets: start the next epoch
            yield from loader

    it = batches()
    for _ in range(10):
        next(it)
    t0 = time.perf_counter()
    for _ in range(args.batches):
        next(it)
    dt = (time.perf_counter() - t0) / args.batches
    print("3. DataLoader (batch %d, %d workers): %.3f s per batch, %.1f samples/s"
          % (cfg.data.samples_per_gpu, workers, dt, cfg.data.samples_per_gpu / dt))


if __name__ == "__main__":
    main()
