"""tools/dog/parallel.py and the threaded scan writer: same results, only faster."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import parallel  # noqa: E402

from dog_perception.io import SequenceWriter  # noqa: E402
from dog_perception.preprocess.pipeline import LidarScan  # noqa: E402


def test_run_each_keeps_the_order_and_splits_the_cores():
    assert list(parallel.run_each(abs, [-3, 1, -2], jobs=1)) == [3, 1, 2]
    per_job = str(max(1, (os.cpu_count() or 1) // 2))
    assert list(parallel.run_each(os.getenv, ["DOG_TOOL_THREADS"] * 3, jobs=2)) == [per_job] * 3
    assert os.getenv("DOG_TOOL_THREADS") is None and parallel.threads() == -1        # parent untouched
    assert parallel.resolve_jobs(0, 3) == min(3, os.cpu_count() or 1) and parallel.resolve_jobs(8, 2) == 2


def test_threaded_writer_writes_the_same_scans(tmp_path):
    rng = np.random.default_rng(0)
    scans = [LidarScan(xyz=rng.normal(size=(500, 3)), intensity=rng.random(500), point_times=k + 0.1 * rng.random(500),
                       stamp=k + 0.1) for k in range(12)]
    with ThreadPoolExecutor(4) as pool:
        for d, p in (("serial", None), ("threads", pool)):
            w = SequenceWriter(str(tmp_path / d), pool, max_pending=3)
            for s in scans:
                w.add_scan(s)
            w.close({})
    for k in range(12):
        a, b = (np.load(tmp_path / d / "scans" / ("%06d.npz" % k)) for d in ("serial", "threads"))
        assert all(np.array_equal(a[f], b[f]) for f in ("xyz", "intensity", "t", "stamp"))
