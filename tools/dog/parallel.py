"""Run one function per sequence in parallel processes: the sequences are independent, so a
machine with N cores processes about N of them at once.

    for result in run_each(fn, items, jobs):     # results in the order of ``items``
        ...

``jobs`` <= 0 or None: one process per core, at most one per item; 1: in this process (no
subprocess, easiest to debug). Each worker gets ``threads()`` = cores // jobs threads for the
parts that use several (KISS-ICP, KD-tree queries), so the machine is not oversubscribed.
Workers are started with "spawn" (KISS-ICP's thread pool and forking do not mix).
"""
import os

_ENV = "DOG_TOOL_THREADS"
_BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def threads(default=-1):
    """Threads this process may use: set by ``run_each`` for its workers, else ``default``
    (-1 / 0: all cores, as scipy's ``workers`` and KISS-ICP's ``max_num_threads`` read them)."""
    return int(os.environ.get(_ENV, default))


def resolve_jobs(jobs, n):
    cores = os.cpu_count() or 1
    jobs = cores if jobs is None or jobs <= 0 else jobs
    return max(1, min(jobs, n))


def run_each(fn, items, jobs=None):
    items = list(items)
    jobs = resolve_jobs(jobs, len(items))
    if jobs == 1:
        for it in items:
            yield fn(it)
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    per_job = str(max(1, (os.cpu_count() or 1) // jobs))
    saved = {k: os.environ.get(k) for k in (_ENV,) + _BLAS}
    os.environ.update({k: per_job for k in (_ENV,) + _BLAS})        # read by the workers when they start
    try:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
            yield from ex.map(fn, items)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
