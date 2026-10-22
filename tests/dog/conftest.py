"""Shared fixtures. Run everything: ``pytest tests/dog -n 4`` (pytest-xdist); skip the slow
end-to-end tests: ``-m "not slow"``. Each test process uses ``DOG_TEST_THREADS`` CPU threads
(default: cores / workers, at most 4): numpy, torch, numba and the KD-trees otherwise start one
thread per core in every worker, which on a large shared server makes the suite many times slower."""
import os
import sys


def _test_threads():
    if os.environ.get("DOG_TEST_THREADS"):
        return int(os.environ["DOG_TEST_THREADS"])
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    return max(1, min(4, (os.cpu_count() or 1) // workers))


_THREADS = str(_test_threads())
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "DOG_TOOL_THREADS"):
    os.environ.setdefault(_k, _THREADS)     # before numpy / numba / torch are imported

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def det3d_available():
    try:
        import det3d.models  # noqa: F401
        return True
    except Exception:
        return False


requires_det3d = pytest.mark.skipif(not det3d_available(), reason="det3d / torch not importable")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end test taking > 5 s (sample bags, training); "
                                       "skip with -m 'not slow'")
    try:
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    except Exception:
        pass


@pytest.fixture(scope="session")
def sim_frames():
    """A few preprocessed frames from the simulator (exact 400 Hz poses)."""
    from dog_perception.pose_buffer import PoseBuffer
    from dog_perception.preprocess import DetFrameConfig, PreprocessConfig, Preprocessor
    from dog_perception.sim import GaitModel, make_scenario

    sim = make_scenario(seed=0, duration=2.0, gait=GaitModel(speed=1.2, yaw_rate=0.3, pitch_amp_deg=6.0))
    buf = PoseBuffer()
    ts = np.arange(0.0, 1.0, 1 / 400.0)
    R, p = sim.robot.pose(ts)
    for i in range(len(ts)):
        buf.add(ts[i], R[i], p[i])
    pre = Preprocessor(PreprocessConfig(T_body_lidar=sim.T_bl, det_frame=DetFrameConfig(base_height=sim.robot.g.base_height)), buf)
    frames, scans, infos = [], [], []
    for k in range(6):
        scan, info = sim.scan(k)
        frames.append(pre.process(scan))
        scans.append(scan)
        infos.append(info)
    return dict(sim=sim, buffer=buf, pre=pre, frames=frames, scans=scans, infos=infos)
