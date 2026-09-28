import os
import sys

import numpy as np
import pytest

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
