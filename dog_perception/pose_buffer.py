"""Time-indexed body poses for per-point motion compensation.

On a legged robot the gait makes the body pitch/roll at several Hz, so the
poses used for deskewing must come at IMU rate (>=200 Hz). Linearly
interpolating 10 Hz LIO poses would smooth exactly the motion we want to
remove. ``ImuPropagator`` turns low-rate LIO odometry + high-rate gyro into
such an IMU-rate pose stream and writes it into a ``PoseBuffer``.
"""
from collections import deque

import numpy as np

from .geometry import make_T, so3_exp, so3_log


class PoseOutOfRange(RuntimeError):
    pass


class PoseBuffer:
    """Stores ``T_world_body(t)`` samples and interpolates them.

    Rotation is SLERP'd (exactly, per point) and translation is linearly
    interpolated. Queries slightly outside the buffered span are linearly
    extrapolated up to ``max_extrapolation`` seconds.
    """

    def __init__(self, max_age=3.0, max_extrapolation=0.05):
        self.max_age = max_age
        self.max_extrapolation = max_extrapolation
        self._t, self._R, self._p = [], [], []
        self._cache = None

    def __len__(self):
        return len(self._t)

    @property
    def t_min(self):
        return self._t[0] if self._t else None

    @property
    def t_max(self):
        return self._t[-1] if self._t else None

    def add(self, t, R, p):
        """Append a sample. Out-of-order samples are ignored (returns False)."""
        t = float(t)
        if self._t and t <= self._t[-1]:
            return False
        self._t.append(t)
        self._R.append(np.asarray(R, dtype=np.float64).reshape(3, 3))
        self._p.append(np.asarray(p, dtype=np.float64).reshape(3))
        self._prune(t - self.max_age)
        self._cache = None
        return True

    def add_T(self, t, T):
        return self.add(t, T[:3, :3], T[:3, 3])

    def truncate_from(self, t):
        """Drop every sample with time >= t (used when re-propagating)."""
        n = int(np.searchsorted(np.asarray(self._t), t, side="left"))
        if n < len(self._t):
            del self._t[n:], self._R[n:], self._p[n:]
            self._cache = None

    def _prune(self, t_old):
        n = int(np.searchsorted(np.asarray(self._t), t_old, side="left"))
        n = min(n, len(self._t) - 2)
        if n > 0:
            del self._t[:n], self._R[:n], self._p[:n]

    def _arrays(self):
        if self._cache is None:
            t = np.asarray(self._t)
            R = np.stack(self._R)
            p = np.stack(self._p)
            if len(t) > 1:
                seg = so3_log(np.einsum("nji,njk->nik", R[:-1], R[1:]))
            else:
                seg = np.zeros((0, 3))
            self._cache = (t, R, p, seg)
        return self._cache

    def interpolate(self, times):
        """Return (R (N,3,3), p (N,3)) of T_world_body at ``times``."""
        if not self._t:
            raise PoseOutOfRange("pose buffer is empty")
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        t, R, p, seg = self._arrays()
        lo, hi = t[0] - self.max_extrapolation, t[-1] + self.max_extrapolation
        if times.min() < lo or times.max() > hi:
            raise PoseOutOfRange(
                "query [%.4f, %.4f] outside buffer [%.4f, %.4f]"
                % (times.min(), times.max(), t[0], t[-1])
            )
        if len(t) == 1:
            n = len(times)
            return np.repeat(R, n, axis=0), np.repeat(p, n, axis=0)

        idx = np.clip(np.searchsorted(t, times, side="right") - 1, 0, len(t) - 2)
        frac = (times - t[idx]) / (t[idx + 1] - t[idx])
        R_out = np.einsum("nij,njk->nik", R[idx], so3_exp(frac[:, None] * seg[idx]))
        p_out = p[idx] + frac[:, None] * (p[idx + 1] - p[idx])
        return R_out, p_out

    def pose_at(self, t):
        R, p = self.interpolate([t])
        return make_T(R[0], p[0])


class ImuPropagator:
    """Builds an IMU-rate ``T_world_body`` stream from LIO odometry + gyro.

    * rotation between odometry updates: integrated from (bias-corrected) gyro,
    * translation: odometry position + world-frame velocity * dt.

    LIO odometry for a scan usually arrives *after* newer IMU samples, so on
    every odometry message the buffered tail is re-propagated from the fresh
    anchor (same scheme FAST-LIO uses internally).
    """

    def __init__(self, buffer, gyro_bias=(0.0, 0.0, 0.0), imu_history=1.0):
        self.buffer = buffer
        self.gyro_bias = np.asarray(gyro_bias, dtype=np.float64)
        self.imu_history = imu_history
        self._imu = deque()
        self._anchor = None
        self._last = None
        self._v_est = None

    def on_odometry(self, t, R_wb, p_wb, v_w=None, gyro_bias=None):
        """``v_w``: linear velocity in the world frame. If the LIO does not
        publish a twist (FAST-LIO2 leaves it empty), it is estimated by finite
        differences of consecutive odometry positions."""
        t = float(t)
        if self._anchor is not None and t <= self._anchor[0]:
            return      # stale / out-of-order odometry must not roll the anchor back
        p_wb = np.asarray(p_wb, dtype=np.float64)
        if gyro_bias is not None:
            self.gyro_bias = np.asarray(gyro_bias, dtype=np.float64)
        if v_w is None:
            if self._anchor is not None and t > self._anchor[0]:
                v_fd = (p_wb - self._anchor[2]) / (t - self._anchor[0])
                self._v_est = v_fd if self._v_est is None else 0.5 * self._v_est + 0.5 * v_fd
            v_w = self._v_est if self._v_est is not None else np.zeros(3)
        self._anchor = (t, np.asarray(R_wb, dtype=np.float64), p_wb, np.asarray(v_w, dtype=np.float64))

        self.buffer.truncate_from(t)
        self.buffer.add(t, R_wb, p_wb)
        self._last = (t, self._anchor[1])
        for ti, w in self._imu:
            if ti > t:
                self._propagate(ti, w)

    def on_imu(self, t, gyro):
        t = float(t)
        gyro = np.asarray(gyro, dtype=np.float64)
        self._imu.append((t, gyro))
        while self._imu and self._imu[0][0] < t - self.imu_history:
            self._imu.popleft()
        if self._anchor is None or t <= self._last[0]:
            return
        self._propagate(t, gyro)

    def _propagate(self, t, gyro):
        t_prev, R_prev = self._last
        R = R_prev @ so3_exp((gyro - self.gyro_bias) * (t - t_prev))
        t0, _, p0, v0 = self._anchor
        self.buffer.add(t, R, p0 + v0 * (t - t0))
        self._last = (t, R)
