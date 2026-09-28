"""Minimal XT32-on-a-legged-robot simulator.

Used by the tests and the demo to exercise the pipeline without a robot:

* Hesai XT32 geometry: 32 channels from -16 to +15 deg (1 deg), 2000 azimuth
  steps per revolution at 10 Hz (0.18 deg), per-point firing time, 1 cm noise.
* Rolling-shutter distortion: every azimuth column is cast from the lidar pose
  *at its firing time*, and moving objects are placed at their pose at that
  time too.
* Gait: forward motion + yaw rate + pitch/roll oscillation + body bounce.
* Self hits: swinging legs and a payload box in the body frame.
* Streams: 400 Hz gyro and 10 Hz LIO-like odometry, plus ground truth.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import make_T, so3_exp, so3_log
from .preprocess.pipeline import LidarScan

CLASS_SIZES = {  # l, w, h
    "pedestrian": (0.6, 0.6, 1.75),
    "cyclist": (1.8, 0.6, 1.7),
    "vehicle": (4.5, 1.9, 1.6),
}


@dataclass
class XT32Model:
    elevations_deg: np.ndarray = field(default_factory=lambda: np.linspace(-16.0, 15.0, 32))
    azimuth_steps: int = 2000
    rate_hz: float = 10.0
    range_noise: float = 0.01
    min_range: float = 0.05
    max_range: float = 60.0

    @property
    def period(self):
        return 1.0 / self.rate_hz


@dataclass
class GaitModel:
    speed: float = 1.0            # forward speed [m/s]
    yaw_rate: float = 0.1         # [rad/s]
    yaw0: float = 0.0
    pitch_amp_deg: float = 5.0
    pitch_freq: float = 2.0       # step frequency [Hz]
    roll_amp_deg: float = 3.0
    roll_freq: float = 1.0
    bounce_amp: float = 0.02
    bounce_freq: float = 4.0
    base_height: float = 0.45


class RobotTrajectory:
    def __init__(self, gait=None, start_xy=(0.0, 0.0)):
        self.g = gait or GaitModel()
        self.start = np.asarray(start_xy, dtype=np.float64)

    def pose(self, t):
        """Vectorised T_world_body: returns R (N,3,3), p (N,3)."""
        g = self.g
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        yaw = g.yaw0 + g.yaw_rate * t
        if abs(g.yaw_rate) > 1e-9:
            x = g.speed / g.yaw_rate * (np.sin(yaw) - np.sin(g.yaw0))
            y = -g.speed / g.yaw_rate * (np.cos(yaw) - np.cos(g.yaw0))
        else:
            x = g.speed * t * np.cos(g.yaw0)
            y = g.speed * t * np.sin(g.yaw0)
        z = g.base_height + g.bounce_amp * np.sin(2 * np.pi * g.bounce_freq * t)
        pitch = np.deg2rad(g.pitch_amp_deg) * np.sin(2 * np.pi * g.pitch_freq * t)
        roll = np.deg2rad(g.roll_amp_deg) * np.sin(2 * np.pi * g.roll_freq * t + 0.7)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        R = np.empty((len(t), 3, 3))
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        p = np.stack([x + self.start[0], y + self.start[1], z], axis=1)
        return R, p

    def T(self, t):
        R, p = self.pose([t])
        return make_T(R[0], p[0])

    def gyro(self, t, h=1e-4):
        """Body-frame angular velocity."""
        R0, _ = self.pose(t)
        R1, _ = self.pose(np.atleast_1d(t) + h)
        return so3_log(np.einsum("nji,njk->nik", R0, R1)) / h

    def velocity(self, t, h=1e-4):
        _, p0 = self.pose(t)
        _, p1 = self.pose(np.atleast_1d(t) + h)
        return (p1 - p0) / h


class SimObject:
    """Box object driven by piecewise-constant (speed, yaw-rate) segments."""

    def __init__(self, obj_id, label, xy0, heading0, segments, duration, size=None, dt=0.01, static=False):
        self.id = obj_id
        self.label = label
        self.size = np.asarray(size if size is not None else CLASS_SIZES[label], dtype=np.float64)
        self.static = static
        n = int(np.ceil(duration / dt)) + 2
        self._t = np.arange(n) * dt
        seg_t = np.array([s[0] for s in segments])
        idx = np.clip(np.searchsorted(seg_t, self._t, side="right") - 1, 0, len(segments) - 1)
        speed = np.array([s[1] for s in segments])[idx]
        omega = np.array([s[2] for s in segments])[idx]
        yaw = heading0 + np.concatenate([[0.0], np.cumsum(omega[:-1] * dt)])
        vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
        x = xy0[0] + np.concatenate([[0.0], np.cumsum(vx[:-1] * dt)])
        y = xy0[1] + np.concatenate([[0.0], np.cumsum(vy[:-1] * dt)])
        self._x, self._y, self._yaw, self._vx, self._vy = x, y, yaw, vx, vy

    def state(self, t):
        """Vectorised: returns center (N,3), yaw (N,), velocity (N,2)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        f = lambda a: np.interp(t, self._t, a)
        c = np.stack([f(self._x), f(self._y), np.full(len(t), self.size[2] / 2)], axis=1)
        return c, f(self._yaw), np.stack([f(self._vx), f(self._vy)], axis=1)

    @staticmethod
    def random(obj_id, label, rng, duration, center, half_extent):
        spd = {"pedestrian": (0.4, 1.8), "cyclist": (2.0, 5.5), "vehicle": (2.0, 9.0)}[label]
        wr = {"pedestrian": 0.6, "cyclist": 0.35, "vehicle": 0.25}[label]
        segs, t = [], 0.0
        while t < duration:
            s = rng.uniform(*spd)
            if label == "pedestrian" and rng.random() < 0.15:
                s = 0.0
            w = rng.uniform(-wr, wr) if rng.random() < 0.6 else 0.0
            segs.append((t, s, w))
            t += rng.uniform(2.0, 5.0)
        xy0 = np.asarray(center) + rng.uniform(-1.0, 1.0, 2) * np.asarray(half_extent)
        return SimObject(obj_id, label, xy0, rng.uniform(-np.pi, np.pi), segs, duration)


def static_box(obj_id, center_xy, size, yaw=0.0, label="static"):
    return SimObject(obj_id, label, center_xy, yaw, [(0.0, 0.0, 0.0)], duration=1.0, size=size, dt=1.0, static=True)


def _ray_box(o, d, center, yaw, half):
    """Slab test for rays (N,3)+(N,3) against boxes given per ray. Returns t (inf=miss)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rel = o - center
    lo = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], 1)
    ld = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / ld
        t1 = (-half - lo) * inv
        t2 = (half - lo) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-6)
    return np.where(hit, tmin, np.inf)


class Xt32Simulator:
    def __init__(self, robot, objects, lidar=None, T_body_lidar=None, seed=0,
                 payload_box=(-0.55, -0.30, -0.15, 0.15, 0.05, 0.35), legs=True):
        self.robot = robot
        self.objects = objects
        self.lidar = lidar or XT32Model()
        self.T_bl = T_body_lidar if T_body_lidar is not None else make_T(t=[0.2, 0.0, 0.15])
        self.rng = np.random.default_rng(seed)
        self.payload_box = payload_box
        self.legs = legs
        el = np.deg2rad(self.lidar.elevations_deg)
        az = np.arange(self.lidar.azimuth_steps) * 2 * np.pi / self.lidar.azimuth_steps
        A, E = np.meshgrid(az, el, indexing="ij")          # (azimuth, channel)
        self._dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self._az_index = np.repeat(np.arange(self.lidar.azimuth_steps), len(el))

    def _self_boxes(self, t):
        """Body-frame boxes as (center (N,3), yaw 0, half (3,)) per ray time."""
        boxes = []
        if self.payload_box is not None:
            b = self.payload_box
            c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
            half = np.array([(b[1] - b[0]) / 2, (b[3] - b[2]) / 2, (b[5] - b[4]) / 2])
            boxes.append((np.repeat(c[None], len(t), 0), half))
        if self.legs:
            g = self.robot.g
            for i, (hx, hy) in enumerate([(0.3, 0.2), (0.3, -0.2), (-0.3, 0.2), (-0.3, -0.2)]):
                phase = 2 * np.pi * g.pitch_freq / 2 * t + (np.pi if i in (1, 2) else 0.0)
                c = np.stack([hx + 0.12 * np.sin(phase), np.full(len(t), hy), np.full(len(t), -0.25)], 1)
                boxes.append((c, np.array([0.05, 0.05, 0.22])))
        return boxes

    def scan(self, k, t0=0.0):
        """Simulate revolution k covering [t0 + k*T, t0 + (k+1)*T)."""
        L = self.lidar
        t_start = t0 + k * L.period
        t_az = t_start + (np.arange(L.azimuth_steps) + 1) * L.period / L.azimuth_steps
        R_wb, p_wb = self.robot.pose(t_az)
        R_wl = np.einsum("nij,jk->nik", R_wb, self.T_bl[:3, :3])
        o_az = np.einsum("nij,j->ni", R_wb, self.T_bl[:3, 3]) + p_wb
        ai = self._az_index
        d_w = np.einsum("nij,nj->ni", R_wl[ai], self._dirs)
        o_w = o_az[ai]
        t_ray = t_az[ai]

        best = np.full(len(d_w), np.inf)
        hit_id = np.full(len(d_w), -1)
        inten = np.zeros(len(d_w), dtype=np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(d_w[:, 2] < -1e-6, -o_w[:, 2] / d_w[:, 2], np.inf)
        upd = tg < best
        best[upd], hit_id[upd], inten[upd] = tg[upd], -2, 15.0

        for obj in self.objects:
            if obj.static:
                c, yaw, _ = obj.state([0.0])
                c, yaw = np.repeat(c, len(d_w), 0), np.repeat(yaw, len(d_w))
            else:
                c_az, yaw_az, _ = obj.state(t_az)
                c, yaw = c_az[ai], yaw_az[ai]
            th = _ray_box(o_w, d_w, c, yaw, obj.size / 2)
            upd = th < best
            best[upd], hit_id[upd] = th[upd], obj.id
            inten[upd] = 60.0 if obj.static else 40.0

        # self hits, computed in the body frame of each ray
        o_b = np.repeat(self.T_bl[:3, 3][None], len(d_w), 0)
        d_b = self._dirs @ self.T_bl[:3, :3].T
        for c, half in self._self_boxes(t_az):
            th = _ray_box(o_b, d_b, c[ai], np.zeros(len(d_w)), half)
            upd = th < best
            best[upd], hit_id[upd], inten[upd] = th[upd], -3, 100.0

        valid = np.isfinite(best) & (best >= L.min_range) & (best <= L.max_range)
        rng_m = best[valid] + self.rng.normal(0, L.range_noise, valid.sum())
        xyz = self._dirs[valid] * rng_m[:, None]
        inten = np.clip(inten[valid] + self.rng.normal(0, 3, valid.sum()), 0, 255).astype(np.float32)
        scan = LidarScan(xyz=xyz, intensity=inten, point_times=t_ray[valid], stamp=float(t_az[-1]))

        ids, counts = np.unique(hit_id[valid], return_counts=True)
        hits = {int(i): int(c) for i, c in zip(ids, counts) if i >= 0}
        return scan, {"gt": self.ground_truth(scan.stamp, hits), "self_hits": int(np.sum(hit_id[valid] == -3))}

    def ground_truth(self, t, hits=None):
        gt = []
        for obj in self.objects:
            if obj.static:
                continue
            c, yaw, v = obj.state([t])
            gt.append(dict(id=obj.id, label=obj.label, center=c[0], size=obj.size.copy(), yaw=float(yaw[0]),
                           velocity=v[0], num_points=(hits or {}).get(obj.id, 0)))
        return gt

    def imu(self, t0, t1, rate=400.0, gyro_noise=0.002, gyro_bias=(0.0, 0.0, 0.0)):
        ts = np.arange(np.ceil(t0 * rate), np.floor(t1 * rate) + 1) / rate
        w = self.robot.gyro(ts) + np.asarray(gyro_bias) + self.rng.normal(0, gyro_noise, (len(ts), 3))
        return ts, w

    def odometry(self, t, pos_noise=0.0, rot_noise=0.0):
        R, p = self.robot.pose([t])
        R, p = R[0], p[0]
        if pos_noise:
            p = p + self.rng.normal(0, pos_noise, 3)
        if rot_noise:
            R = R @ so3_exp(self.rng.normal(0, rot_noise, 3))
        return R, p, self.robot.velocity([t])[0]


def make_scenario(seed=0, duration=10.0, num_pedestrians=6, num_cyclists=2, num_vehicles=2,
                  gait=None, with_walls=True):
    """A street-like scene around the robot's path."""
    rng = np.random.default_rng(seed)
    robot = RobotTrajectory(gait or GaitModel())
    mid = robot.pose([duration / 2])[1][0, :2]
    objects, oid = [], 0
    for label, n in (("pedestrian", num_pedestrians), ("cyclist", num_cyclists), ("vehicle", num_vehicles)):
        for _ in range(n):
            objects.append(SimObject.random(oid, label, rng, duration + 1.0, (mid[0], 0.0), (25.0, 11.0)))
            oid += 1
    if with_walls:
        objects.append(static_box(1000, (0.0, 14.0), (200.0, 0.3, 3.0)))
        objects.append(static_box(1001, (0.0, -14.0), (200.0, 0.3, 3.0)))
        for i in range(8):
            objects.append(static_box(1100 + i, (-20.0 + 8.0 * i, 9.0 * (-1) ** i), (0.3, 0.3, 2.5)))
    return Xt32Simulator(robot, objects, seed=seed)
