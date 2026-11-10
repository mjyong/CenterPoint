"""Yield gate: may the robot start across (or keep going along) its path now, given tracked vehicles and
their predictions? Reference implementation of docs/PLANNING_INTEGRATION.md section 4; Nav_D20/planning
ports it to C++ (Tracking, through the mission's pause() / resume()), and the two are compared frame by frame.

Frame agnostic: the path, the objects and their predictions are in one 2D frame (planning: base_link;
offline evaluation: world). Two parts, so the offline evaluation can assess each frame once and replay the
state machine from many arrival times:

    a = assess(path_xy, objects, now, stamp, cfg)      # stateless: what is in the way, and when
    d = gate.update(a, now, dog_moving)                # YieldGate: hysteresis, timers -> go / wait

``objects``: dicts with ``id``, ``label`` (str), ``position`` (>= 2), ``velocity`` (2), ``yaw``, ``size``
(l, w, ...), ``motion`` ("moving" / "stopped" / "static"), optional ``age`` [s], ``covariance`` (4x4 over
[x, y, vx, vy]) and the prediction: ``pred_times`` (T,) seconds after ``stamp``, ``pred_modes`` (K, T, 2),
``pred_probs`` (K,), optional ``pred_covs`` (K, T, 2, 2). No prediction: constant velocity. Optional
``standing_for`` [s]: how long the tracker has seen it standing (else a "static" object's ``age``, else the
gate's own timer from when it first sees it standing).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class YieldConfig:
    classes: tuple = ("vehicle", "cyclist")    # the robot yields to these; the rest is the local planner's
    dog_speed: float = 1.0          # [m/s] along the path while crossing
    dog_length: float = 1.0         # [m] D20 robot_param.pb.txt
    dog_width: float = 0.4
    lateral_margin: float = 0.5     # [m] each side of the robot: the corridor is width + 2 * this
    vehicle_margin: float = 0.5     # [m] each side of a vehicle (length and width each + 1.0 m)
    sigma_k: float = 1.0            # the vehicle box also grows by k * sigma of its predicted position
    min_mode_prob: float = 0.1      # prediction modes below this are dropped (the most likely one never)
    horizon: float = 8.0            # [s] vehicles are followed at least this far (and until the robot has left
                                    # the checked path, plus pet); constant velocity past the prediction
    sigma_growth: float = 0.3       # [m/s] position sigma growth per second past the prediction
    sigma_max: float = 1.0          # [m] cap: the IMM's 3 s sigma is 3-5 m (the tracker's process noise, not
                                    # calibrated for prediction); the gap (pet) covers speed changes
    dt: float = 0.1                 # [s] time step of the check
    path_step: float = 0.1          # [m] the path is resampled at this
    path_length: float = 10.0       # [m] of the path ahead that is checked
    pet: float = 2.0                # [s] robot and vehicle must pass any piece of the path this far apart
    clear_time: float = 2.0         # [s] no conflict this long before going again
    stopped_speed: float = 0.5      # [m/s] slower (and not "moving" in the tracker): a standing vehicle
    stopped_wait: float = 3.0       # [s] a standing vehicle near the path must be seen standing this long
    start_zone: float = 10.0        # [m] a standing vehicle this close to the path, nose or tail towards it
    stale_after: float = 0.3        # [s] perception older than this: wait
    # T44 (1104): vehicles out of the rear blind sector or round the corner were tracked only moments before the
    # robot went, or turned into its path
    young_age: float = 1.0          # [s] a vehicle track younger than this (velocity not settled) ...
    young_range: float = 30.0       # [m] ... this close to the path: wait (0: off)
    turn_range: float = 25.0        # [m] a moving vehicle this close to the path may also turn left / right (0: off)
    turn_radius: float = 10.0       # [m] of those turns, at the vehicle's speed ...
    turn_max_deg: float = 90.0      # ... up to this much, then straight on


@dataclass
class Assessment:
    now: float
    stale: bool
    conflicts: list = field(default_factory=list)   # dicts: id, gap [s] (nan: young), s [m], kind
                                                    # (predicted / turn / young)
    standing: list = field(default_factory=list)    # dicts: id, standing_for [s] or None
    approaching: set = field(default_factory=set)   # ids of moving vehicles heading towards the path
    present: set = field(default_factory=set)       # ids of every vehicle looked at
    in_lane: bool = False           # the robot's footprint is in a moving vehicle's swept body
    min_gap: float = float("inf")   # [s] smallest gap of any moving vehicle (conflict or not)


@dataclass
class Decision:
    go: bool
    reason: str                     # clear / committed / vehicle / young / standing / clearing / not_passed / stale
    ids: list = field(default_factory=list)
    gap: float = float("inf")


def resample_path(path_xy, step, length):
    """Points every ``step`` metres along the polyline, from its start, at most ``length``; and their s."""
    p = np.asarray(path_xy, dtype=float)[:, :2]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, min(cum[-1], length) + 1e-9, step)
    if len(p) == 1 or cum[-1] == 0:
        return p[:1].copy(), np.zeros(1)
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])]), s


def inside_boxes(points, centers, headings, half_l, half_w):
    """(J, K) bool: point j inside the oriented box k (centers[k], headings[k], half_l[k], half_w[k])."""
    d = points[:, None, :] - centers[None, :, :]
    c, s = np.cos(headings)[None], np.sin(headings)[None]
    return ((np.abs(d[..., 0] * c + d[..., 1] * s) <= np.asarray(half_l)[None])
            & (np.abs(-d[..., 0] * s + d[..., 1] * c) <= np.asarray(half_w)[None]))


def occupancy(points, centers, headings, half_l, half_w, times):
    """First and last time each point is inside the oriented box (centers[k], headings[k], half_l[k],
    half_w[k]) at times[k]; NaN where never. points (J, 2); centers (K, 2); the rest (K,)."""
    hit = inside_boxes(points, centers, headings, half_l, half_w)
    t = np.broadcast_to(np.asarray(times, dtype=float)[None], hit.shape)
    first = np.where(hit, t, np.inf).min(1)
    last = np.where(hit, t, -np.inf).max(1)
    never = ~hit.any(1)
    first[never] = np.nan
    last[never] = np.nan
    return first, last


def robot_windows(s, cfg):
    """[s] when the robot, leaving now at ``dog_speed``, covers each path point (its length along the path)."""
    return (np.maximum(0.0, s - 0.5 * cfg.dog_length) / cfg.dog_speed,
            (s + 0.5 * cfg.dog_length) / cfg.dog_speed)


def gaps(first, last, d_in, d_out):
    """Time gap between the vehicle's and the robot's pass of each point (negative: both there at once);
    +inf where the vehicle never comes."""
    g = np.maximum(first - d_out, d_in - last)
    return np.where(np.isnan(first), np.inf, g)


def _lam_max(cov):
    a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    return 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0.0))


def rollout(obj, now, stamp, cfg, horizon=None):
    """Modes of the object at the check times now + k * dt up to ``horizon`` (default ``cfg.horizon``): list
    of (centers (K, 2), headings (K,), sigma (K,)), most likely first; and the check times (K,) from now."""
    tau = np.arange(0.0, (cfg.horizon if horizon is None else horizon) + 1e-9, cfg.dt)
    q = tau + max(0.0, now - stamp)                       # seconds after the perception stamp
    p0 = np.asarray(obj["position"], dtype=float)[:2]
    v0 = np.asarray(obj["velocity"], dtype=float)[:2]
    cov0 = np.zeros((2, 2))
    if obj.get("covariance") is not None:
        cov0 = np.asarray(obj["covariance"], dtype=float)[:2, :2]
    times = obj.get("pred_times")
    if times is None or len(times) == 0:
        times, modes, probs, covs = np.zeros(0), np.zeros((1, 0, 2)), np.ones(1), None
    else:
        times = np.asarray(times, dtype=float)
        modes = np.asarray(obj["pred_modes"], dtype=float)
        probs = np.asarray(obj["pred_probs"], dtype=float)
        covs = obj.get("pred_covs")
    order = np.argsort(-probs)
    keep = [order[0]] + [k for k in order[1:] if probs[k] >= cfg.min_mode_prob]
    out = []
    for k in keep:
        t = np.concatenate([[0.0], times])
        m = np.vstack([p0[None], modes[k]]) if len(times) else p0[None]
        c = np.concatenate([cov0[None], np.asarray(covs[k], dtype=float)]) if covs is not None and len(times) \
            else np.repeat(cov0[None], len(t), 0)
        x = np.interp(q, t, m[:, 0])
        y = np.interp(q, t, m[:, 1])
        var = np.interp(q, t, _lam_max(c))
        if len(times) >= 1:
            v_end = (m[-1] - m[-2]) / max(t[-1] - t[-2], 1e-6)
        else:
            v_end = v0
        past = q > t[-1]
        x[past] = m[-1, 0] + v_end[0] * (q[past] - t[-1])
        y[past] = m[-1, 1] + v_end[1] * (q[past] - t[-1])
        var[past] = var[past] + (cfg.sigma_growth * (q[past] - t[-1])) ** 2
        var = np.minimum(var, cfg.sigma_max ** 2)
        centers = np.column_stack([x, y])
        step = np.diff(centers, axis=0)
        head = np.arctan2(step[:, 1], step[:, 0])
        head = np.concatenate([head, head[-1:]]) if len(head) else np.array([obj.get("yaw", 0.0)])
        still = np.linalg.norm(np.vstack([step, step[-1:]]) if len(step) else np.zeros((1, 2)), axis=1) < 1e-3
        head[still] = float(obj.get("yaw", 0.0))
        out.append((centers, head, np.sqrt(var)))
    return out, tau


def turn_modes(obj, now, stamp, cfg, horizon, sigma):
    """Left and right turns of radius ``turn_radius`` from the object's position and velocity, up to
    ``turn_max_deg`` then straight on: two (centers, headings, sigma) like ``rollout``'s."""
    tau = np.arange(0.0, horizon + 1e-9, cfg.dt)
    q = tau + max(0.0, now - stamp)
    p0 = np.asarray(obj["position"], dtype=float)[:2]
    v0 = np.asarray(obj["velocity"], dtype=float)[:2]
    speed = float(np.hypot(*v0))
    th0 = float(np.arctan2(v0[1], v0[0]))
    out = []
    for sign in (1.0, -1.0):
        w = sign * speed / cfg.turn_radius
        t_turn = np.radians(cfg.turn_max_deg) / abs(w)
        qt = np.minimum(q, t_turn)
        th = th0 + w * qt
        x = p0[0] + speed / w * (np.sin(th) - np.sin(th0))
        y = p0[1] - speed / w * (np.cos(th) - np.cos(th0))
        rest = np.maximum(q - t_turn, 0.0)
        out.append((np.column_stack([x + speed * rest * np.cos(th), y + speed * rest * np.sin(th)]), th, sigma))
    return out


def _is_moving(obj, cfg):
    return obj.get("motion") == "moving" or float(np.hypot(*obj["velocity"][:2])) >= cfg.stopped_speed


def assess(path_xy, objects, now, stamp, cfg=YieldConfig()):
    """Stateless check of one perception message (``stamp``) at time ``now``, for a robot at the start of
    ``path_xy`` that would leave now along it."""
    a = Assessment(now=now, stale=stamp is None or now - stamp > cfg.stale_after)
    if a.stale:
        return a
    pts, s = resample_path(path_xy, cfg.path_step, cfg.path_length)
    d_in, d_out = robot_windows(s, cfg)
    horizon = max(cfg.horizon, float(d_out[-1]) + cfg.pet)   # a vehicle later than this cannot conflict
    corridor = 0.5 * cfg.dog_width + cfg.lateral_margin
    here = s <= 0.5 * cfg.dog_length                      # under the robot now
    for obj in objects:
        if obj["label"] not in cfg.classes:
            continue
        oid = obj["id"]
        a.present.add(oid)
        size = np.asarray(obj["size"], dtype=float)
        hl0, hw0 = 0.5 * size[0], 0.5 * size[1]
        p = np.asarray(obj["position"], dtype=float)[:2]
        dist = np.linalg.norm(pts - p, axis=1)
        conflicted = False
        if _is_moving(obj, cfg):
            modes, tau = rollout(obj, now, stamp, cfg, horizon)
            kinds = ["predicted"] * len(modes)
            if cfg.turn_range > 0 and dist.min() <= cfg.turn_range:
                modes += turn_modes(obj, now, stamp, cfg, horizon, modes[0][2])
                kinds += ["turn", "turn"]
            worst = (np.inf, None, None)
            for (centers, head, sigma), kind in zip(modes, kinds):
                grow = cfg.vehicle_margin + cfg.sigma_k * sigma + corridor
                first, last = occupancy(pts, centers, head, hl0 + grow, hw0 + grow, tau)
                g = gaps(first, last, d_in, d_out)
                j = int(np.argmin(g))
                if g[j] < worst[0]:
                    worst = (float(g[j]), float(s[j]), kind)
                if not a.in_lane and kind == "predicted":
                    raw_first, _ = occupancy(pts[here], centers, head, np.full(len(tau), hl0),
                                             np.full(len(tau), hw0 + 0.5 * cfg.dog_width), tau)
                    a.in_lane = bool(np.any(~np.isnan(raw_first)))
            a.min_gap = min(a.min_gap, worst[0])
            if worst[0] < cfg.pet:
                a.conflicts.append(dict(id=oid, gap=worst[0], s=worst[1], kind=worst[2]))
                conflicted = True
            r = pts[int(np.argmin(dist))] - p
            if float(np.dot(np.asarray(obj["velocity"], dtype=float)[:2], r)) > 0:
                a.approaching.add(oid)
        else:
            # standing: in the corridor, or nose / tail towards it within start_zone
            yaw = float(obj.get("yaw", 0.0))
            d = pts - p
            along = d[:, 0] * np.cos(yaw) + d[:, 1] * np.sin(yaw)
            across = -d[:, 0] * np.sin(yaw) + d[:, 1] * np.cos(yaw)
            near = np.abs(across) <= hw0 + cfg.vehicle_margin + corridor
            if np.any(near & (np.abs(along) <= hl0 + cfg.start_zone)):
                standing_for = obj.get("standing_for")
                if standing_for is None and obj.get("motion") == "static":
                    standing_for = obj.get("age")
                a.standing.append(dict(id=oid, standing_for=standing_for))
        # a new track: its velocity (and so its prediction) is not to be trusted yet
        if (not conflicted and cfg.young_range > 0 and obj.get("age") is not None and obj["age"] < cfg.young_age
                and dist.min() <= cfg.young_range):
            a.conflicts.append(dict(id=oid, gap=float("nan"), s=float(s[int(np.argmin(dist))]), kind="young"))
    return a


class YieldGate:
    """Hysteresis on top of ``assess``: wait at once on a conflict; go again only after ``clear_time``
    without one, when every vehicle waited for has passed (moving away), gone, or stood ``stopped_wait``."""

    def __init__(self, cfg=YieldConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.paused = False
        self.last_conflict = -np.inf
        self.yielded = set()
        self.still_since = {}           # id -> first time seen standing (continuously)

    def update(self, a, now, dog_moving=False):
        cfg = self.cfg
        if a.stale:
            self.paused = True
            self.last_conflict = now
            return Decision(False, "stale")
        standing_ids = {st["id"] for st in a.standing}
        for oid in list(self.still_since):
            if oid not in standing_ids:
                del self.still_since[oid]
        waiting_for = []
        for st in a.standing:
            t0 = self.still_since.setdefault(
                st["id"], now - st["standing_for"] if st["standing_for"] is not None else now)
            if now - t0 < cfg.stopped_wait:
                waiting_for.append(st["id"])
        conflict_ids = [c["id"] for c in a.conflicts]
        gap = min([c["gap"] for c in a.conflicts if c["gap"] == c["gap"]], default=a.min_gap)
        if dog_moving and not self.paused and a.in_lane:
            return Decision(True, "committed", conflict_ids, gap)     # never stop in front of a vehicle
        if conflict_ids or waiting_for:
            self.paused = True
            self.last_conflict = now
            self.yielded.update(conflict_ids + waiting_for)
            reason = ("young" if all(c["kind"] == "young" for c in a.conflicts) else "vehicle") if conflict_ids \
                else "standing"
            return Decision(False, reason, conflict_ids + waiting_for, gap)
        if self.paused:
            if now - self.last_conflict < cfg.clear_time:
                return Decision(False, "clearing", sorted(self.yielded), gap)
            pending = sorted(i for i in self.yielded if i in a.present and i in a.approaching)
            if pending:
                return Decision(False, "not_passed", pending, gap)
            self.paused = False
            self.yielded.clear()
        return Decision(True, "clear", [], gap)


def required_sight_range(crossing_length, v_vehicle, latency, cfg=YieldConfig()):
    """[m] how far the robot must see an unseen vehicle coming at ``v_vehicle`` [m/s] to cross
    ``crossing_length`` [m] with ``cfg.pet`` to spare, perception ``latency`` [s] included: the worst
    conflict point is the far end of the crossing."""
    t_out = (crossing_length + 0.5 * cfg.dog_length) / cfg.dog_speed
    return crossing_length + v_vehicle * (t_out + cfg.pet + latency)
