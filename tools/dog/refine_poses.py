"""Refine the localization of converted sequences with lidar odometry (KISS-ICP) anchored to RTK.

    python tools/dog/refine_poses.py --seq data/seq/0921_* [--plots data/preview/poses]

Why: the kygi INS position is right while the robot walks steadily, but when it brakes hard the
INS velocity overshoots (2.5 m/s) and the position runs ~0.5 m ahead for a second or more; the
backward steps that follow only partly correct it (moving sample bag: 0.44 m ahead at 2.1 s,
measured by registering lidar frames). And its position is that of a point 0.54 m behind the
lidar, which swings around the robot in every turn. Sweeps stacked across those times show ghosts,
and auto-labels that pin a static object in the world frame land up to ~0.5 m off the points there.

How, per sequence:

0. INS lever arm: the INSPVAX position is that of a point 0.54 m behind the lidar (the body
   origin), so it swings around the robot whenever it turns (dog_perception/inspvax.py). The
   rows are moved to the body origin (``--lever-arm``; default: what the converter applied,
   ``meta["ins_lever_arm"]``, or the value measured on 0921 for sequences converted before it). The lever arm
   left over is fitted per sequence (``lever_residual_m``, sequences turning > 45 deg) as a check.
1. Lidar odometry: every scan (same de-duplication and self filter as the online preprocessing,
   body frame, per-point times) goes through KISS-ICP, which deskews it to the scan end (our
   stamp) and registers it against its local map. Its motion prediction gets the rotation from
   the gyro between the two scans (``--prior imu``): with its own constant-velocity prediction
   it jumps back and forth when a turn starts or stops.
2. Health of both sources. INS: over 0.5 s windows its position must agree with its own
   velocity integral (not while braking). Odometry (``lo_health``): no step faster than 3.5 m/s,
   no zigzag (path more than twice the displacement over 0.6 s), no stall (less than 30 % of the
   speed of a healthy INS).
3. Fusion, separately for east and north: positions x_k at the scan stamps minimise

       sum_k v_k (x_k+1 - x_k - d_k)^2 / sigma_lo^2  +  sum_k w_k (x_k - p_ins,k)^2 / sigma_ins^2

   with d_k the odometry step turned into ENU with the fused heading: the odometry heading plus a
   slowly varying offset to the INS heading (``heading_offset``, time constant 10 s). After a fast
   turn the INS heading is off by up to 10 deg for 3-5 s while its position stays
   self-consistent; the odometry heading is right from scan to scan.
   Both healthy: the odometry carries the motion and the INS (sigma_ins 1 m) only removes its
   slow drift - on the 0921 data the INS is 0.2-0.5 m off at the 1 s scale even where it is
   self-consistent, the odometry is not. INS unhealthy: w_k = 0. A run of unhealthy odometry
   steps: the INS velocity where the INS is healthy along the whole run, else the net odometry
   motion over the run spread evenly (``sigma_lo_bad``).
4. Self-check: an INS gap whose odometry misses the INS at its far end by more than
   ``--gap-tol`` + ``--gap-rel`` x its length gets its odometry steps down-weighted (kept, not
   replaced), so the miss is absorbed inside the gap; such gaps and stretches where neither
   source is healthy are listed as bad in meta.json (do not label those frames). Gaps touching
   the start or end cannot be checked and are listed as unverified.
5. Output: odom.npy keeps the INS rows (100 Hz times, heading-only attitude, height), at the body
   origin, with east / north and the heading replaced by the fused ones (linear between scan stamps, so the
   IMU propagation and deskew work exactly as before) and the velocity by its derivative. The INS
   original is kept once as odom_ins.npy and is always the input, so re-running is safe.
   meta.json gets ``pose_source = "kiss-icp+ins"`` and ``pose_refine`` (correction statistics,
   heading correction, health shares, lever arm, bad and unverified intervals, the per-source pose
   checks).
6. With ``--plots``: per sequence a plot (trajectory; correction and health; the lidar shift
   check for INS / odometry / fused poses) and ``<seq>_diag.npz``; ``refine_summary.csv`` over all.

Check the result with tools/dog/sequence_preview.py: its ``shift_p90_m`` (the lidar-measured
position error between frames 1 s apart) must drop; on the moving sample bag 0.51 -> 0.12 m.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # sequence_preview (pose checks)

from dog_perception.geometry import make_T, quat_to_R, rot_z, so3_exp, transform_points, yaw_of  # noqa: E402
from dog_perception.inspvax import KYGI_INS_LEVER_ARM, apply_lever_arm  # noqa: E402
from dog_perception.io import SequenceReader, load_scan  # noqa: E402
from dog_perception.preprocess import SelfFilter, SelfFilterConfig, dedup_returns  # noqa: E402
from parallel import run_each, threads  # noqa: E402

ODOM_INS = "odom_ins.npy"


def scene_points(xyz_body, base_height, cell=0.05):
    """Points 0.3-3 m above the ground and 1.5-30 m away (walls, poles, trees, cars), one per 5 cm
    cell, body frame (M, 3) float32: what the pose checks register."""
    r = np.hypot(xyz_body[:, 0], xyz_body[:, 1])
    z = xyz_body[:, 2] + base_height
    p = xyz_body[(z > 0.3) & (z < 3.0) & (r > 1.5) & (r < 30)]
    k = np.unique(np.floor(p / cell).astype(np.int64), axis=0, return_index=True)[1]
    return p[k].astype(np.float32)


def gyro_rotation(imu, t0, t1):
    """Body rotation from t0 to t1 integrated from the gyro rows (t, wx, wy, wz) [rad/s]:
    R_world_body(t1) = R_world_body(t0) @ R. Identity without IMU rows."""
    R = np.eye(3)
    if len(imu) == 0 or t1 <= t0:
        return R
    m = (imu[:, 0] > t0) & (imu[:, 0] < t1)
    ts = np.concatenate([[t0], imu[m, 0], [t1]])
    k = np.clip(np.searchsorted(imu[:, 0], ts[:-1], side="right") - 1, 0, len(imu) - 1)
    for w, dt in zip(imu[k, 1:4], np.diff(ts)):
        R = R @ so3_exp(w * dt)
    return R


def lidar_odometry(rd, voxel=0.4, max_range=60.0, min_range=1.5, keep_every=10, prior="imu"):
    """(stamps (K,), poses (K, 4, 4), {index: scene points}): KISS-ICP body poses at the scan
    ends, in the frame of the first scan, plus the registration points of every ``keep_every``-th
    scan for the pose checks.

    KISS-ICP predicts each scan's motion from the previous one (constant velocity), for the
    initial guess and for deskewing. That fails when a turn starts or stops: on the 0921 data its
    steps jump (0.5-1 m in 0.1 s) in 4 % of the scans below 5 deg/s yaw rate and in 53 % above
    60 deg/s. With ``prior="imu"`` the predicted rotation comes from the gyro integrated between
    the two scans instead (translation stays constant-velocity in the body frame)."""
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = max_range, min_range, True
    cfg.mapping.voxel_size = voxel
    cfg.registration.max_num_threads = max(0, threads(0))      # 0: all cores; fewer when sequences run in parallel
    odo = KissICP(cfg)
    sf = SelfFilter(SelfFilterConfig(), rd.T_body_lidar)
    stamps, poses, kept = [], [], {}
    base = float(rd.meta.get("base_height", 0.45))
    for k, f in enumerate(rd.scan_files):
        scan = dedup_returns(load_scan(f).shifted(rd.lidar_time_offset))
        if prior == "imu" and stamps:
            delta = odo.last_delta.copy()              # last estimated motion, body frame
            delta[:3, :3] = gyro_rotation(rd.imu, stamps[-1], scan.stamp)
            odo.last_delta = delta
        keep = np.isfinite(scan.xyz).all(axis=1)
        keep[keep] = sf(scan.xyz[keep])
        xyz = transform_points(rd.T_body_lidar, scan.xyz[keep]).astype(np.float64)
        frame, _ = odo.register_frame(xyz, np.asarray(scan.point_times, np.float64)[keep])
        stamps.append(scan.stamp)
        poses.append(odo.last_pose.copy())
        if keep_every and k % keep_every == 0:
            kept[k] = scene_points(np.asarray(frame), base)          # deskewed to the scan end
    return np.array(stamps), np.array(poses), kept


def ins_at(odom, t):
    """INS east / north / up and unwrapped heading at times ``t`` (linear interpolation)."""
    yaw = np.unwrap([yaw_of(quat_to_R(q)) for q in odom[:, 4:8]])
    p = np.column_stack([np.interp(t, odom[:, 0], odom[:, i]) for i in (1, 2, 3)])
    return p, np.interp(t, odom[:, 0], yaw)


def _solve(d, w_lo, p, w_ins):
    """argmin_x sum w_lo (x[k+1] - x[k] - d[k])^2 + sum w_ins (x - p)^2 (tridiagonal system)."""
    from scipy.linalg import solveh_banded

    n = len(p)
    diag = w_ins.copy()
    diag[:-1] += w_lo
    diag[1:] += w_lo
    rhs = w_ins * p
    rhs[:-1] -= w_lo * d
    rhs[1:] += w_lo * d
    ab = np.zeros((2, n))
    ab[0, 1:] = -w_lo
    ab[1] = diag
    return solveh_banded(ab, rhs)


def _window_change(t, y, stamps, half):
    """y(s + half) - y(s - half) for every stamp s (linear interpolation, clamped at the ends)."""
    return np.column_stack([np.interp(stamps + half, t, y[:, i]) - np.interp(stamps - half, t, y[:, i])
                            for i in range(y.shape[1])])


def _runs(mask):
    """[(i0, i1)] index ranges (inclusive) where ``mask`` is True."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def lever_check(stamps, lo_cum, p_ins, yaw_ins, ok, min_turn=45.0):
    """Lever arm left between the INS rows and the odometry, body frame [m]: least squares of
    p_ins - lo_cum = c + v t + R(yaw) L over the scans in ``ok`` (an offset and a slow drift of the
    odometry, plus the INS point turning with the robot). ~0 once the INS rows refer to the body
    origin. None when those scans turn by less than ``min_turn`` deg (not observable)."""
    if ok.sum() < 20 or np.degrees(np.ptp(yaw_ins[ok])) < min_turn:
        return None
    t, c, s = stamps[ok] - stamps[0], np.cos(yaw_ins[ok]), np.sin(yaw_ins[ok])
    one, zero = np.ones_like(t), np.zeros_like(t)
    A = np.concatenate([np.column_stack([one, zero, t, zero, c, -s]), np.column_stack([zero, one, zero, t, s, c])])
    b = np.concatenate([p_ins[ok, 0] - lo_cum[ok, 0], p_ins[ok, 1] - lo_cum[ok, 1]])
    w = np.ones(len(b))
    for _ in range(5):                                            # robust: a few bad samples do not move it
        x = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        w = np.sqrt(np.minimum(1.0, 0.1 / np.maximum(np.abs(A @ x - b), 1e-9)))
    return [round(float(x[4]), 3), round(float(x[5]), 3)]


def ins_health(odom_ins, stamps, lo_cum, window=0.5, max_rate=0.4, margin=0.3, min_island=1.0):
    """True where the INS position can be trusted: over ``window``-long intervals it agrees with
    its own velocity integral (it does not while the robot brakes: 1 m/s apart in the moving sample
    bag, 0.1 m/s while walking). Bad stretches are widened by ``margin`` seconds, and a consistent
    stretch shorter than ``min_island`` seconds between two bad ones is bad too (during a braking
    event the INS can hold a wrong position consistently for a moment). The odometry is
    not used to judge the INS (a stalled registration would blame the INS); its disagreement is
    returned for the plots only."""
    t, p, v = odom_ins[:, 0], odom_ins[:, 1:3], odom_ins[:, 8:10]
    if not np.isfinite(v).all():
        v = np.gradient(p, t, axis=0)
    vint = np.concatenate([[[0.0, 0.0]], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t)[:, None], axis=0)])
    h = window / 2
    dp = _window_change(t, p, stamps, h)
    span = np.minimum(stamps + h, t[-1]) - np.maximum(stamps - h, t[0])
    rate_v = np.linalg.norm(dp - _window_change(t, vint, stamps, h), axis=1) / np.maximum(span, 1e-3)
    span_lo = np.minimum(stamps + h, stamps[-1]) - np.maximum(stamps - h, stamps[0])
    dp_s = _window_change(stamps, np.column_stack([np.interp(stamps, t, p[:, i]) for i in (0, 1)]), stamps, h)
    rate_lo = np.linalg.norm(dp_s - _window_change(stamps, lo_cum, stamps, h), axis=1) / np.maximum(span_lo, 1e-3)
    bad = rate_v > max_rate
    for k in np.nonzero(bad)[0]:
        bad |= np.abs(stamps - stamps[k]) <= margin
    for i0, i1 in _runs(~bad):
        if i0 > 0 and i1 < len(stamps) - 1 and stamps[i1] - stamps[i0] < min_island:
            bad[i0:i1 + 1] = True
    return ~bad, rate_v, rate_lo


def lo_health(d, vel_step, stamps, ins_ok=None, max_speed=3.5, window=0.6, zigzag=2.0, stall=0.3, moving=0.8,
              margin=0.2):
    """True per odometry step where the lidar odometry can be trusted. Bad: a step faster than
    ``max_speed`` (the robot walks at up to ~2.5 m/s); a zigzag (over ``window`` the odometry path
    is more than ``zigzag`` times its displacement: it jumps back and forth, as KISS-ICP does
    when a turn starts or stops); a stall (less than ``stall`` of the INS speed while a healthy
    INS, ``ins_ok`` per scan, says the robot walks faster than ``moving`` m/s: registration along a
    featureless wall; while braking the INS velocity overshoots and must not judge the odometry).
    Bad steps are widened by ``margin`` seconds."""
    n = len(d)
    if n == 0:
        return np.ones(0, bool)
    dt = np.maximum(np.diff(stamps), 1e-3)
    sp = np.linalg.norm(d, axis=1) / dt
    cum = np.concatenate([[[0.0, 0.0]], np.cumsum(d, axis=0)])
    plen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
    vlen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(vel_step, axis=1))])
    tm = 0.5 * (stamps[1:] + stamps[:-1])
    i0 = np.clip(np.searchsorted(stamps, tm - window / 2), 0, n)
    i1 = np.clip(np.searchsorted(stamps, tm + window / 2), 1, n)
    path = plen[i1] - plen[i0]
    disp = np.linalg.norm(cum[i1] - cum[i0], axis=1)
    ins_len = vlen[i1] - vlen[i0]
    span = np.maximum(stamps[i1] - stamps[i0], 1e-3)
    bad = (sp > max_speed) | ((path > zigzag * np.maximum(disp, 0.05)) & (path > 0.3))
    ins_good = np.ones(n, bool) if ins_ok is None else (ins_ok[1:] & ins_ok[:-1])
    bad |= ins_good & (ins_len / span > moving) & (path < stall * ins_len)
    for k in np.nonzero(bad)[0]:
        bad |= np.abs(tm - tm[k]) <= margin
    return ~bad


def heading_offset(yaw_lo, yaw_ins, ok, sigma_ins=2.0, sigma_drift=0.02, huber=1.0, iters=6):
    """Slowly varying offset s_k [rad] with yaw_lo + s the fused heading: least squares of
    s_k ~ yaw_ins,k - yaw_lo,k (weight 1 / sigma_ins^2 where ``ok``, Huber above ``huber`` deg) and
    s_k+1 ~ s_k (1 / sigma_drift^2), sigmas in deg: a time constant of sigma_ins / sigma_drift
    = 100 scans (10 s). The odometry heading is right from scan to scan (it registers the scans,
    with the gyro as its prior); the INS heading is off by 4-10 deg for 3-5 s after a fast turn
    (0921: 60-90 deg/s), while its position stays self-consistent, and only removes the slow drift
    of the odometry heading."""
    off = np.unwrap(yaw_ins - yaw_lo)
    n = len(off)
    if n == 0:
        return off
    base = np.where(ok, 1.0, 0.0) if ok.any() else np.ones(n)
    w, si, sd = base.copy(), np.radians(sigma_ins), np.radians(sigma_drift)
    for _ in range(iters):
        s = _solve(np.zeros(n - 1), np.full(n - 1, 1.0 / sd ** 2), off, np.maximum(w, 1e-6) / si ** 2)
        w = base * np.minimum(1.0, np.radians(huber) / np.maximum(np.abs(s - off), 1e-12))
    return s


def fuse(stamps, lo_poses, odom_ins, sigma_lo=0.02, sigma_ins=1.0, sigma_lo_bad=0.5, huber=0.3, gap_tol=0.5,
         gap_rel=0.05, iters=6):
    """Fused east / north at the scan stamps, plus diagnostics. ``odom_ins`` must already refer to
    the body origin (``apply_lever_arm``).

    Two sources, each with its own health. Where both are healthy, the odometry carries the
    motion from scan to scan and the INS only removes its slow drift (``sigma_ins`` = 1 m against
    0.02 m per odometry step: a time constant of ~5 s). On the 0921 data the INS is off by
    0.2-0.5 m at the 1 s scale even where it is self-consistent, while the odometry is not, so it
    must not pull harder. Where the INS is not healthy (braking: position and velocity disagree)
    its weight is zero. A run of unhealthy odometry steps (``lo_health``) is replaced by the INS
    velocity where the INS is healthy all along the run, else by the net odometry motion over the
    run spread evenly (a zigzag of KISS-ICP keeps its net motion; the INS velocity is what
    overshoots there), with ``sigma_lo_bad``. Where neither source is healthy the stretch is
    reported as bad (do not label it). A gap in the INS bridged by the odometry that misses the
    INS at its far end by more than ``gap_tol`` + ``gap_rel`` x its length is reported as bad too
    and its odometry steps get ``sigma_lo_bad``, so the miss is absorbed inside the gap instead of
    shifting the healthy stretches around it (the steps themselves are kept: on the 0921 data the
    odometry was right in every such gap, see docs/BAG_SPLIT.md 5.6). Gaps that touch the start or
    end cannot be checked and are listed as unverified."""
    p_ins, yaw_ins = ins_at(odom_ins, stamps)
    yaw_lo = np.unwrap([yaw_of(T[:3, :3]) for T in lo_poses])
    ins_ok0, _, _ = ins_health(odom_ins, stamps, np.zeros((len(stamps), 2)))
    yaw = yaw_lo + heading_offset(yaw_lo, yaw_ins, ins_ok0)      # fused heading
    step = np.diff(lo_poses[:, :2, 3], axis=0)
    a = (yaw - yaw_lo)[:-1]                                       # odometry frame -> ENU, per step
    c, s = np.cos(a), np.sin(a)
    d = np.column_stack([c * step[:, 0] - s * step[:, 1], s * step[:, 0] + c * step[:, 1]])
    ins_step = np.diff(p_ins[:, :2], axis=0)
    v = odom_ins[:, 8:10]
    if np.isfinite(v).all():
        v_s = np.column_stack([np.interp(stamps, odom_ins[:, 0], v[:, i]) for i in (0, 1)])
        vel_step = 0.5 * (v_s[1:] + v_s[:-1]) * np.diff(stamps)[:, None]
    else:
        vel_step = ins_step
    healthy, rate_v, rate_lo = ins_health(odom_ins, stamps, np.concatenate([[[0.0, 0.0]], np.cumsum(d, axis=0)]))
    lo_ok = lo_health(d, vel_step, stamps, healthy)
    bad_lo = ~lo_ok
    ins_ok_step = healthy[1:] & healthy[:-1]
    d_used, bridged = d.copy(), np.zeros(len(d), bool)
    for i0, i1 in _runs(bad_lo):
        if ins_ok_step[i0:i1 + 1].all():
            d_used[i0:i1 + 1] = vel_step[i0:i1 + 1]
        else:
            d_used[i0:i1 + 1] = d[i0:i1 + 1].sum(axis=0) / (i1 - i0 + 1)
            bridged[i0:i1 + 1] = True
    w_lo = np.where(bad_lo, 1.0 / sigma_lo_bad ** 2, 1.0 / sigma_lo ** 2)
    lo_cum = np.concatenate([[[0.0, 0.0]], np.cumsum(d_used, axis=0)])
    if not healthy.any():                                         # keep the problem anchored
        healthy[:] = True
    base = np.where(healthy, 1.0, 0.0)

    def solve(w_lo):
        w = base.copy()
        for _ in range(iters):
            xy = np.column_stack([_solve(d_used[:, i], w_lo, p_ins[:, i], np.maximum(w, 1e-6) / sigma_ins ** 2)
                                  for i in (0, 1)])
            r = np.linalg.norm(xy - p_ins[:, :2], axis=1)
            w = base * np.minimum(1.0, huber / np.maximum(r, 1e-9))
        return xy

    def gaps():
        """INS gaps: (t0, t1, closure or None if unverifiable, step range)."""
        out = []
        for i0, i1 in _runs(~healthy):
            if i0 == 0 or i1 == len(stamps) - 1:
                out.append((float(stamps[i0]), float(stamps[i1]), None, None))
                continue
            a_, b_ = i0 - 1, i1 + 1
            closure = float(np.linalg.norm((lo_cum[b_] - lo_cum[a_]) - (p_ins[b_, :2] - p_ins[a_, :2])))
            length = float(np.sum(np.linalg.norm(d_used[a_:b_], axis=1)))
            out.append((float(stamps[i0]), float(stamps[i1]), closure if closure > gap_tol + gap_rel * length else 0.0,
                        (a_, b_)))
        return out

    failed = [g for g in gaps() if g[2]]
    for _, _, _, (a_, b_) in failed:
        w_lo[a_:b_] = np.minimum(w_lo[a_:b_], 1.0 / sigma_lo_bad ** 2)
    xy = solve(w_lo)
    corr = np.linalg.norm(xy - p_ins[:, :2], axis=1)
    # odometry quality where it can be checked: 1 s displacement against the healthy INS
    k1 = max(1, int(round(1.0 / max(np.median(np.diff(stamps)), 1e-3)))) if len(stamps) > 1 else 1
    ok = [i for i in range(len(stamps) - k1) if healthy[i:i + k1 + 1].all()]
    lo_err = np.array([np.linalg.norm((lo_cum[i + k1] - lo_cum[i]) - (p_ins[i + k1, :2] - p_ins[i, :2])) for i in ok])
    lo_ok_scan = np.concatenate([[True], lo_ok]) & np.concatenate([lo_ok, [True]])
    bad_gaps = [[round(t0, 3), round(t1, 3), round(c, 3)] for t0, t1, c, _ in failed]
    unverified = [[round(t0, 3), round(t1, 3)] for t0, t1, c, _ in gaps() if c is None]
    for i0, i1 in _runs(~healthy & ~lo_ok_scan):                   # neither source
        t0, t1 = float(stamps[i0]), float(stamps[i1])
        if t1 - t0 >= 0.2 and not any(g[0] <= t0 and t1 <= g[1] for g in bad_gaps):
            bad_gaps.append([round(t0, 3), round(t1, 3), None])
    bad_gaps.sort(key=lambda g: g[0])
    in_bad = np.zeros(len(stamps), bool)
    for g in bad_gaps:
        in_bad |= (stamps >= g[0]) & (stamps <= g[1])
    vlen = float(np.sum(np.linalg.norm(vel_step, axis=1)))
    raw_cum = np.concatenate([[[0.0, 0.0]], np.cumsum(d, axis=0)])
    info = dict(scans=int(len(stamps)), lo_steps_rejected=int(bad_lo.sum()), lo_steps_bridged=int(bridged.sum()),
                lo_unhealthy_share=round(float(np.mean(bad_lo)), 3) if len(bad_lo) else 0.0,
                lo_path_ratio=round(float(np.sum(np.linalg.norm(d, axis=1)) / vlen), 2) if vlen > 0.5 else None,
                ins_unhealthy_share=round(float(np.mean(~healthy)), 3),
                correction_median_m=round(float(np.median(corr)), 3),
                correction_p90_m=round(float(np.percentile(corr, 90)), 3),
                correction_max_m=round(float(corr.max()), 3),
                heading_correction_p90_deg=round(float(np.percentile(np.degrees(np.abs(yaw - yaw_ins)), 90)), 2),
                heading_correction_max_deg=round(float(np.degrees(np.abs(yaw - yaw_ins)).max()), 2),
                lo_vs_ins_1s_median_m=round(float(np.median(lo_err)), 3) if len(lo_err) else None,
                lo_vs_ins_1s_p90_m=round(float(np.percentile(lo_err, 90)), 3) if len(lo_err) else None,
                bad_intervals=bad_gaps, unverified_intervals=unverified, bad_share=round(float(in_bad.mean()), 3),
                lo_path_m=round(float(np.sum(np.linalg.norm(d_used, axis=1))), 2),
                ins_path_m=round(float(np.sum(np.linalg.norm(ins_step, axis=1))), 2),
                lever_residual_m=lever_check(stamps, raw_cum, p_ins, yaw_ins, healthy & lo_ok_scan))
    return xy, corr, info, dict(healthy=healthy, rate_v=rate_v, rate_lo=rate_lo, bad_lo=bad_lo, bridged=bridged,
                                p_ins=p_ins, yaw_ins=yaw_ins, yaw=yaw)


def pose_checks(stamps, lo_poses, xy, diag, kept, gap=1.0, seed=0):
    """Position error between scans ``gap`` seconds apart as the lidar sees it (the preview's
    shift check), for the INS, the lidar odometry alone and the fused poses: tells whether a
    remaining error comes from the odometry or from the fusion."""
    from sequence_preview import pose_shift

    p_ins, yaw_ins, yaw = diag["p_ins"], diag["yaw_ins"], diag["yaw"]
    pose = dict(ins=lambda k: make_T(rot_z(yaw_ins[k]), p_ins[k]), lo=lambda k: lo_poses[k],
                fused=lambda k: make_T(rot_z(yaw[k]), [xy[k, 0], xy[k, 1], p_ins[k, 2]]))
    step = max(1, int(round(gap / max(np.median(np.diff(stamps)), 1e-3)))) if len(stamps) > 1 else 1
    pairs = [(i, i + step) for i in sorted(kept) if i + step in kept]
    out = dict(pair_t=np.array([stamps[i] - stamps[0] for i, _ in pairs]))
    for name, T in pose.items():
        rng = np.random.default_rng(seed)
        w = {k: transform_points(T(k), kept[k].astype(np.float64))[:, :2] for k in kept}
        out[name] = np.array([pose_shift(w[i], w[j], rng) for i, j in pairs])
    return out


def refined_odom(odom_ins, stamps, xy, yaw=None):
    """INS rows with east / north replaced by the fused trajectory (and its velocity), and with
    ``yaw`` (fused heading at the stamps) the heading too; outside the scans the INS rows plus the
    correction at the first / last scan."""
    out = odom_ins.copy()
    t = odom_ins[:, 0]
    p_ins, yaw_ins = ins_at(odom_ins, stamps)
    off = xy - p_ins[:, :2]                                       # correction at the scan stamps
    inside = (t >= stamps[0]) & (t <= stamps[-1])
    if yaw is not None:
        dy = yaw - yaw_ins                                        # heading correction at the stamps
        y_new = np.unwrap([yaw_of(quat_to_R(q)) for q in odom_ins[:, 4:8]]) + np.where(t < stamps[0], dy[0], dy[-1])
        y_new[inside] = np.interp(t[inside], stamps, yaw)
        out[:, 4:8] = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.sin(y_new / 2), np.cos(y_new / 2)])
    for i in (0, 1):
        out[inside, 1 + i] = np.interp(t[inside], stamps, xy[:, i])
        out[~inside, 1 + i] = odom_ins[~inside, 1 + i] + np.where(t[~inside] < stamps[0], off[0, i], off[-1, i])
        v = np.gradient(xy[:, i], stamps) if len(stamps) > 1 else np.zeros(1)
        out[inside, 8 + i] = np.interp(t[inside], stamps, v)
    return out


def draw(path, name, odom_ins, stamps, xy, corr, info, diag, checks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3 if checks else 2, figsize=(21 if checks else 14, 5.5))
    a1, a2 = axs[0], axs[1]
    a1.plot(odom_ins[:, 1], odom_ins[:, 2], c="0.6", lw=1, label="INS")
    a1.plot(xy[:, 0], xy[:, 1], c="tab:blue", lw=1.2, label="lidar odometry + RTK")
    a1.set_aspect("equal")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a1.set_title(name)
    t0 = stamps[0]
    a2.plot(stamps - t0, corr, c="tab:red", lw=1, label="|fused - INS| [m]")
    sp = np.hypot(odom_ins[:, 8], odom_ins[:, 9])
    a2.plot(odom_ins[:, 0] - t0, sp, c="0.6", lw=0.8, label="INS speed [m/s]")
    bad = ~diag["healthy"]
    if bad.any():
        a2.fill_between(stamps - t0, 0, 1, where=bad, transform=a2.get_xaxis_transform(), color="tab:orange",
                        alpha=0.2, label="INS not trusted")
    a2.set_xlabel("time [s]")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    a2.set_title("correction: median %.2f m, p90 %.2f m, max %.2f m" % (
        info["correction_median_m"], info["correction_p90_m"], info["correction_max_m"]))
    bl = np.nonzero(diag["bad_lo"])[0]
    if len(bl):
        a2.plot(stamps[bl + 1] - t0, np.zeros(len(bl)), "kx", ms=5, label="odometry step rejected")
        a2.legend(fontsize=8)
    if checks:
        a3 = axs[2]
        for k, c in (("ins", "0.5"), ("lo", "tab:green"), ("fused", "tab:blue")):
            a3.plot(checks["pair_t"], checks[k], "o-", ms=3, lw=0.8, c=c,
                    label="%s (p90 %.2f m)" % (k, np.nanpercentile(checks[k], 90) if len(checks[k]) else np.nan))
        a3.set_xlabel("time [s]")
        a3.set_ylabel("lidar shift between scans 1 s apart [m]")
        a3.grid(alpha=0.3)
        a3.legend(fontsize=8)
        a3.set_title("pose check per source")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def refine(seq_dir, plots=None, prior="imu", lever_arm=None, **kw):
    """``lever_arm``: INS position point in the body frame (default: what the converter applied,
    ``meta["ins_lever_arm"]``; INSPVAX sequences converted before it have none and get the value
    measured on 0921); whatever the converter applied already is taken into account."""
    rd = SequenceReader(seq_dir)
    p_ins = os.path.join(seq_dir, ODOM_INS)
    odom_ins = np.load(p_ins) if os.path.exists(p_ins) else rd.odom
    if len(rd) < 2 or len(odom_ins) < 2:
        return None
    applied = np.asarray(rd.meta.get("ins_lever_arm", [0.0, 0.0, 0.0]), np.float64)
    if lever_arm is None:                         # what the converter chose; older INSPVAX sequences: 0921 value
        lever_arm = applied if "ins_lever_arm" in rd.meta or "lat0" not in rd.meta else KYGI_INS_LEVER_ARM
    lever_arm = np.asarray(lever_arm, np.float64)
    odom_body = apply_lever_arm(odom_ins, lever_arm - applied)     # INS rows of the body origin
    stamps, poses, kept = lidar_odometry(rd, prior=None if prior == "none" else prior)
    xy, corr, info, diag = fuse(stamps, poses, odom_body, **kw)
    info["lever_arm"] = [round(float(a), 3) for a in lever_arm]
    checks = pose_checks(stamps, poses, xy, diag, kept) if kept else None
    if checks:
        for k in ("ins", "lo", "fused"):
            v = checks[k][np.isfinite(checks[k])]
            info["shift_%s_p90_m" % k] = round(float(np.percentile(v, 90)), 3) if len(v) else None
            info["shift_%s_median_m" % k] = round(float(np.median(v)), 3) if len(v) else None
    out = refined_odom(odom_body, stamps, xy, diag["yaw"])
    if not os.path.exists(p_ins):
        np.save(p_ins, odom_ins)
    np.save(os.path.join(seq_dir, "odom.npy"), out)
    meta = dict(rd.meta, pose_source="kiss-icp+ins", pose_refine=dict(info, prior=prior, **kw))
    with open(os.path.join(seq_dir, "meta.json") + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(os.path.join(seq_dir, "meta.json") + ".tmp", os.path.join(seq_dir, "meta.json"))
    name = os.path.basename(os.path.normpath(seq_dir))
    if plots:
        os.makedirs(plots, exist_ok=True)
        draw(os.path.join(plots, name + ".png"), name, odom_body, stamps, xy, corr, info, diag, checks)
        # everything needed to look at one sequence again without its point clouds (a few tens of kB)
        np.savez_compressed(os.path.join(plots, name + "_diag.npz"), stamps=stamps, xy=xy,
                            lo_xyz=poses[:, :3, 3], lo_yaw=[yaw_of(T[:3, :3]) for T in poses],
                            p_ins=diag["p_ins"], yaw_ins=diag["yaw_ins"], yaw=diag["yaw"], healthy=diag["healthy"],
                            rate_v=diag["rate_v"], bad_lo=diag["bad_lo"], bridged=diag["bridged"],
                            odom_ins=odom_body,                       # float64: unix times
                            **({"check_" + k: v for k, v in checks.items()} if checks else {}))
    return info


def refine_one(job):
    """(sequence, info or None, seconds): one sequence for ``run_each``."""
    import time
    seq, plots, prior, lever_arm, kw = job
    t0 = time.time()
    return seq, refine(seq, plots, prior=prior, lever_arm=lever_arm, **kw), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=0, help="sequences refined at once (0: one per core, 1: no subprocess)")
    ap.add_argument("--plots", default=None,
                    help="directory for one plot and one <seq>_diag.npz per sequence, and refine_summary.csv")
    ap.add_argument("--sigma-lo", type=float, default=0.02, help="[m] trust in one odometry step")
    ap.add_argument("--sigma-ins", type=float, default=1.0,
                    help="[m] trust in one healthy INS position (1 m: it only removes the odometry drift)")
    ap.add_argument("--sigma-lo-bad", type=float, default=0.5, help="[m] trust in an unhealthy odometry step")
    ap.add_argument("--prior", choices=["imu", "none"], default="imu",
                    help="KISS-ICP motion prediction: gyro rotation (imu) or its own constant velocity (none)")
    ap.add_argument("--huber", type=float, default=0.3, help="[m] INS residual above which it is down-weighted")
    ap.add_argument("--gap-tol", type=float, default=0.5,
                    help="[m] a bridged INS gap may miss the INS at its far end by this (+ --gap-rel x its length) "
                         "before it counts as bad")
    ap.add_argument("--gap-rel", type=float, default=0.05, help="see --gap-tol")
    ap.add_argument("--lever-arm", type=float, nargs=3, default=None,
                    help="[m] INS position point in the body frame (default: meta ins_lever_arm from the "
                         "converter, else the value measured on 0921)")
    args = ap.parse_args()
    kw = dict(sigma_lo=args.sigma_lo, sigma_ins=args.sigma_ins, sigma_lo_bad=args.sigma_lo_bad, huber=args.huber,
              gap_tol=args.gap_tol, gap_rel=args.gap_rel)
    print("%-26s %5s  %-28s %8s %7s %8s %9s  %-24s" % (
        "sequence", "scans", "correction med / p90 / max [m]", "INS bad", "LO bad", "pose bad", "LO/INS path",
        "shift p90 ins / lo / fused"))
    rows = []
    seqs = [s for s in sorted(args.seq) if os.path.exists(os.path.join(s, "meta.json"))]
    jobs = [(s, args.plots, args.prior, args.lever_arm, kw) for s in seqs]
    for seq, info, _ in run_each(refine_one, jobs, args.jobs):
        if info is None:
            print("%-26s  too short - skipped" % os.path.basename(os.path.normpath(seq)))
            continue
        name = os.path.basename(os.path.normpath(seq))
        f = lambda k: "-" if info.get(k) is None else "%.2f" % info[k]   # noqa: E731
        print("%-26s %5d  %6.3f / %6.3f / %6.3f        %6.1f%% %6.1f%% %7.1f%% %9s  %-24s" % (
            name, info["scans"], info["correction_median_m"], info["correction_p90_m"], info["correction_max_m"],
            100 * info["ins_unhealthy_share"], 100 * info["lo_unhealthy_share"], 100 * info["bad_share"],
            f("lo_path_ratio"), " / ".join(f("shift_%s_p90_m" % k) for k in ("ins", "lo", "fused"))), flush=True)
        for g in info["bad_intervals"]:
            why = "neither source healthy" if g[2] is None else \
                "odometry misses the INS by %.2f m across the gap (absorbed inside it)" % g[2]
            print("    pose bad %.3f .. %.3f: %s" % (g[0], g[1], why))
        rows.append(dict(name=name, **{k: v for k, v in info.items() if not isinstance(v, list) and "lever" not in k},
                         lever_arm=json.dumps(info["lever_arm"]), lever_residual_m=json.dumps(info["lever_residual_m"]),
                         bad_intervals=json.dumps(info["bad_intervals"]),
                         unverified_intervals=json.dumps(info["unverified_intervals"])))
    fits = np.array([json.loads(r["lever_residual_m"]) for r in rows if r["lever_residual_m"] != "null"])
    if len(fits):
        print("INS lever arm left over, median of %d sequences turning > 45 deg: (%.2f, %.2f) m (should be ~0; "
              "else add it to --lever-arm)" % (len(fits), *np.median(fits, axis=0)))
    if args.plots and rows:
        import csv
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(args.plots, "refine_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
