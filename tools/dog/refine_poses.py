"""Refine the localization of converted sequences with lidar odometry (KISS-ICP) anchored to RTK.

    python tools/dog/refine_poses.py --seq data/seq/0921_* [--plots data/preview/poses]

Why: the kygi INS position is right while the robot walks steadily, but when it brakes hard the
INS velocity overshoots (2.5 m/s) and the position runs ~0.5 m ahead for a second or more; the
backward steps that follow only partly correct it (moving sample bag: 0.44 m ahead at 2.1 s,
measured by registering lidar frames). Sweeps stacked across that time show ghosts, and
auto-labels that pin a static object in the world frame land ~0.5 m off the points there.

How, per sequence:

1. Lidar odometry: every scan (same de-duplication and self filter as the online preprocessing,
   body frame, per-point times) goes through KISS-ICP, which deskews it to the scan end (our
   stamp) and registers it against its local map: the motion from scan to scan, without a global
   frame and with some drift (along a featureless wall it can stall).
2. INS health: over 0.5 s windows the INS position must agree with its own velocity integral
   (1 m/s apart while braking in the sample bag, 0.1 m/s while walking); bad stretches are widened
   by 0.3 s, and short consistent stretches between bad ones count as bad.
3. Fusion, separately for east and north: positions x_k at the scan stamps minimise

       sum_k v_k (x_k+1 - x_k - d_k)^2 / sigma_lo^2  +  sum_k w_k (x_k - p_ins,k)^2 / sigma_ins^2

   with d_k the odometry step turned into ENU with the INS heading (which agrees with the gyro).
   w_k is 0 where the INS is not healthy, so the odometry bridges those gaps between the healthy
   INS on both sides; elsewhere the INS holds the trajectory and the odometry adds the motion
   between scans. Both weights are re-estimated (Huber) so a stalled or jumping registration, or a
   stray INS position, counts less.
4. Self-check: a bridged gap whose odometry misses the INS at its far end by more than
   ``--gap-tol`` (+5 % of its length) is listed as bad in meta.json (neither source is right
   there; do not label those frames). Gaps touching the start or end cannot be checked and are
   listed as unverified.
5. Output: odom.npy keeps the INS rows (100 Hz times, heading-only attitude, height), with east /
   north replaced by the fused trajectory (linear between scan stamps, so the IMU propagation and
   deskew work exactly as before) and the velocity by its derivative. The INS original is kept
   once as odom_ins.npy and is always the input, so re-running is safe. meta.json gets
   ``pose_source = "kiss-icp+ins"`` and ``pose_refine`` (correction statistics, bad and
   unverified intervals, odometry error against the healthy INS over 1 s).

Check the result with tools/dog/sequence_preview.py: its ``shift_p90_m`` (the lidar-measured
position error between frames 1 s apart) must drop; on the moving sample bag 0.51 -> 0.12 m.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.geometry import quat_to_R, transform_points, yaw_of  # noqa: E402
from dog_perception.io import SequenceReader, load_scan  # noqa: E402
from dog_perception.preprocess import SelfFilter, SelfFilterConfig, dedup_returns  # noqa: E402

ODOM_INS = "odom_ins.npy"


def lidar_odometry(rd, voxel=0.4, max_range=60.0, min_range=1.5):
    """(stamps (K,), poses (K, 4, 4)): KISS-ICP body poses at the scan ends, in the frame of the
    first scan."""
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = max_range, min_range, True
    cfg.mapping.voxel_size = voxel
    odo = KissICP(cfg)
    sf = SelfFilter(SelfFilterConfig(), rd.T_body_lidar)
    stamps, poses = [], []
    for f in rd.scan_files:
        scan = dedup_returns(load_scan(f).shifted(rd.lidar_time_offset))
        keep = np.isfinite(scan.xyz).all(axis=1)
        keep[keep] = sf(scan.xyz[keep])
        xyz = transform_points(rd.T_body_lidar, scan.xyz[keep]).astype(np.float64)
        odo.register_frame(xyz, np.asarray(scan.point_times, np.float64)[keep])
        stamps.append(scan.stamp)
        poses.append(odo.last_pose.copy())
    return np.array(stamps), np.array(poses)


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


def fuse(stamps, lo_poses, odom_ins, sigma_lo=0.02, sigma_ins=0.05, huber=0.2, huber_lo=0.03, lo_gate=0.3,
         gap_tol=0.25, iters=6):
    """Fused east / north at the scan stamps, plus diagnostics.

    Where the INS is healthy it holds the trajectory (it is within a few cm there, while the
    odometry can drift along walls); the odometry adds the motion between scans. Where it is not
    (hard braking), its weight is zero and the odometry bridges the gap between the healthy INS on
    both sides. A bridged gap whose odometry does not close on the INS at its far end within
    ``gap_tol`` (plus 5 % of its length) is reported as bad: neither source can be trusted there,
    so its frames should not be labelled. A gap that touches the start or end of the sequence
    cannot be checked this way and is reported as unverified."""
    p_ins, yaw_ins = ins_at(odom_ins, stamps)
    yaw_lo = np.array([yaw_of(T[:3, :3]) for T in lo_poses])
    step = np.diff(lo_poses[:, :2, 3], axis=0)
    a = (yaw_ins - yaw_lo)[:-1]                                   # odometry frame -> ENU, per step
    c, s = np.cos(a), np.sin(a)
    d = np.column_stack([c * step[:, 0] - s * step[:, 1], s * step[:, 0] + c * step[:, 1]])
    # a failed registration: the odometry step disagrees with the INS step by more than lo_gate
    ins_step = np.diff(p_ins[:, :2], axis=0)
    bad_lo = np.linalg.norm(d - ins_step, axis=1) > lo_gate
    d_used = np.where(bad_lo[:, None], ins_step, d)
    w_lo = np.where(bad_lo, 1.0, 1.0 / sigma_lo ** 2)
    lo_cum = np.concatenate([[[0.0, 0.0]], np.cumsum(d_used, axis=0)])
    healthy, rate_v, rate_lo = ins_health(odom_ins, stamps, lo_cum)
    if not healthy.any():                                         # keep the problem anchored
        healthy[:] = True
    base = np.where(healthy, 1.0, 0.0)
    w, wl = base.copy(), w_lo.copy()
    for _ in range(iters):
        xy = np.column_stack([_solve(d_used[:, i], wl, p_ins[:, i], np.maximum(w, 1e-6) / sigma_ins ** 2)
                              for i in (0, 1)])
        r = np.linalg.norm(xy - p_ins[:, :2], axis=1)
        w = base * np.minimum(1.0, huber / np.maximum(r, 1e-9))
        # an odometry step far from the fused step (a stalled or jumping registration) counts less
        rl = np.linalg.norm(np.diff(xy, axis=0) - d_used, axis=1)
        wl = w_lo * np.minimum(1.0, huber_lo / np.maximum(rl, 1e-9))
    corr = np.linalg.norm(xy - p_ins[:, :2], axis=1)
    # odometry quality where it can be checked: 1 s displacement against the healthy INS
    k1 = max(1, int(round(1.0 / max(np.median(np.diff(stamps)), 1e-3)))) if len(stamps) > 1 else 1
    ok = [i for i in range(len(stamps) - k1) if healthy[i:i + k1 + 1].all()]
    lo_err = np.array([np.linalg.norm((lo_cum[i + k1] - lo_cum[i]) - (p_ins[i + k1, :2] - p_ins[i, :2])) for i in ok])
    bad_gaps, unverified = [], []
    for i0, i1 in _runs(~healthy):
        t0, t1 = float(stamps[i0]), float(stamps[i1])
        if i0 == 0 or i1 == len(stamps) - 1:
            unverified.append([round(t0, 3), round(t1, 3)])
            continue
        a_, b_ = i0 - 1, i1 + 1
        closure = float(np.linalg.norm((lo_cum[b_] - lo_cum[a_]) - (p_ins[b_, :2] - p_ins[a_, :2])))
        length = float(np.sum(np.linalg.norm(d_used[a_:b_], axis=1)))
        if closure > gap_tol + 0.05 * length:
            bad_gaps.append([round(t0, 3), round(t1, 3), round(closure, 3)])
    info = dict(scans=int(len(stamps)), lo_steps_rejected=int(bad_lo.sum()),
                ins_unhealthy_share=round(float(np.mean(~healthy)), 3),
                correction_median_m=round(float(np.median(corr)), 3),
                correction_p90_m=round(float(np.percentile(corr, 90)), 3),
                correction_max_m=round(float(corr.max()), 3),
                lo_vs_ins_1s_median_m=round(float(np.median(lo_err)), 3) if len(lo_err) else None,
                lo_vs_ins_1s_p90_m=round(float(np.percentile(lo_err, 90)), 3) if len(lo_err) else None,
                bad_intervals=bad_gaps, unverified_intervals=unverified,
                bad_share=round(float(sum(np.sum((stamps >= g[0]) & (stamps <= g[1])) for g in bad_gaps)
                                      / len(stamps)), 3),
                lo_path_m=round(float(np.sum(np.linalg.norm(d_used, axis=1))), 2),
                ins_path_m=round(float(np.sum(np.linalg.norm(ins_step, axis=1))), 2))
    return xy, corr, info, dict(healthy=healthy, rate_v=rate_v, rate_lo=rate_lo)


def refined_odom(odom_ins, stamps, xy):
    """INS rows with east / north replaced by the fused trajectory (and its velocity)."""
    out = odom_ins.copy()
    t = odom_ins[:, 0]
    p_ins, _ = ins_at(odom_ins, stamps)
    off = xy - p_ins[:, :2]                                       # correction at the scan stamps
    inside = (t >= stamps[0]) & (t <= stamps[-1])
    for i in (0, 1):
        out[inside, 1 + i] = np.interp(t[inside], stamps, xy[:, i])
        out[~inside, 1 + i] = odom_ins[~inside, 1 + i] + np.where(t[~inside] < stamps[0], off[0, i], off[-1, i])
        v = np.gradient(xy[:, i], stamps) if len(stamps) > 1 else np.zeros(1)
        out[inside, 8 + i] = np.interp(t[inside], stamps, v)
    return out


def draw(path, name, odom_ins, stamps, xy, corr, info, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
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
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def refine(seq_dir, plots=None, **kw):
    rd = SequenceReader(seq_dir)
    p_ins = os.path.join(seq_dir, ODOM_INS)
    odom_ins = np.load(p_ins) if os.path.exists(p_ins) else rd.odom
    if len(rd) < 2 or len(odom_ins) < 2:
        return None
    stamps, poses = lidar_odometry(rd)
    xy, corr, info, diag = fuse(stamps, poses, odom_ins, **kw)
    out = refined_odom(odom_ins, stamps, xy)
    if not os.path.exists(p_ins):
        np.save(p_ins, odom_ins)
    np.save(os.path.join(seq_dir, "odom.npy"), out)
    meta = dict(rd.meta, pose_source="kiss-icp+ins", pose_refine=dict(info, **{k: v for k, v in kw.items()}))
    with open(os.path.join(seq_dir, "meta.json") + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(os.path.join(seq_dir, "meta.json") + ".tmp", os.path.join(seq_dir, "meta.json"))
    name = os.path.basename(os.path.normpath(seq_dir))
    if plots:
        os.makedirs(plots, exist_ok=True)
        draw(os.path.join(plots, name + ".png"), name, odom_ins, stamps, xy, corr, info, diag)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--plots", default=None, help="directory for one trajectory / correction plot per sequence")
    ap.add_argument("--sigma-lo", type=float, default=0.02, help="[m] trust in one odometry step")
    ap.add_argument("--sigma-ins", type=float, default=0.05, help="[m] trust in one healthy INS position")
    ap.add_argument("--huber", type=float, default=0.2, help="[m] INS residual above which it is down-weighted")
    ap.add_argument("--lo-gate", type=float, default=0.3, help="[m] odometry step this far from the INS step is rejected")
    ap.add_argument("--gap-tol", type=float, default=0.25,
                    help="[m] (+5%% of its length) a bridged gap may miss the INS at its far end before it counts as bad")
    args = ap.parse_args()
    kw = dict(sigma_lo=args.sigma_lo, sigma_ins=args.sigma_ins, huber=args.huber, lo_gate=args.lo_gate,
              gap_tol=args.gap_tol)
    print("%-26s %5s  %-28s %9s %9s %10s  %s" % ("sequence", "scans", "correction med / p90 / max [m]", "INS bad",
                                                  "pose bad", "LO 1s p90", "path lidar / INS [m]"))
    for seq in sorted(args.seq):
        if not os.path.exists(os.path.join(seq, "meta.json")):
            continue
        info = refine(seq, args.plots, **kw)
        if info is None:
            print("%-26s  too short - skipped" % os.path.basename(os.path.normpath(seq)))
            continue
        lo = info["lo_vs_ins_1s_p90_m"]
        print("%-26s %5d  %6.3f / %6.3f / %6.3f        %6.1f%%   %6.1f%% %10s  %7.1f / %7.1f" % (
            os.path.basename(os.path.normpath(seq)), info["scans"], info["correction_median_m"],
            info["correction_p90_m"], info["correction_max_m"], 100 * info["ins_unhealthy_share"],
            100 * info["bad_share"], "-" if lo is None else "%.3f" % lo, info["lo_path_m"], info["ins_path_m"]),
            flush=True)
        for g in info["bad_intervals"]:
            print("    pose bad %.3f .. %.3f: odometry misses the INS by %.2f m across the gap" % tuple(g))


if __name__ == "__main__":
    main()
