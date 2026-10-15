"""Visual and numeric checks of converted sequences: one PNG per sequence, plus checks.csv and
index.html over all of them.

    python tools/dog/sequence_preview.py --seq data/seq/0921_165659_s* --out data/preview/0921_165659 [--every 5]

Every ``--every``-th scan goes through the same preprocessing as online (deskew, self filter,
gravity / ground levelling, sweep stacking) and is checked:

* map (left panel): the current sweep of every processed frame moved into the world frame and
  drawn on one bird's-eye view, colour = time, with the trajectory. Walls, kerbs and poles must
  be single sharp lines; doubled or smeared walls mean bad poses, a wrong lidar extrinsic or a
  time offset between lidar and INS.
* alignment: for frames about 0.5 s apart, the median horizontal distance from the points on
  vertical structures (walls, poles, parked cars) of one to the nearest such point of the other,
  both in the world frame (2-D, so different beams hitting the same wall still match). On the
  moving sample bag (1.5-2.5 m/s) it is 2 cm at 0.1 s and 4 cm at 0.5 s; it grows with the gap
  only because the view changes. Above 0.1 m look at the map, above 0.2 m the frames do not
  line up. A constant lidar-INS time offset while driving straight shifts all frames alike and
  is not caught here; turns and speed changes show it.
* ground: in the detection frame the ground has to be at z = 0 and level: height of the ground
  (10th percentile of z per 2 m cell, 3-12 m away) and the tilt of a plane fitted to it (sloped
  terrain counts too, so a few degrees can be the place, not the levelling).
* model input (middle panels): the middle frame exactly as the detector gets it (5 sweeps,
  detection frame) from above and from the side, with the +-40.8 m detection range.
* INS position steps: jumps of the localization position that its own velocity does not explain
  (|dp - v dt| per sample). The kygi INS pulls its position back every ~0.2 s while the robot slows
  down (up to 0.33 m in the moving sample bag); sweeps on either side of such a step do not line up,
  which shows as alignment spikes at speed changes. Marked on the speed / alignment plot.
* rates and gaps: lidar / IMU / localization rates and largest gaps, RTK pos_type, speed.

Status per check: OK, WARN or FAIL (thresholds in ``LIMITS``); the sequence takes the worst.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.io import SequenceReader  # noqa: E402
from dog_perception.pipeline import PerceptionPipeline  # noqa: E402
from dog_perception.preprocess import DetFrameConfig, PreprocessConfig  # noqa: E402

# (ok_max, warn_max) per check; above warn_max is FAIL
LIMITS = dict(align_m=(0.10, 0.20), ground_z_m=(0.10, 0.20), tilt_deg=(2.0, 4.0), lidar_gap_s=(0.2, 0.5),
              imu_gap_s=(0.05, 0.2), odom_gap_s=(0.1, 0.5), ins_step_m=(0.15, 0.5))
XY_RANGE = 40.8


def grade(name, value):
    if value is None or not np.isfinite(value):
        return "-"                      # not measured (e.g. a sequence shorter than the frame gap)
    ok, warn = LIMITS[name]
    return "OK" if abs(value) <= ok else ("WARN" if abs(value) <= warn else "FAIL")


def ground_stats(pts):
    """Current-sweep points (det frame) -> (median ground z, plane tilt [deg]), 3-12 m around the
    robot (where levelling matters for detection; farther out the terrain itself slopes)."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    p = pts[(r > 3) & (r < 12)]
    if len(p) < 100:
        return np.nan, np.nan
    cell = np.floor(p[:, :2] / 2.0).astype(np.int64)
    key = cell[:, 0] * 100003 + cell[:, 1]
    order = np.argsort(key)
    key, p = key[order], p[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    ends = np.r_[starts[1:], len(key)]
    g = np.array([[*p[a:b, :2].mean(0), np.percentile(p[a:b, 2], 10)] for a, b in zip(starts, ends) if b - a >= 5])
    if len(g) < 5:
        return np.nan, np.nan
    A = np.column_stack([g[:, 0], g[:, 1], np.ones(len(g))])
    coef, *_ = np.linalg.lstsq(A, g[:, 2], rcond=None)
    return float(np.median(g[:, 2])), float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))


def ins_steps(od, t0=-np.inf, t1=np.inf):
    """(times, steps [m]): horizontal position change minus velocity x dt, per localization sample
    in [t0, t1]. Empty when the odometry has no velocity (LIO without twist)."""
    if len(od) < 2 or not np.isfinite(od[:, 8:10]).all():
        return np.zeros(0), np.zeros(0)
    dt = np.diff(od[:, 0])
    step = np.hypot(*(np.diff(od[:, 1:3], axis=0) - 0.5 * (od[1:, 8:10] + od[:-1, 8:10]) * dt[:, None]).T)
    t = od[1:, 0]
    m = (t >= t0) & (t <= t1)
    return t[m], step[m]


def structure_xy(frame, cell=0.4, min_extent=0.6):
    """Current-sweep points 3-30 m away that belong to vertical structures (walls, poles, parked
    cars: 0.4 m cells spanning > 0.6 m in height), world frame, (M, 2). Ground rings on uneven
    terrain move with the robot and would look like misalignment, so they are left out."""
    p = frame.points[frame.points[:, 4] <= 1e-6]
    r = np.hypot(p[:, 0], p[:, 1])
    p = p[(p[:, 2] > -0.5) & (p[:, 2] < 3.5) & (r > 3) & (r < 30)]
    if len(p) == 0:
        return np.zeros((0, 2))
    c = np.floor(p[:, :2] / cell).astype(np.int64)
    key = c[:, 0] * 1000003 + c[:, 1]
    order = np.argsort(key, kind="stable")
    key, p = key[order], p[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    zmax, zmin = np.maximum.reduceat(p[:, 2], starts), np.minimum.reduceat(p[:, 2], starts)
    keep_cell = (zmax - zmin) > min_extent
    keep = np.repeat(keep_cell, np.diff(np.r_[starts, len(key)]))
    p = p[keep & (p[:, 2] > 0.2)]
    w = p[:, :3] @ frame.T_world_det[:3, :3].T + frame.T_world_det[:3, 3]
    return w[:, :2]


def align_residual(a, b, rng):
    from scipy.spatial import cKDTree
    if len(a) < 50 or len(b) < 50:
        return np.nan
    q = a[rng.choice(len(a), min(len(a), 4000), replace=False)]
    d, _ = cKDTree(b).query(q)
    return float(np.median(d))


def pipeline_for(rd):
    """Preprocessing exactly as auto-labelling and the online node run it on this sequence."""
    pcfg = PreprocessConfig(T_body_lidar=rd.T_body_lidar,
                            det_frame=DetFrameConfig(base_height=rd.meta.get("base_height", 0.45),
                                                     **rd.meta.get("det_frame", {})))
    return PerceptionPipeline(pcfg, detector=None)


def check_sequence(seq_dir, every, rng):
    rd = SequenceReader(seq_dir)
    meta = rd.meta
    pipe = pipeline_for(rd)
    n = len(rd)
    mid = n // 2
    wanted = set(range(0, n, every)) | set(range(max(0, mid - 4), mid + 1))   # mid: 5 consecutive sweeps
    frames, mid_frame, stamps, k = [], None, [], 0
    for ev in rd.events():
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            stamps.append(ev[1].stamp)
            if k in wanted:
                fr = pipe.pre.process(ev[1])
                if fr is not None:
                    if k % every == 0:
                        frames.append(fr)
                    if k == mid:
                        mid_frame = fr
            k += 1
    stamps = np.array(stamps)

    gz, tilt = zip(*[ground_stats(f.points[f.points[:, 4] <= 1e-6]) for f in frames]) if frames else ([], [])
    xy = [structure_xy(f) for f in frames]
    t = np.array([f.stamp for f in frames])
    res, res_t = [], []
    for i in range(len(frames)):
        j = np.searchsorted(t, t[i] + 0.45)
        if j < len(frames) and t[j] - t[i] < 1.0:
            res.append(align_residual(xy[i], xy[j], rng))
            res_t.append(t[i])
    gap = lambda x: float(np.max(np.diff(x))) if len(x) > 1 else np.nan   # noqa: E731
    rate = lambda x: float((len(x) - 1) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else np.nan   # noqa: E731
    od = rd.odom
    path = float(np.sum(np.hypot(*np.diff(od[:, 1:3], axis=0).T))) if len(od) > 1 else 0.0
    speed = np.hypot(od[:, 8], od[:, 9]) if len(od) and np.isfinite(od[:, 8]).any() else np.zeros(len(od))
    step_t, steps = ins_steps(od, stamps[0], stamps[-1]) if n else (np.zeros(0), np.zeros(0))
    c = dict(name=os.path.basename(os.path.normpath(seq_dir)), split=meta.get("segment", {}).get("split", ""),
             scans=n, duration_s=float(stamps[-1] - stamps[0]) if n > 1 else 0.0, lidar_hz=rate(stamps),
             lidar_gap_s=gap(stamps), imu_hz=rate(rd.imu[:, 0]), imu_gap_s=gap(rd.imu[:, 0]),
             odom_hz=rate(od[:, 0]), odom_gap_s=gap(od[:, 0]), path_m=path,
             mean_speed=float(np.mean(speed)) if len(speed) else 0.0,
             pos_type=" ".join("%s:%s" % kv for kv in meta.get("inspvax_pos_type", {}).items()),
             base_height=meta.get("base_height"), frames_checked=len(frames),
             align_m=float(np.nanmedian(res)) if res and np.isfinite(res).any() else np.nan,
             align_p90_m=float(np.nanpercentile(res, 90)) if res and np.isfinite(res).any() else np.nan,
             ground_z_m=float(np.nanmedian(gz)) if len(gz) else np.nan,
             tilt_deg=float(np.nanmedian(tilt)) if len(tilt) else np.nan,
             ins_step_m=float(steps.max()) if len(steps) else np.nan,
             ins_steps_over_10cm=int(np.sum(steps > 0.1)))
    status = {k: grade(k, c[k]) for k in LIMITS}
    pts = [str(p) for p in meta.get("inspvax_pos_type", {})]
    status["rtk"] = "OK" if pts and set(pts) <= {"56"} else ("WARN" if pts else "-")
    order = {"OK": 0, "-": 0, "WARN": 1, "FAIL": 2}
    c["status"] = max(status.values(), key=lambda s: order[s])
    c["problems"] = " ".join("%s=%s" % (k, v) for k, v in status.items() if v in ("WARN", "FAIL"))
    return c, dict(frames=frames, xy=xy, mid=mid_frame, stamps=stamps, odom=od, speed=speed, res=res, res_t=res_t,
                   gz=gz, tilt=tilt, step_t=step_t, steps=steps)


def draw(path, c, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1, 0.8])
    t0 = d["stamps"][0] if len(d["stamps"]) else 0.0

    ax = fig.add_subplot(gs[:2, 0])
    if d["xy"]:
        w = np.concatenate(d["xy"])
        tt = np.concatenate([np.full(len(x), f.stamp - t0) for x, f in zip(d["xy"], d["frames"])])
        if len(w) > 300000:
            k = np.random.default_rng(0).choice(len(w), 300000, replace=False)
            w, tt = w[k], tt[k]
        sc = ax.scatter(w[:, 0], w[:, 1], s=0.15, c=tt, cmap="turbo")
        fig.colorbar(sc, ax=ax, fraction=0.03, label="time [s]")
    od = d["odom"]
    if len(od):
        ax.plot(od[:, 1], od[:, 2], "k-", lw=1.2)
        ax.plot(od[0, 1], od[0, 2], "g^", ms=9)
        ax.plot(od[-1, 1], od[-1, 2], "rs", ms=7)
    ax.set_aspect("equal")
    ax.set_title("world map of %d frames (vertical structures; sharp walls = good poses)" % len(d["frames"]),
                 fontsize=10)

    m = d["mid"]
    ax1, ax2 = fig.add_subplot(gs[0, 1:]), fig.add_subplot(gs[1, 1:])
    if m is not None:
        p = m.points
        cur = p[:, 4] <= 1e-6
        ax1.scatter(p[~cur, 0], p[~cur, 1], s=0.05, c="0.7")
        ax1.scatter(p[cur, 0], p[cur, 1], s=0.08, c=np.clip(p[cur, 2], -0.5, 3), cmap="viridis")
        sq = XY_RANGE * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        ax1.plot(sq[:, 0], sq[:, 1], "r--", lw=0.8)
        for r in (10, 20):
            a = np.linspace(0, 2 * np.pi, 200)
            ax1.plot(r * np.cos(a), r * np.sin(a), ":", c="0.5", lw=0.6)
        ax1.set_aspect("equal")
        ax1.set_xlim(-45, 45)
        ax1.set_ylim(-45, 45)
        ax1.set_title("model input, middle frame (5 sweeps, det frame; grey = older sweeps)", fontsize=10)
        sel = cur & (np.abs(p[:, 1]) < 3)
        ax2.scatter(p[sel, 0], p[sel, 2], s=0.2, c="k")
        ax2.axhline(0, c="r", lw=0.8)
        ax2.set_xlim(-45, 45)
        ax2.set_ylim(-1.5, 5)
        ax2.set_title("side view of the middle frame, |y| < 3 m (ground must lie on the red line z = 0)",
                      fontsize=10)

    ax3 = fig.add_subplot(gs[2, 0])
    if len(od):
        ax3.plot(od[:, 0] - t0, d["speed"], lw=0.8, label="speed [m/s]")
    if d["res"]:
        ax3.plot(np.array(d["res_t"]) - t0, d["res"], "o-", ms=3, lw=0.8, label="alignment [m]")
    big = d["steps"] > 0.05
    if big.any():
        ax3.plot(d["step_t"][big] - t0, d["steps"][big], "v", c="C3", ms=4, label="INS position step [m]")
    ax3.axhline(LIMITS["align_m"][1], c="r", ls=":", lw=0.8)
    ax3.legend(fontsize=8)
    ax3.set_xlabel("time [s]")
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(gs[2, 1])
    st = d["stamps"]
    if len(st) > 1:
        ax4.plot(st[1:] - t0, np.diff(st), ".", ms=2)
    ax4.set_title("lidar interval [s]", fontsize=10)
    ax4.grid(alpha=0.3)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")
    lines = ["%s   [%s]" % (c["name"], c["status"]), "split %s, %d scans, %.1f s, path %.0f m, %.2f m/s" % (
        c["split"] or "-", c["scans"], c["duration_s"], c["path_m"], c["mean_speed"]),
        "lidar %.1f Hz (gap %.2f s), imu %.0f Hz, odom %.0f Hz" % (c["lidar_hz"], c["lidar_gap_s"], c["imu_hz"],
                                                                  c["odom_hz"]),
        "pos_type %s, base height %.2f m" % (c["pos_type"] or "-", c["base_height"] or np.nan),
        "alignment %.3f m (p90 %.3f)" % (c["align_m"], c["align_p90_m"]),
        "ground z %.3f m, tilt %.2f deg" % (c["ground_z_m"], c["tilt_deg"]),
        "INS position steps: max %.2f m, %d over 0.1 m" % (c["ins_step_m"], c["ins_steps_over_10cm"]),
        "problems: %s" % (c["problems"] or "none")]
    ax5.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9,
             color={"OK": "k", "WARN": "darkorange", "FAIL": "red"}[c["status"]])
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


INDEX_COLUMNS = ["name", "status", "split", "scans", "duration_s", "path_m", "mean_speed", "pos_type", "align_m",
                 "ground_z_m", "tilt_deg", "ins_step_m", "lidar_gap_s", "problems"]


def write_index(out, rows):
    fields = list(rows[0].keys())
    with open(os.path.join(out, "checks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "%.3f" % v if isinstance(v, float) else str(v)   # noqa: E731
    colour = {"OK": "#2e7d32", "WARN": "#ef6c00", "FAIL": "#c62828"}
    body = []
    for r in rows:
        cells = "".join("<td>%s</td>" % fmt(r[k]) for k in INDEX_COLUMNS)
        body.append('<tr style="color:%s">%s<td><a href="%s.png"><img src="%s.png" width="360"></a></td></tr>'
                    % (colour.get(r["status"], "#000"), cells, r["name"], r["name"]))
    html = ("<!doctype html><meta charset='utf-8'><title>sequence preview</title>"
            "<style>body{font-family:sans-serif;font-size:13px}td,th{border-bottom:1px solid #ddd;padding:3px 6px}"
            "</style><h2>%d sequences: %d OK, %d WARN, %d FAIL</h2><table><tr>%s<th>preview</th></tr>%s</table>"
            % (len(rows), *[sum(r["status"] == s for r in rows) for s in ("OK", "WARN", "FAIL")],
               "".join("<th>%s</th>" % k for k in INDEX_COLUMNS), "\n".join(body)))
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True, help="sequence directories (globs are expanded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="check every n-th scan")
    args = ap.parse_args()
    dirs = sorted({d for pat in args.seq for d in (glob.glob(pat) or [pat])})
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "meta.json"))]
    if not dirs:
        raise SystemExit("no finished sequences (directories with meta.json) in %s" % args.seq)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for d in dirs:
        c, data = check_sequence(d, args.every, rng)
        draw(os.path.join(args.out, c["name"] + ".png"), c, data)
        rows.append(c)
        print("%-40s %-4s %4d scans %6.1f s  align %.3f m  ground %.3f m  tilt %.2f deg  INS step %.2f m  %s" % (
            c["name"], c["status"], c["scans"], c["duration_s"], c["align_m"], c["ground_z_m"], c["tilt_deg"],
            c["ins_step_m"], c["problems"]), flush=True)
    write_index(args.out, rows)
    with open(os.path.join(args.out, "checks.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    print("%d sequences: %d OK, %d WARN, %d FAIL -> %s/index.html" % (
        len(rows), *[sum(r["status"] == s for r in rows) for s in ("OK", "WARN", "FAIL")], args.out))


if __name__ == "__main__":
    main()
