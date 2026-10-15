"""Time offset between the lidar and the IMU / localization clock, measured on turns.

    python tools/dog/time_offset_check.py --seq data/seq/0921_164127_s003 data/seq/0921_165659_s026 \
        --out data/preview/time_offset [--offsets -0.15 0.15 0.025]

A constant offset puts every sweep at the pose of a slightly wrong moment. While the yaw rate is
steady, all sweeps are turned by the same wrong angle and still line up with each other, which is
why ``sequence_preview.py`` cannot see it on straight walking. When the yaw rate changes (a turn
starts or stops, an in-place spin) two sweeps 0.5 s apart get different errors, about
range x (yaw-rate change) x offset: at 10 m, a 90 deg/s change and 0.05 s give 0.8 m. So this
check takes the sweep pairs ~0.5 s apart whose mean yaw rate (IMU) differs by at least
``--turn-dw`` deg/s, computes the same alignment residual as the preview (vertical structures,
world frame) with the lidar times shifted by each candidate offset, and picks the offset with the
smallest residual. Steady pairs (yaw rate change < 5 deg/s) are reported as a baseline: they must
stay flat.

Offsets are added to the lidar times (on top of ``lidar_time_offset`` already in meta.json): a
lidar that stamps 0.05 s late needs -0.05.

Output in ``--out``: time_offset.csv (sequence x offset: pairs, median / p90 residual of the
turning pairs, median of the steady pairs), <sequence>.png (residual vs offset; residual vs time at
0 and at the best offset, with the yaw-rate change), time_offset.json (best offset per sequence and
over all of them, and a verdict).

Reading it: the same minimum away from 0 on several sequences (|offset| >= 0.02 s and a turning
residual at least 1.5x smaller than at 0) -> write it into the sequences with
``tools/dog/sequence_meta.py --lidar-time-offset`` and into the node's ``lidar_time_offset``
parameter. A flat curve or a minimum at 0 -> the clocks agree, and turn spikes in the preview come
from the view (body shadow, few structures), not from the poses.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sequence_preview import align_residual, pipeline_for, structure_xy  # noqa: E402

from dog_perception.io import SequenceReader  # noqa: E402

STEADY_DW = 5.0     # [deg/s] yaw-rate change below which a pair counts as steady


def sweep_yaw_rate(imu, stamps, span=0.1):
    """Mean body z rate [deg/s] over each sweep (``span`` before its end stamp), from the gyro."""
    if len(imu) < 2:
        return np.zeros(len(stamps))
    t, wz = imu[:, 0], imu[:, 3]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (wz[1:] + wz[:-1]) * np.diff(t))])
    a, b = np.interp(stamps - span, t, cum), np.interp(stamps, t, cum)
    return np.degrees((b - a) / span)


def select_pairs(stamps, rate, gap=0.5, turn_dw=20.0, steady_every=10):
    """(i, j, |rate change| [deg/s], kind) for sweeps ``gap`` apart: every turning pair, and every
    ``steady_every``-th steady one."""
    out = []
    for i in range(len(stamps)):
        j = int(np.searchsorted(stamps, stamps[i] + gap - 0.05))
        if j >= len(stamps) or stamps[j] - stamps[i] > gap + 0.3:
            continue
        dw = abs(rate[j] - rate[i])
        if dw >= turn_dw:
            out.append((i, j, dw, "turn"))
        elif dw < STEADY_DW and i % steady_every == 0:
            out.append((i, j, dw, "steady"))
    return out


def scan_stamps(rd):
    stamps = []
    for f in rd.scan_files:
        with np.load(f) as d:
            stamps.append(float(d["stamp"]))
    return np.array(stamps)


def residuals_at(seq_dir, offset, pairs, seed=0):
    """Alignment residual [m] of each pair with ``offset`` added to the lidar times."""
    rd = SequenceReader(seq_dir)
    rd.lidar_time_offset += offset
    pipe = pipeline_for(rd)
    sel = sorted({k for i, j, _, _ in pairs for k in (i, j)})
    xy, it = {}, iter(sel)
    for ev in rd.events(indices=sel):
        if ev[0] == "imu":
            pipe.on_imu(ev[1], ev[2])
        elif ev[0] == "odom":
            pipe.on_odometry(ev[1], ev[2], ev[3], ev[4])
        else:
            k = next(it)
            fr = pipe.pre.process(ev[1])
            xy[k] = structure_xy(fr) if fr is not None else np.zeros((0, 2))
    rng = np.random.default_rng(seed)
    return np.array([align_residual(xy[i], xy[j], rng) for i, j, _, _ in pairs])


def stats(r):
    r = r[np.isfinite(r)]
    return (float(np.median(r)), float(np.percentile(r, 90))) if len(r) else (np.nan, np.nan)


def check(seq_dir, offsets, gap=0.5, turn_dw=20.0, min_pairs=3):
    rd = SequenceReader(seq_dir)
    stamps = scan_stamps(rd) + rd.lidar_time_offset
    rate = sweep_yaw_rate(rd.imu, stamps)
    pairs = select_pairs(stamps, rate, gap, turn_dw)
    turn = np.array([p[3] == "turn" for p in pairs], bool)
    name = os.path.basename(os.path.normpath(seq_dir))
    res = dict(name=name, meta_offset=rd.lidar_time_offset, n_turn=int(turn.sum()), n_steady=int((~turn).sum()),
               pair_t=[float(stamps[i] - stamps[0]) for i, _, _, _ in pairs], pair_dw=[p[2] for p in pairs],
               turn=turn.tolist(), rows=[], per_pair={})
    print("%s: %d turning pairs (yaw-rate change >= %.0f deg/s), %d steady" % (name, res["n_turn"], turn_dw,
                                                                                 res["n_steady"]), flush=True)
    if res["n_turn"] < min_pairs:
        return res
    for off in offsets:
        r = residuals_at(seq_dir, off, pairs)
        (tm, tp), (sm, _) = stats(r[turn]), stats(r[~turn])
        res["rows"].append(dict(name=name, offset=float(off), n_turn=res["n_turn"], turn_median_m=tm, turn_p90_m=tp,
                                n_steady=res["n_steady"], steady_median_m=sm))
        res["per_pair"][float(off)] = r.tolist()
        print("  offset %+.3f s: turning %.3f m (p90 %.3f), steady %.3f m" % (off, tm, tp, sm), flush=True)
    rows = res["rows"]
    k = int(np.nanargmin([x["turn_median_m"] for x in rows]))
    at0 = next(x for x in rows if abs(x["offset"]) < 1e-9)
    res.update(best_offset=rows[k]["offset"], turn_median_at_0=at0["turn_median_m"],
               turn_median_at_best=rows[k]["turn_median_m"])
    return res


def verdict(best, at0, at_best, n_seq, step, per_seq=()):
    """``per_seq``: (name, best offset) of each sequence; a clock offset is the same on all of them."""
    off = [(n, b) for n, b in per_seq if abs(b - best) > step + 1e-9]
    if off:
        return ("sequences disagree (best %s; %s): look at the PNGs one by one - a clock offset is the same "
                "on every sequence, so this is not one" % ("%+.3f s over all" % best,
                                                          ", ".join("%s %+.3f s" % x for x in per_seq)))
    gain = at0 / at_best if at_best > 0 else np.inf
    if abs(best) >= max(0.02, step) and gain >= 1.5:
        return ("offset %+.3f s: turning residual %.3f -> %.3f m (%.1fx) over %d sequence(s). Apply it with "
                "tools/dog/sequence_meta.py --lidar-time-offset and the node parameter lidar_time_offset"
                % (best, at0, at_best, gain, n_seq))
    return ("no clear offset: best %+.3f s changes the turning residual only %.3f -> %.3f m; the clocks agree, "
            "turn spikes in the preview are not timing" % (best, at0, at_best))


def draw(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    off = np.array([x["offset"] for x in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(off, [x["turn_median_m"] for x in rows], "o-", label="turning pairs, median")
    a1.plot(off, [x["turn_p90_m"] for x in rows], ":", c="C0", label="turning pairs, p90")
    a1.plot(off, [x["steady_median_m"] for x in rows], "s-", c="C2", label="steady pairs, median")
    a1.axvline(res["best_offset"], c="r", lw=0.8)
    a1.set_xlabel("offset added to lidar times [s]")
    a1.set_ylabel("alignment residual [m]")
    a1.set_title("%s: best %+.3f s" % (res["name"], res["best_offset"]))
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    t, turn = np.array(res["pair_t"]), np.array(res["turn"])
    for o, c in ((0.0, "C0"), (res["best_offset"], "C3")):
        r = np.array(res["per_pair"][float(o)])
        a2.plot(t[turn], r[turn], "o", ms=4, c=c, label="offset %+.3f s" % o)
    a2.set_xlabel("time [s]")
    a2.set_ylabel("residual of turning pairs [m]")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8, loc="upper left")
    b = a2.twinx()
    b.plot(t, res["pair_dw"], ".", ms=2, c="0.6")
    b.set_ylabel("yaw-rate change over the pair [deg/s]", color="0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True, help="sequence directories (pick ones with turns)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", type=float, nargs=3, default=[-0.15, 0.15, 0.025], metavar=("FROM", "TO", "STEP"))
    ap.add_argument("--gap", type=float, default=0.5, help="[s] time between the two sweeps of a pair")
    ap.add_argument("--turn-dw", type=float, default=20.0, help="[deg/s] yaw-rate change that makes a turning pair")
    ap.add_argument("--min-pairs", type=int, default=3)
    args = ap.parse_args()
    lo, hi, step = args.offsets
    offsets = np.unique(np.round(np.r_[np.arange(lo, hi + step / 2, step), 0.0], 6)) + 0.0   # + 0.0: no -0.0
    os.makedirs(args.out, exist_ok=True)

    results = [check(s, offsets, args.gap, args.turn_dw, args.min_pairs) for s in args.seq]
    used = [r for r in results if r["rows"]]
    with open(os.path.join(args.out, "time_offset.csv"), "w", newline="") as f:
        cols = ["name", "offset", "n_turn", "turn_median_m", "turn_p90_m", "n_steady", "steady_median_m"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in used:
            w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in r["rows"]])
    for r in used:
        draw(os.path.join(args.out, r["name"] + ".png"), r)

    summary = dict(offsets=offsets.tolist(), turn_dw_deg_s=args.turn_dw, gap_s=args.gap,
                   sequences=[{k: r.get(k) for k in ("name", "meta_offset", "n_turn", "n_steady", "best_offset",
                                                    "turn_median_at_0", "turn_median_at_best")} for r in results])
    if used:
        # every sequence counts the same: residual relative to its own value at offset 0
        rel = np.array([[x["turn_median_m"] / r["turn_median_at_0"] for x in r["rows"]] for r in used])
        k = int(np.nanargmin(np.nanmean(rel, axis=0)))
        best = float(offsets[k])
        at0 = float(np.mean([r["turn_median_at_0"] for r in used]))
        at_best = float(np.mean([r["rows"][k]["turn_median_m"] for r in used]))
        summary.update(best_offset=best, verdict=verdict(best, at0, at_best, len(used), step,
                                                         [(r["name"], r["best_offset"]) for r in used]))
    else:
        summary["verdict"] = ("no sequence has %d pairs with a yaw-rate change >= %.0f deg/s: pick sequences with "
                              "turns" % (args.min_pairs, args.turn_dw))
    with open(os.path.join(args.out, "time_offset.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
