"""Survey a long kygi665 ROS 2 bag before converting it, and propose a split into scenes.

Reads the INSPVAX solutions (decoded) and only the header stamps of the lidar and IMU messages
(taken from the first bytes of the CDR message, the point clouds are not decoded), so a 30 GB
bag takes about one sequential read of the file.

    python tools/dog/bag_survey.py --bag /data/rosbag2_2026_09_21-16_56_59 --out data/survey/0921_165659 \
        [--scene-length 30 --min-length 10 --pos-types 56 --val-fraction 0.2]

Outputs in --out:

    survey.png    trajectory (ENU) with the proposed scenes, speed, RTK / INS status and the lidar
                  frame interval over time, scene boundaries shaded (train blue, val orange)
    survey.json   topics and counts, rates, gaps, pos_type histogram, valid time, scene summary
    segments.csv  one row per proposed scene: name, t_start, t_end (header stamps, seconds),
                  duration, scans, path_m, mean_speed, moving_share, pos_types, split, keep, note

A scene is cut out of the stretches where everything is usable: lidar frames at most
``--max-lidar-gap`` apart, IMU at most ``--max-imu-gap``, INS solution good with a pos_type in
``--pos-types`` (56 = INS_RTKFIXED). Each stretch is divided into scenes of about
``--scene-length`` seconds; stretches shorter than ``--min-length`` are left out. The last
``--val-fraction`` of the valid time is proposed as validation, with one buffer scene before it
left out (keep = 0) so near-identical frames do not end up on both sides; survey.json reports
how much of the validation path lies within 10 m of the training path (the same place seen
twice still leaks). Edit ``keep`` / ``split`` / ``name`` by hand, then convert with
``rosbag_to_sequence.py --segments``.

Clock check: per topic, the bag's log time minus the header stamp. All three sensors stamp in
GPS-synchronised time, so this is the transport delay: about one frame (0.1 s, the stamp is the
scan start) plus driver latency for the lidar, a few ms for IMU and INS. A value of seconds, a
5-95 % spread of more than ~50 ms, or a jump between parts of the bag means a sensor clock was
not synchronised, which the sequence checks cannot see while the robot drives straight.
"""
import argparse
import csv
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.inspvax import INS_SOLUTION_GOOD, EnuOrigin, register_kygi  # noqa: E402
from dog_perception.ros_utils import stamp_of  # noqa: E402

SEGMENT_FIELDS = ["name", "t_start", "t_end", "duration", "scans", "path_m", "mean_speed", "moving_share",
                  "pos_types", "split", "keep", "note"]


def cdr_header_stamp(raw):
    """Header stamp of a CDR-serialized message whose first field is std_msgs/Header."""
    if raw[1] != 1:                                  # 0x0001 = CDR little endian
        raise ValueError("big-endian CDR not supported")
    sec, nsec = struct.unpack_from("<iI", raw, 4)
    return sec + nsec * 1e-9


def read_bag(bag, lidar_topic, imu_topic, inspvax_topic, progress_every=2000):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    register_kygi(ts)
    lidar, lidar_bytes, imu, ins = [], [], [], []
    lag = {lidar_topic: [], imu_topic: [], inspvax_topic: []}         # bag log time - header stamp
    with AnyReader([Path(bag)], default_typestore=ts) as reader:
        topics = [dict(topic=c.topic, type=c.msgtype, count=c.msgcount) for c in reader.connections]
        want = {lidar_topic, imu_topic, inspvax_topic}
        conns = [c for c in reader.connections if c.topic in want]
        missing = want - {c.topic for c in conns}
        if missing:
            raise SystemExit("topics not in bag: %s (have: %s)" % (sorted(missing), sorted(t["topic"] for t in topics)))
        bag_start, bag_end = reader.start_time * 1e-9, reader.end_time * 1e-9
        n = 0
        for conn, log_ns, raw in reader.messages(connections=conns):
            if conn.topic == lidar_topic:
                lidar.append(cdr_header_stamp(raw))
                lag[lidar_topic].append(log_ns * 1e-9 - lidar[-1])
                lidar_bytes.append(len(raw))
                n += 1
                if progress_every and n % progress_every == 0:
                    print("  %d lidar frames read (%.0f s into the bag)" % (n, lidar[-1] - bag_start), flush=True)
            elif conn.topic == imu_topic:
                imu.append(cdr_header_stamp(raw))
                lag[imu_topic].append(log_ns * 1e-9 - imu[-1])
            else:
                m = reader.deserialize(raw, conn.msgtype)
                lag[inspvax_topic].append(log_ns * 1e-9 - stamp_of(m.header))
                ins.append((stamp_of(m.header), int(m.ins_status.data), int(m.pos_type.data), m.latitude,
                            m.longitude, m.height, m.azimuth, m.east_velocity, m.north_velocity))
    lag = {k: dict(median_s=float(np.median(v)), p5_s=float(np.percentile(v, 5)), p95_s=float(np.percentile(v, 95)))
           for k, v in lag.items() if v}
    return dict(topics=topics, bag_start=bag_start, bag_end=bag_end, lag=lag, lidar=np.sort(np.array(lidar)),
                lidar_bytes=np.array(lidar_bytes), imu=np.sort(np.array(imu)),
                ins=np.array(sorted(ins), np.float64).reshape(-1, 9))


def ins_track(ins):
    """INS rows -> t, good mask, pos_type, ENU xy (about the first good fix), speed, yaw rate [deg/s]."""
    t, status, ptype = ins[:, 0], ins[:, 1].astype(int), ins[:, 2].astype(int)
    good = status == INS_SOLUTION_GOOD
    xy = np.full((len(ins), 2), np.nan)
    if good.any():
        k0 = int(np.argmax(good))
        origin = EnuOrigin(*ins[k0, 3:6])
        for k in np.nonzero(good)[0]:
            xy[k] = origin.enu(*ins[k, 3:6])[:2]
    speed = np.hypot(ins[:, 7], ins[:, 8])
    yaw = np.unwrap(np.deg2rad(-ins[:, 6]))
    yaw_rate = np.degrees(np.gradient(yaw, t)) if len(t) > 2 else np.zeros(len(t))
    return t, good, ptype, xy, speed, yaw_rate


def valid_intervals(times_ok, max_gap):
    """Sorted times of usable samples -> [(t0, t1)] where consecutive samples are <= max_gap apart."""
    if len(times_ok) == 0:
        return []
    cut = np.nonzero(np.diff(times_ok) > max_gap)[0]
    starts = np.r_[times_ok[0], times_ok[cut + 1]]
    ends = np.r_[times_ok[cut], times_ok[-1]]
    return list(zip(starts, ends))


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def propose_segments(intervals, scene_length, min_length):
    segs = []
    for lo, hi in intervals:
        dur = hi - lo
        if dur < min_length:
            continue
        n = max(1, int(round(dur / scene_length)))
        edges = np.linspace(lo, hi + 0.05, n + 1)      # +half a frame: the last scan's stamp is hi
        segs += list(zip(edges[:-1], edges[1:]))
    return segs


def segment_rows(segs, prefix, lidar, ins_t, good, ptype, xy, speed, moving_speed, val_fraction):
    rows = []
    for i, (lo, hi) in enumerate(segs):
        m = (ins_t >= lo) & (ins_t < hi) & good
        p = xy[m]
        path = float(np.nansum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0
        types = {int(k): int(v) for k, v in zip(*np.unique(ptype[(ins_t >= lo) & (ins_t < hi)], return_counts=True))}
        rows.append(dict(name="%s_s%03d" % (prefix, i), t_start="%.6f" % lo, t_end="%.6f" % hi,
                         duration="%.1f" % (hi - lo), scans=int(((lidar >= lo) & (lidar < hi)).sum()),
                         path_m="%.1f" % path, mean_speed="%.2f" % (float(np.mean(speed[m])) if m.any() else 0.0),
                         moving_share="%.2f" % (float(np.mean(speed[m] > moving_speed)) if m.any() else 0.0),
                         pos_types=" ".join("%d:%d" % kv for kv in types.items()), split="train", keep=1, note=""))
    # validation: the last val_fraction of the valid time, one buffer scene before it left out
    total = sum(float(r["duration"]) for r in rows)
    acc, k = 0.0, len(rows)
    while k > 0 and acc < val_fraction * total - 1e-6:
        k -= 1
        acc += float(rows[k]["duration"])
    if 0 < k < len(rows) and len(rows) >= 3:
        for r in rows[k:]:
            r["split"] = "val"
        rows[k - 1].update(keep=0, note="buffer before val")
    for r in rows:
        if float(r["moving_share"]) < 0.1:
            r["note"] = (r["note"] + "; " if r["note"] else "") + "static (robot not moving)"
    return rows


def leak_share(rows, ins_t, good, xy, radius=10.0):
    """Share of the val path within ``radius`` m of the train path."""
    from scipy.spatial import cKDTree

    def path(split):
        m = np.zeros(len(ins_t), bool)
        for r in rows:
            if r["split"] == split and int(r["keep"]):
                m |= (ins_t >= float(r["t_start"])) & (ins_t < float(r["t_end"]))
        return xy[m & good & np.isfinite(xy).all(1)]

    tr, va = path("train"), path("val")
    if len(tr) == 0 or len(va) == 0:
        return None
    d, _ = cKDTree(tr).query(va)
    return float(np.mean(d <= radius))


def plot(path, d, t, good, ptype, xy, speed, yaw_rate, rows, intervals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = d["bag_start"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1.4])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(xy[:, 0], xy[:, 1], "-", c="0.8", lw=1, label="INS good")
    for r in rows:
        m = (t >= float(r["t_start"])) & (t < float(r["t_end"]))
        c = "0.5" if not int(r["keep"]) else ("tab:orange" if r["split"] == "val" else "tab:blue")
        ax.plot(xy[m, 0], xy[m, 1], "-", c=c, lw=2)
        if m.any() and np.isfinite(xy[m][0]).all():
            ax.annotate(r["name"].rsplit("_", 1)[-1], xy[m][0], fontsize=7)
    ax.plot(*xy[np.argmax(good)], "g^", ms=10, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("trajectory (blue train, orange val, grey left out)")
    ax.legend(loc="best", fontsize=8)

    axs = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for a in axs[1:]:
        a.sharex(axs[0])
    rel = t - t0
    axs[0].plot(rel, speed, lw=0.8)
    axs[0].set_ylabel("speed [m/s]")
    axs[1].plot(rel, yaw_rate, lw=0.8)
    axs[1].set_ylabel("yaw rate [deg/s]")
    axs[1].set_ylim(-60, 60)
    axs[2].step(rel, ptype, where="post", lw=0.8, label="pos_type")
    axs[2].step(rel, np.where(good, 1, 0) * 10, where="post", lw=0.8, label="INS good x10")
    axs[2].set_ylabel("RTK / INS")
    axs[2].legend(fontsize=7, loc="lower right")
    li = d["lidar"]
    if len(li) > 1:
        axs[3].plot(li[1:] - t0, np.diff(li), ".", ms=1.5)
    axs[3].set_ylabel("lidar interval [s]")
    axs[3].set_yscale("log")
    axs[3].set_xlabel("time since bag start [s]")
    for a in axs:
        for r in rows:
            c = "0.5" if not int(r["keep"]) else ("tab:orange" if r["split"] == "val" else "tab:blue")
            a.axvspan(float(r["t_start"]) - t0, float(r["t_end"]) - t0, color=c, alpha=0.12, lw=0)
        a.grid(alpha=0.3)
    fig.suptitle("%s   %.0f s, %d lidar frames, %d scenes (%d val)" % (
        os.path.basename(os.path.normpath(d["bag"])), d["bag_end"] - d["bag_start"], len(li), len(rows),
        sum(r["split"] == "val" and int(r["keep"]) for r in rows)))
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lidar-topic", default="/robot/topic/rdap/top_point_cloud2")
    ap.add_argument("--imu-topic", default="/rte/rdap/rtk/imus/data_raw")
    ap.add_argument("--inspvax-topic", default="/rte/rdap/rtk/inspvax")
    ap.add_argument("--prefix", default=None, help="scene name prefix (default: from the bag name)")
    ap.add_argument("--scene-length", type=float, default=30.0, help="[s] target scene length")
    ap.add_argument("--min-length", type=float, default=10.0, help="[s] shorter usable stretches are left out")
    ap.add_argument("--pos-types", type=int, nargs="+", default=[56], help="usable RTK pos_type (56 = RTK fixed)")
    ap.add_argument("--max-lidar-gap", type=float, default=0.3)
    ap.add_argument("--max-imu-gap", type=float, default=0.1)
    ap.add_argument("--max-ins-gap", type=float, default=0.2)
    ap.add_argument("--moving-speed", type=float, default=0.3, help="[m/s] for moving_share")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.bag))
    prefix = args.prefix or name.replace("rosbag2_", "")

    print("reading %s ..." % args.bag, flush=True)
    d = read_bag(args.bag, args.lidar_topic, args.imu_topic, args.inspvax_topic)
    d["bag"] = args.bag
    t, good, ptype, xy, speed, yaw_rate = ins_track(d["ins"])
    usable = good & np.isin(ptype, args.pos_types)
    intervals = intersect(intersect(valid_intervals(d["lidar"], args.max_lidar_gap),
                                    valid_intervals(d["imu"], args.max_imu_gap)),
                          valid_intervals(t[usable], args.max_ins_gap))
    segs = propose_segments(intervals, args.scene_length, args.min_length)
    rows = segment_rows(segs, prefix, d["lidar"], t, good, ptype, xy, speed, args.moving_speed, args.val_fraction)
    with open(os.path.join(args.out, "segments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    plot(os.path.join(args.out, "survey.png"), d, t, good, ptype, xy, speed, yaw_rate, rows, intervals)

    li, im = d["lidar"], d["imu"]
    gaps = lambda x: float(np.max(np.diff(x))) if len(x) > 1 else None   # noqa: E731
    rate = lambda x: float((len(x) - 1) / (x[-1] - x[0])) if len(x) > 1 else None   # noqa: E731
    valid = sum(hi - lo for lo, hi in intervals)
    kept = [r for r in rows if int(r["keep"])]
    summary = dict(
        bag=args.bag, duration_s=d["bag_end"] - d["bag_start"], topics=d["topics"],
        lidar=dict(frames=len(li), rate_hz=rate(li), max_gap_s=gaps(li),
                   gaps_over_limit=int(np.sum(np.diff(li) > args.max_lidar_gap)),
                   mean_msg_mb=float(np.mean(d["lidar_bytes"]) / 2 ** 20) if len(li) else None),
        imu=dict(msgs=len(im), rate_hz=rate(im), max_gap_s=gaps(im)),
        ins=dict(msgs=len(t), rate_hz=rate(t), good_share=float(np.mean(good)) if len(t) else None,
                 pos_type={int(k): int(v) for k, v in zip(*np.unique(ptype, return_counts=True))},
                 path_m=float(np.nansum(np.hypot(*np.diff(xy[good], axis=0).T))) if good.sum() > 1 else 0.0),
        log_minus_header=d["lag"], usable_s=valid,
        usable_intervals=[(float(a) - d["bag_start"], float(b) - d["bag_start"]) for a, b in intervals],
        scenes=len(rows), scenes_kept=len(kept),
        train_s=sum(float(r["duration"]) for r in kept if r["split"] == "train"),
        val_s=sum(float(r["duration"]) for r in kept if r["split"] == "val"),
        val_path_within_10m_of_train=leak_share(rows, t, good, xy))
    with open(os.path.join(args.out, "survey.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("bag %.0f s: lidar %d frames (%.1f Hz, max gap %.2f s, %.1f MB/msg), imu %.0f Hz, ins %.0f Hz" % (
        summary["duration_s"], len(li), rate(li) or 0, gaps(li) or 0, summary["lidar"]["mean_msg_mb"] or 0,
        rate(im) or 0, rate(t) or 0))
    for k, v in d["lag"].items():
        print("  %-40s log - header stamp: median %+.3f s (5-95%%: %+.3f .. %+.3f)" % (k, v["median_s"], v["p5_s"],
                                                                                    v["p95_s"]))
    print("INS good %.0f%%, pos_type %s, path %.0f m; usable %.0f s in %d stretches" % (
        100 * (summary["ins"]["good_share"] or 0), summary["ins"]["pos_type"], summary["ins"]["path_m"], valid,
        len(intervals)))
    leak = summary["val_path_within_10m_of_train"]
    print("%d scenes (%d kept): train %.0f s, val %.0f s; val path within 10 m of train: %s" % (
        len(rows), len(kept), summary["train_s"], summary["val_s"], "-" if leak is None else "%.0f%%" % (100 * leak)))
    print("-> %s/{segments.csv,survey.png,survey.json}" % args.out)


if __name__ == "__main__":
    main()
