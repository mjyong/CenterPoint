"""Summary of a long run on the Orin (lidar_3d_detection's scripts/orin_soak.sh): the node's once-a-second lines in time windows
(milliseconds of those frames, boxes, tracks), with and without the load; its warnings; its memory; tegrastats.

    python tools/deploy/soak_report.py data/deploy/T30/soak --window 300 --out soak.json

Times are the board's clock (node.log's ROS stamps, rss.log and load.log from ``date +%s``). node.log can also be
the vehicle node's (lidar_3d_detection, glog lines; their local time is read with the year of rss.log / load.log), whose
lines also count the clouds received, the frames out and the scans skipped, and give the longest callback: those
are summed per window. clocks.txt (orin_clocks.sh) is printed first: unlocked clocks make the node 2-3x slower.
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np

LINE = re.compile(r"\[(INFO|WARN|ERROR)\] \[(\d+\.\d+)\] \[dog_perception\]: (.*)")
# glog: Lmmdd or Lyyyymmdd, hh:mm:ss.uuuuuu, thread, file:line]
GLOG = re.compile(r"\b([IWEF])(\d{8}|\d{4}) (\d\d):(\d\d):(\d\d(?:\.\d+)?)\s+\d+ [^\]\s]+\] (.*)")
GLOG_LEVEL = dict(I="INFO", W="WARN", E="ERROR", F="ERROR")
STATS = re.compile(r"(\d+) dets, (\d+) tracks \|(.*)")


def _glog_time(day, hh, mm, ss, year):
    y, md = (int(day[:4]), day[4:]) if len(day) == 8 else (year, day)
    whole = time.mktime((y, int(md[:2]), int(md[2:]), int(hh), int(mm), 0, 0, 0, -1))
    return whole + float(ss)


def parse_line(line, year):
    """(level, t, text) of a ROS log or glog line, None for anything else."""
    m = LINE.search(line)
    if m:
        return m.group(1), float(m.group(2)), m.group(3)
    m = GLOG.search(line)
    if m:
        return GLOG_LEVEL[m.group(1)], _glog_time(*m.group(2, 3, 4, 5), year), m.group(6)
    return None


def read_node_log(path, year=None):
    year = year or time.localtime().tm_year
    frames, warnings = [], Counter()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = parse_line(line, year)
            if p is None:
                continue
            level, t, text = p
            s = STATS.match(text)
            if s:
                ms = {k: float(v) for k, v in re.findall(r"(\w+_ms) ([0-9.]+)", s.group(3))}
                # the vehicle node's counts since its previous line (clouds received, frames out, scans skipped)
                n = {k: int(v) for k, v in re.findall(r"\b(clouds|frames|skipped) (\d+)", s.group(3))}
                frames.append(dict(t=t, dets=int(s.group(1)), tracks=int(s.group(2)), **ms, **n))
            elif level != "INFO":
                warnings[level + ": " + re.sub(r"\d+(\.\d+)?", "#", text)] += 1
    return frames, warnings


def read_pairs(path):
    """'<epoch s> <value>' lines; lines without a value (the process was gone) are skipped."""
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    out.append((float(p[0]), float(p[1])))
    return out


def read_load(path):
    on = off = None
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.split()
                if p[:2] == ["load", "on"]:
                    on = float(p[2])
                elif p[:2] == ["load", "off"]:
                    off = float(p[2])
    return on, off


def read_tegrastats(path):
    gr3d, ram, temps = [], [], {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"GR3D_FREQ (\d+)%", line)
                if m:
                    gr3d.append(int(m.group(1)))
                m = re.search(r"RAM (\d+)/(\d+)MB", line)
                if m:
                    ram.append(int(m.group(1)))
                for name, v in re.findall(r"(\w+)@(-?[\d.]+)C", line):
                    temps[name] = max(temps.get(name, -1e9), float(v))
    return dict(samples=len(gr3d), gr3d_mean=float(np.mean(gr3d)) if gr3d else None,
                ram_mb_min=min(ram) if ram else None, ram_mb_max=max(ram) if ram else None, temp_max_c=temps)


def windows(frames, width, load):
    if not frames:
        return []
    t0, rows = frames[0]["t"], []
    for k in range(int((frames[-1]["t"] - t0) // width) + 1):
        w = [f for f in frames if t0 + k * width <= f["t"] < t0 + (k + 1) * width]
        if not w:
            continue
        total = np.array([f["total_ms"] for f in w])
        loaded = load[0] is not None and w[0]["t"] >= load[0] and (load[1] is None or w[-1]["t"] <= load[1])
        rows.append(dict(start_min=k * width / 60.0, load=bool(loaded), frames=len(w),
                         total_median=float(np.median(total)), total_p99=float(np.percentile(total, 99)),
                         total_max=float(total.max()), dets_median=float(np.median([f["dets"] for f in w])),
                         tracks_median=float(np.median([f["tracks"] for f in w]))))
        if all("clouds" in f for f in w):            # the vehicle node's counts (none in older logs)
            rows[-1].update(clouds=sum(f["clouds"] for f in w), frames_out=sum(f["frames"] for f in w),
                            skipped=sum(f["skipped"] for f in w),
                            callback_max=max(f.get("callback_max_ms", 0.0) for f in w))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="orin_soak.sh's output directory")
    ap.add_argument("--window", type=float, default=300.0, help="[s]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load = read_load(os.path.join(args.dir, "load.log"))
    rss = read_pairs(os.path.join(args.dir, "rss.log"))
    epoch = load[0] or (rss[0][0] if rss else None)
    frames, warnings = read_node_log(os.path.join(args.dir, "node.log"),
                                     time.localtime(epoch).tm_year if epoch else None)
    rows = windows(frames, args.window, load)
    report = dict(frames_logged=len(frames), windows=rows, warnings=dict(warnings),
                  rss_mb=dict(first=rss[0][1] / 1024, last=rss[-1][1] / 1024, max=max(r[1] for r in rss) / 1024)
                  if rss else None, tegrastats=read_tegrastats(os.path.join(args.dir, "tegrastats.log")))
    clocks = os.path.join(args.dir, "clocks.txt")             # orin_soak.sh: were the clocks locked
    if os.path.exists(clocks):
        with open(clocks, encoding="utf-8", errors="replace") as f:
            report["clocks"] = f.readline().strip()
        print(report["clocks"])
    print("| from min | load | frames logged | total ms median | p99 | max | boxes | tracks "
          "| clouds / out / skipped | callback max ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        counts = "%d / %d / %d" % (r["clouds"], r["frames_out"], r["skipped"]) if "clouds" in r else "-"
        cb = "%.1f" % r["callback_max"] if "clouds" in r else "-"
        print("| %.0f | %s | %d | %.1f | %.1f | %.1f | %.0f | %.0f | %s | %s |" % (
            r["start_min"], "yes" if r["load"] else "no", r["frames"], r["total_median"], r["total_p99"],
            r["total_max"], r["dets_median"], r["tracks_median"], counts, cb))
    print()
    for text, n in sorted(warnings.items(), key=lambda kv: -kv[1]):
        print("%6d  %s" % (n, text))
    print(json.dumps(dict(rss_mb=report["rss_mb"], tegrastats=report["tegrastats"])))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
