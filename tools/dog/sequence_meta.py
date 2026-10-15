"""Show or fix meta.json of converted sequences without converting the bag again.

    python tools/dog/sequence_meta.py --seq data/seq/0921_*                              # table only
    python tools/dog/sequence_meta.py --seq data/seq/0921_* --base-height auto           # median over sweeps
    python tools/dog/sequence_meta.py --seq data/seq/0921_165659_s002 --base-height 0.72  # fixed value
    python tools/dog/sequence_meta.py --seq data/seq/0921_* --lidar-time-offset -0.03

``--base-height auto`` runs the converter's estimate (``estimate_base_height``) on every
``--every``-th sweep and takes the median, which is what ``rosbag_to_sequence.py`` now writes;
sequences converted before used the first sweep alone, which scattered 0.59-1.05 m for the standing
robot. The value it replaces is kept once as ``base_height_first_sweep``.

``--lidar-time-offset`` sets the offset that ``SequenceReader`` adds to the lidar times (measure it
with ``tools/dog/time_offset_check.py``). Both only change meta.json: run the preview / auto-label
again afterwards.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.geometry import transform_points  # noqa: E402
from dog_perception.io import SequenceReader  # noqa: E402
from dog_perception.preprocess import estimate_base_height, median_base_height  # noqa: E402


def sweep_base_height(rd, every=10):
    hs = []
    for f in rd.scan_files[::every]:
        with np.load(f) as d:
            hs.append(estimate_base_height(transform_points(rd.T_body_lidar, d["xyz"].astype(np.float64))))
    return median_base_height(hs)


def write_meta(seq_dir, meta):
    p = os.path.join(seq_dir, "meta.json")
    with open(p + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(p + ".tmp", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--base-height", default=None, help="'auto' (median over sweeps) or a value in metres")
    ap.add_argument("--every", type=int, default=10, help="with --base-height auto: use every k-th sweep")
    ap.add_argument("--lidar-time-offset", type=float, default=None, help="[s] added to the lidar times on reading")
    ap.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    args = ap.parse_args()

    print("%-26s %-5s %5s  %-24s %-15s %s" % ("sequence", "split", "scans", "base_height [m]", "sweeps p10-p90",
                                             "lidar_time_offset [s]"))
    for seq in sorted(args.seq):
        if not os.path.exists(os.path.join(seq, "meta.json")):
            print("%-26s  no meta.json (unfinished conversion) - skipped" % os.path.basename(os.path.normpath(seq)))
            continue
        rd = SequenceReader(seq)
        meta = dict(rd.meta)
        old_h, old_dt = meta.get("base_height"), meta.get("lidar_time_offset", 0.0)
        spread = meta.get("base_height_sweeps")
        if args.base_height is not None:
            if args.base_height == "auto":
                h, info = sweep_base_height(rd, args.every)
                spread = dict(info, every=args.every)
            else:
                h, spread = float(args.base_height), dict(fixed=True)
            if h is not None:
                if "base_height_sweeps" not in meta and old_h is not None:
                    meta.setdefault("base_height_first_sweep", old_h)
                meta["base_height"], meta["base_height_sweeps"] = h, spread
        if args.lidar_time_offset is not None:
            meta["lidar_time_offset"] = args.lidar_time_offset
        changed = meta != rd.meta
        if changed and not args.dry_run:
            write_meta(seq, meta)
        h_txt = "%.3f" % meta["base_height"] if meta.get("base_height") is not None else "-"
        if old_h is not None and meta.get("base_height") != old_h:
            h_txt = "%.3f -> %s" % (old_h, h_txt)
        sp = "%.2f-%.2f (n %d)" % (spread["p10"], spread["p90"], spread["n"]) if spread and "p10" in spread else \
            ("fixed" if spread and spread.get("fixed") else "-")
        dt_txt = "%+.3f" % meta.get("lidar_time_offset", 0.0)
        if meta.get("lidar_time_offset", 0.0) != old_dt:
            dt_txt = "%+.3f -> %s" % (old_dt, dt_txt)
        print("%-26s %-5s %5d  %-24s %-15s %s%s" % (os.path.basename(os.path.normpath(seq)),
                                                   meta.get("segment", {}).get("split", ""), len(rd), h_txt, sp,
                                                   dt_txt, "   (dry run)" if changed and args.dry_run else ""))


if __name__ == "__main__":
    main()
