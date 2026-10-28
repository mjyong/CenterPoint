"""How many pillars the recorded frames need, to size P_max of the PFN graph.

The PFN engine has a static input of P_max pillars; a frame with more loses the rest (the exported
detector counts them in ``last_pillars``). PFN time grows with P_max, so it should not be much more
than the largest frame needs.

    python tools/deploy/pillar_stats.py --model-dir data/deploy/nusc_pp \\
        --frames "data/runs/*/frames/*.npy" --stride 5 --jobs 8 --out data/reports/T26/pillars.json

Prints the distribution and a suggested P_max: the maximum times ``--margin``, rounded up to 1000.
"""
import argparse
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.model_meta import ModelMeta  # noqa: E402
from dog_perception.detection.voxelize import voxelize  # noqa: E402

_META = None


def _init(path):
    global _META
    _META = ModelMeta.load(path)


def count(path):
    m = _META
    pts = np.load(path)
    z = pts[:, 2]
    pts = pts[(z >= m.z_range[0]) & (z <= m.z_range[1])]
    pts = m.model_frame.points_to_model(pts)[:, :m.num_point_features]
    v, _, n = voxelize(pts, m.voxel_size, m.pc_range, m.max_points, 10 ** 9)
    return len(v), int((n >= m.max_points).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--margin", type=float, default=1.2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(f for p in args.frames for f in glob.glob(p))[::args.stride]
    meta_path = os.path.join(args.model_dir, "model_meta.json")
    if args.jobs > 1:
        with Pool(args.jobs, initializer=_init, initargs=(meta_path,)) as pool:
            res = pool.map(count, files, chunksize=8)
    else:
        _init(meta_path)
        res = [count(f) for f in files]
    p = np.array([r[0] for r in res])
    full = np.array([r[1] for r in res])
    m = ModelMeta.load(meta_path)
    out = dict(frames=len(files), max=int(p.max()), p50=float(np.median(p)), p99=float(np.percentile(p, 99)),
               p999=float(np.percentile(p, 99.9)), mean=float(p.mean()), current_max_pillars=m.max_pillars,
               frames_over_current=int((p > m.max_pillars).sum()), full_pillars_mean=float(full.mean()),
               suggested_max_pillars=int(math.ceil(p.max() * args.margin / 1000.0) * 1000),
               worst_frame=files[int(np.argmax(p))])
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
