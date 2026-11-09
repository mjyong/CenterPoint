#!/usr/bin/env python3
"""Is a lightning localization map level? Ground plane of its 3D points against the vertical.

    python3 Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py <map dir | .pcd> [--max-tilt 5]

The ground is the lowest point of each 0.5 m cell, fitted with a plane. A map built for a robot walking on flat
ground has it within a few degrees of level. The built-in fallback map 0c3f8a2d-... (8 keyframes, 15 m of nearly
straight walking from the 0921 bag) is tilted about 51 deg: aligned to INS positions along a line, its rotation
about that line was free. Localizing against such a map matches a few dozen points per scan (T42: effect_num
median 51), declares a fault every 10 s and jumps.

Reads <map>/3d/global.pcd, or every <map>/3d/*.pcd tile when there is no global.pcd. PCD ascii, binary and
binary_compressed (LZF), float fields. Exit code 1 when the tilt is above --max-tilt. Needs numpy only.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np


def lzf_decompress(data, out_len):
    """LZF (liblzf) as PCD binary_compressed stores it."""
    out = bytearray(out_len)
    i = o = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:                                  # literal run of ctrl + 1 bytes
            ln = ctrl + 1
            out[o:o + ln] = data[i:i + ln]
            i += ln
            o += ln
        else:                                          # back reference
            ln = ctrl >> 5
            ref = o - ((ctrl & 0x1F) << 8) - 1
            if ln == 7:
                ln += data[i]
                i += 1
            ref -= data[i]
            i += 1
            ln += 2
            if ref < 0:
                raise ValueError("bad LZF back reference")
            for k in range(ln):                        # may overlap itself
                out[o + k] = out[ref + k]
            o += ln
    if o != out_len:
        raise ValueError("LZF: %d bytes, header says %d" % (o, out_len))
    return bytes(out)


def read_pcd_xyz(path):
    """(N, 3) float64 x, y, z of a PCD file."""
    with open(path, "rb") as f:
        raw = f.read()
    header, pos = {}, 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0])
    kind = header["DATA"][0].lower()
    dtypes = [np.dtype({"F": "f", "I": "i", "U": "u"}[t] + str(s)).newbyteorder("<") for t, s in zip(types, sizes)]
    want = [fields.index(c) for c in ("x", "y", "z")]
    if kind == "ascii":
        rows = np.loadtxt(raw[pos:].decode("ascii").splitlines(), ndmin=2)
        starts = np.cumsum([0] + counts[:-1])
        return rows[:, [starts[i] for i in want]].astype(np.float64)
    if kind == "binary":
        dt = np.dtype([("f%d" % i, dtypes[i], (counts[i],)) for i in range(len(fields))])
        rec = np.frombuffer(raw, dt, count=n, offset=pos)
        return np.column_stack([rec["f%d" % i][:, 0] for i in want]).astype(np.float64)
    if kind == "binary_compressed":
        csize, usize = struct.unpack_from("<II", raw, pos)
        data = lzf_decompress(raw[pos + 8:pos + 8 + csize], usize)
        cols, off = [], 0                                # column-major: every x, then every y, ...
        for i in range(len(fields)):
            nbytes = n * sizes[i] * counts[i]
            cols.append(np.frombuffer(data, dtypes[i], count=n * counts[i], offset=off).reshape(n, counts[i])[:, 0])
            off += nbytes
        return np.column_stack([cols[i] for i in want]).astype(np.float64)
    raise ValueError("%s: DATA %s" % (path, kind))


def ground_tilt(xyz, cell=0.5):
    """(tilt [deg], normal, cells): plane through the lowest point of each cell."""
    xyz = xyz[np.isfinite(xyz).all(1)]
    k = np.floor(xyz[:, :2] / cell).astype(np.int64)
    order = np.lexsort((xyz[:, 2], k[:, 1], k[:, 0]))
    k, xyz = k[order], xyz[order]
    first = np.ones(len(k), bool)
    first[1:] = (np.diff(k, axis=0) != 0).any(1)
    low = xyz[first]
    _, _, vt = np.linalg.svd(low - low.mean(0), full_matrices=False)
    normal = vt[2] if vt[2][2] >= 0 else -vt[2]
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2]))))), normal, len(low)


def map_files(path):
    if os.path.isfile(path):
        return [path]
    g = os.path.join(path, "3d", "global.pcd")
    if os.path.isfile(g):
        return [g]
    tiles = sorted(glob.glob(os.path.join(path, "3d", "*.pcd")))
    if not tiles:
        raise SystemExit("%s: no 3d/global.pcd and no 3d/*.pcd" % path)
    return tiles


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="map directory (with 3d/) or one .pcd")
    ap.add_argument("--max-tilt", type=float, default=5.0, help="[deg] above this the map is reported as tilted")
    args = ap.parse_args(argv)
    files = map_files(args.map)
    xyz = np.vstack([read_pcd_xyz(f) for f in files])
    tilt, normal, cells = ground_tilt(xyz)
    lo, hi = xyz.min(0), xyz.max(0)
    print("%s: %d points from %d file(s); x %.1f .. %.1f, y %.1f .. %.1f, z %.1f .. %.1f m" % (
        args.map, len(xyz), len(files), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    print("ground plane (lowest point of %d cells of 0.5 m): normal %s, tilt %.1f deg" % (
        cells, np.round(normal, 3).tolist(), tilt))
    if tilt > args.max_tilt:
        print("TILTED: the map is %.1f deg off level (limit %.1f); a level robot cannot localize against it" % (
            tilt, args.max_tilt))
        return 1
    print("level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
