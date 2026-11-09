"""Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py: the localization map's ground against the vertical.
The built-in fallback map 0c3f8a2d-... (15 m of the 0921 bag aligned to INS along a nearly straight line) is
tilted about 51 deg, which is why T42's localization matched a few dozen points per scan."""
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT

LIGHTNING = os.path.join(ROOT, "Nav_D20", "slam", "lightning-lm-master")
FALLBACK = os.path.join(LIGHTNING, "map", "0c3f8a2d-41eb-416d-8fca-aaad500e2be9")


def _tool():
    sys.path.insert(0, os.path.join(LIGHTNING, "scripts"))
    import check_map_tilt
    return check_map_tilt


def _scene(tilt_deg, rng):
    """Ground 30 x 30 m with a few walls and boxes, then rotated by tilt_deg about the x axis."""
    g = np.column_stack([rng.uniform(-15, 15, 20000), rng.uniform(-15, 15, 20000), rng.normal(0, 0.02, 20000)])
    wall = np.column_stack([rng.uniform(-15, 15, 4000), np.full(4000, 10.0), rng.uniform(0, 4, 4000)])
    box = np.column_stack([rng.uniform(2, 4, 2000), rng.uniform(-3, -1, 2000), rng.uniform(0, 2, 2000)])
    xyz = np.vstack([g, wall, box])
    a = np.radians(tilt_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return xyz @ rot.T


def _write_pcd(path, xyz, kind):
    head = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            "WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA %s\n" % (len(xyz), len(xyz), kind))
    pts = np.column_stack([xyz, np.ones(len(xyz))]).astype("<f4")
    with open(path, "wb") as f:
        f.write(head.encode())
        if kind == "ascii":
            f.write("\n".join(" ".join("%.4f" % v for v in p) for p in pts).encode() + b"\n")
        else:
            f.write(pts.tobytes())


@pytest.mark.parametrize("tilt, kind", [(0.0, "binary"), (20.0, "ascii")])
def test_check_map_tilt_measures_the_ground(tmp_path, capsys, tilt, kind):
    tool = _tool()
    (tmp_path / "3d").mkdir()
    _write_pcd(str(tmp_path / "3d" / "global.pcd"), _scene(tilt, np.random.default_rng(0)), kind)
    code = tool.main([str(tmp_path)])
    out = capsys.readouterr().out
    measured = float(out.split("tilt ")[1].split(" deg")[0])
    assert abs(measured - tilt) < 1.0, out
    assert code == (1 if tilt > 5 else 0)


def test_the_builtin_fallback_map_is_tilted(capsys):
    if not os.path.isfile(os.path.join(FALLBACK, "3d", "global.pcd")):
        pytest.skip("no Nav_D20 fallback map")
    tool = _tool()
    assert tool.main([FALLBACK]) == 1
    out = capsys.readouterr().out
    assert 45.0 < float(out.split("tilt ")[1].split(" deg")[0]) < 60.0 and "TILTED" in out
    # its keyframes are level in their own frame: the scans are fine, the alignment to INS is not
    kf = tool.read_pcd_xyz(os.path.join(FALLBACK, "session", "keyframes", "0.pcd"))
    assert tool.ground_tilt(kf)[0] < 5.0
