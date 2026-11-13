"""Nav_D20/planning/tools/bag_on_map.py: a made-up db3 bag (INSPVAX along a road that crosses a crosswalk,
perception's objects) on a made-up route-network map."""
import math
import os
import sys

import numpy as np
import pytest

from .conftest import ROOT
from .test_planning_mcap import _make

TOOLS = os.path.join(ROOT, "Nav_D20", "planning", "tools")
LAT0, LON0, H0 = 29.731405, 106.783891, 217.2        # the repository's lightning config (system.ins_lat ...)
R = 6378137.0


def _lla(x, y):
    """WGS84 latitude, longitude at east x, north y of the origin (height H0): inverts loc_viz.lla_to_enu, the
    conversion the tool uses, to well under a millimetre."""
    from loc_viz import lla_to_enu
    lat, lon = LAT0 + math.degrees(y / R), LON0 + math.degrees(x / (R * math.cos(math.radians(LAT0))))
    for _ in range(5):
        e, n, _u = lla_to_enu(lat, lon, H0, (LAT0, LON0, H0))
        lat += math.degrees((y - n) / R)
        lon += math.degrees((x - e) / (R * math.cos(math.radians(LAT0))))
    return lat, lon


def _map(tmp_path, with_crosswalk=True):
    """40 x 40 m at 0.1 m, origin (80, 180): a 2 m road along x = 100 (white), the rest black; a crosswalk
    across the road at y = 200..204."""
    folder = tmp_path / "site" / "2d" / "1"
    folder.mkdir(parents=True)
    w = h = 400
    ox, oy = 80.0, 180.0
    img = np.zeros((h, w), np.uint8)
    xs = ox + (np.arange(w) + 0.5) * 0.1
    img[:, np.abs(xs - 100.0) < 1.0] = 255
    with open(folder / "map.pgm", "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h) + img.tobytes())
    (folder / "map.yaml").write_text("image: map.pgm\nmode: trinary\nresolution: 0.1\norigin: [%.1f, %.1f, 0]\n"
                                     "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n" % (ox, oy))
    if with_crosswalk:
        corners = [(97.0, 200.0), (103.0, 200.0), (103.0, 204.0), (97.0, 204.0)]
        nodes = "".join('<node id="%d" lat="0" lon="0"><tag k="local_x" v="%.3f"/><tag k="local_y" v="%.3f"/>'
                        '</node>' % (i + 1, x, y) for i, (x, y) in enumerate(corners))
        way = '<way id="10">' + "".join('<nd ref="%d"/>' % (i + 1) for i in range(4)) + '<nd ref="1"/></way>'
        rel = ('<relation id="20"><member type="way" ref="10" role="outer"/><tag k="type" v="multipolygon"/>'
               '<tag k="subtype" v="crosswalk"/></relation>')
        (folder / "semantic.osm").write_text('<?xml version="1.0"?><osm version="0.6">%s%s%s</osm>' % (
            nodes, way, rel))
    return str(tmp_path / "site")


def _bag(path, ts, objects=True, loop=False):
    """The dog walks north along the road from y = 185 to y = 215 (1 m/s, INSPVAX at 100 Hz) and then 10 m off
    it; a moving vehicle in every third object message, 11.2 m away."""
    from rosbags.rosbag2 import Writer
    pts = np.vstack([np.column_stack([np.full(300, 100.0), np.linspace(185.0, 215.0, 300)]),
                     np.column_stack([np.linspace(100.0, 110.0, 100), np.full(100, 215.0)])])
    if loop:            # up the road and back to where it started (1112: a loop)
        pts = np.column_stack([np.full(400, 100.0), np.concatenate([np.linspace(185.0, 197.0, 200),
                                                                     np.linspace(197.0, 185.0, 200)])])
    with Writer(path, version=8) as w:
        ins = w.add_connection("/rte/rdap/rtk/inspvax", "kygi665_msgs/msg/INSPVAX", typestore=ts)
        obj = w.add_connection("/perception/lidar_3d/objects", "perception_interface/msg/LidarObjectArray",
                               typestore=ts) if objects else None
        for i, (x, y) in enumerate(pts):
            la, lo = _lla(x, y)
            for k in range(10):
                t = int(round((1.7e9 + i * 0.1 + k * 0.01) * 1e9))
                m = _make(ts, "kygi665_msgs/msg/INSPVAX", latitude=la, longitude=lo, height=H0,
                          ins_status=_make(ts, "kygi665_msgs/msg/InertialSolutionStatus", data=3),
                          pos_type=_make(ts, "kygi665_msgs/msg/PositionOrVelocityType", data=56))
                w.write(ins, t, ts.serialize_cdr(m, "kygi665_msgs/msg/INSPVAX"))
            if obj is not None:
                car = _make(ts, "perception_interface/msg/LidarObject", label=0, motion=1,
                            position=_make(ts, "geometry_msgs/msg/Point", x=10.0, y=5.0, z=0.0))
                a = _make(ts, "perception_interface/msg/LidarObjectArray", objects=[car] if i % 3 == 0 else [])
                w.write(obj, int(round((1.7e9 + i * 0.1) * 1e9)),
                        ts.serialize_cdr(a, "perception_interface/msg/LidarObjectArray"))


@pytest.fixture
def tool():
    pytest.importorskip("rosbags")
    sys.path.insert(0, TOOLS)
    try:
        import bag_on_map
        yield bag_on_map
    finally:
        sys.path.remove(TOOLS)


def test_polygon_distance_is_to_the_edges(tool):
    poly = np.array([[0.0, 0.0], [32.0, 0.0], [32.0, 4.0], [0.0, 4.0]])      # a 32 m crosswalk
    d = tool.polygon_distance(np.array([[16.0, -2.0], [16.0, 2.0], [-3.0, -4.0]]), poly)
    assert np.allclose(d, [2.0, 2.0, 5.0])


def test_bag_on_map_finds_the_crosswalk_the_track_crosses_and_a_goal_past_it(tmp_path, capsys, tool):
    ts = tool.typestore()
    _bag(tmp_path / "bag", ts)
    assert tool.main(["--bag", str(tmp_path / "bag"), "--map", _map(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "4000 INSPVAX (100% good, 100% RTK fixed)" in out
    assert "start (100.0, 185.0)" in out
    assert "end (110.0, 215.0)" in out and "40 m long" in out
    # 300 of the 400 points on the road (2 m wide), and the first 1.5 m of the turn off it (half the road + 0.5 m)
    assert "map site:2d/1/map.pgm: 100% of the track inside the map, 79% within 0.5 m of a route cell" in out
    assert "crosswalk 20 (centre 100.0, 202.0): track CROSSES it (15..19 m along), closest to its edge 0.0 m" in out
    goal = next(l for l in out.splitlines() if "goal for replay_planning_check.sh" in l)
    gx, gy = [float(v) for v in goal.split(": ")[1].split()[:2]]
    assert abs(gx - 100.0) < 0.2 and 205.0 < gy < 215.5, goal
    assert "past crosswalk 20" in goal
    assert "vehicles per message: max 1, frames with a vehicle 34%, with a moving one 34%" in out
    assert "nearest vehicle: median 11.2 m" in out


def test_bag_on_map_without_objects_or_crosswalks(tmp_path, capsys, tool):
    ts = tool.typestore()
    _bag(tmp_path / "bag", ts, objects=False)
    assert tool.main(["--bag", str(tmp_path / "bag"), "--map", _map(tmp_path, with_crosswalk=False)]) == 0
    out = capsys.readouterr().out
    assert "not in the bag (the yield gate will see no vehicles" in out
    assert "no crosswalks in this map" in out
    goal = next(l for l in out.splitlines() if "goal for replay_planning_check.sh" in l)
    assert "past crosswalk" not in goal



def test_bag_on_map_goal_of_a_loop_is_its_far_end(tmp_path, capsys, tool):
    ts = tool.typestore()
    _bag(tmp_path / "bag", ts, objects=False, loop=True)
    assert tool.main(["--bag", str(tmp_path / "bag"), "--map", _map(tmp_path, with_crosswalk=False)]) == 0
    goal = next(l for l in capsys.readouterr().out.splitlines() if "goal for replay_planning_check.sh" in l)
    gx, gy = [float(v) for v in goal.split(": ")[1].split()[:2]]
    assert abs(gx - 100.0) < 0.2 and abs(gy - 197.0) < 0.5, goal          # not back at the start (185)
    assert "12 m from its start" in goal
