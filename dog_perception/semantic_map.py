"""Crosswalks of the site maps' semantic layer (``<map>/2d/<id>/semantic.osm``, the robot team's VMB export, 2026-11).

Each crosswalk is an OSM relation ``type=multipolygon``, ``subtype=crosswalk`` whose one ``outer`` member is a
closed way of nodes carrying ``local_x`` / ``local_y`` (metres, the map frame of the planning and localization).
Planning reads the same file (``third_party/osm_map``, GlobalMapManager::crosswalks()).

    for cid, polygon in read_crosswalks("ad68517c-.../2d/1780551612/semantic.osm"):
        span = yield_gate.crosswalk_span(path_xy, polygon)
"""
import xml.etree.ElementTree as ET

import numpy as np


def read_crosswalks(osm_path):
    """[(relation id, (N, 2) polygon without the closing point)] in file order."""
    root = ET.parse(osm_path).getroot()
    nodes = {}
    for n in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            nodes[n.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    ways = {w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.findall("way")}
    out = []
    for r in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        if tags.get("type") != "multipolygon" or tags.get("subtype") != "crosswalk":
            continue
        outer = [m for m in r.findall("member") if m.get("type") == "way" and m.get("role") == "outer"]
        if len(outer) != 1 or outer[0].get("ref") not in ways:
            continue
        pts = [nodes[i] for i in ways[outer[0].get("ref")] if i in nodes]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            out.append((r.get("id"), np.asarray(pts, dtype=float)))
    return out


def crosswalk_size(polygon):
    """(length, width) [m] of the polygon's minimum-area rectangle: length is across the road (what the robot
    walks), width along it (the stripes' band, about 5 m on the site maps)."""
    p = np.asarray(polygon, dtype=float)
    best = None
    for i in range(len(p)):
        e = p[(i + 1) % len(p)] - p[i]
        n = np.hypot(*e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        a, b = p @ u, p @ v
        dims = (np.ptp(a), np.ptp(b))
        if best is None or dims[0] * dims[1] < best[0] * best[1]:
            best = dims
    return max(best), min(best)
