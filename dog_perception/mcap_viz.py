"""Write a Foxglove-readable MCAP: map-frame cloud, detection cubes, tracks, ego pose."""
import base64
import json

import numpy as np

from .geometry import R_to_quat

_COLORS = {
    0: (0.2, 0.85, 0.3, 0.65),
    1: (1.0, 0.45, 0.1, 0.85),
    2: (0.25, 0.55, 1.0, 0.8),
}
_MAX_POINTS = 20000

# Foxglove only turns the base64 `data` string into a byte buffer when the JSON
# schema says contentEncoding=base64. A bare {"type":"object"} leaves `data` as
# text, and the 3D panel then has no points to draw.
_POINT_CLOUD_SCHEMA = json.dumps({
    "title": "foxglove.PointCloud",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "pose": {"type": "object"},
        "point_stride": {"type": "integer"},
        "fields": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "offset": {"type": "integer"},
            "type": {"type": "integer"}}}},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
}).encode()


def _stamp(t):
    sec = int(np.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return {"sec": sec, "nsec": nsec}


def _ns(t):
    return int(round(t * 1e9))


def _quat(R):
    q = R_to_quat(R)
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def _yaw_quat(yaw):
    return {"x": 0.0, "y": 0.0, "z": float(np.sin(yaw / 2)), "w": float(np.cos(yaw / 2))}


class McapViz:
    def __init__(self, path):
        from mcap.writer import Writer

        self._fh = open(path, "wb")
        self._w = Writer(self._fh)
        self._w.start(profile="", library="dog_perception.mcap_viz")
        other = json.dumps({"type": "object"}).encode()
        cloud = self._w.register_schema("foxglove.PointCloud", "jsonschema", _POINT_CLOUD_SCHEMA)
        scene = self._w.register_schema("foxglove.SceneUpdate", "jsonschema", other)
        pose = self._w.register_schema("foxglove.PoseInFrame", "jsonschema", other)
        self._cloud = self._w.register_channel("/lidar", "json", cloud)
        self._scene = self._w.register_channel("/detections", "json", scene)
        self._tracks = self._w.register_channel("/tracks", "json", scene)
        self._pose = self._w.register_channel("/pose", "json", pose)
        self.frames = 0

    def add(self, stamp, xyz, intensity, detections, T_world_body, tracks=None):
        """xyz (N, 3), boxes and tracks are already in the map / world frame.

        ``detections`` are drawn as given: pass ``dets.above(DEFAULT_SCORE_THRESHOLDS)``,
        not the raw detector output (which keeps scores down to 0.1 for the tracker).
        ``tracks``: TrackState list; confirmed tracks are drawn on /tracks with id,
        class and speed, coasting ones faded."""
        xyz = np.asarray(xyz, np.float32)
        intensity = np.asarray(intensity, np.float32).reshape(-1)
        if len(xyz) > _MAX_POINTS:
            step = int(np.ceil(len(xyz) / _MAX_POINTS))
            xyz = xyz[::step]
            intensity = intensity[::step]
        packed = np.concatenate([xyz, intensity[:, None]], axis=1).astype("<f4").tobytes()
        ts = _stamp(stamp)
        cloud = {
            "timestamp": ts,
            "frame_id": "map",
            "pose": {"position": {"x": 0, "y": 0, "z": 0},
                     "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
            "point_stride": 16,
            "fields": [
                {"name": "x", "offset": 0, "type": 7},
                {"name": "y", "offset": 4, "type": 7},
                {"name": "z", "offset": 8, "type": 7},
                {"name": "intensity", "offset": 12, "type": 7},
            ],
            "data": base64.b64encode(packed).decode("ascii"),
        }
        cubes = []
        for box, score, label in zip(detections.boxes, detections.scores, detections.labels):
            r, g, b, a = _COLORS.get(int(label), (1, 1, 1, 0.5))
            cubes.append({
                "pose": {"position": {"x": float(box[0]), "y": float(box[1]), "z": float(box[2])},
                         "orientation": _yaw_quat(box[6])},
                "size": {"x": float(box[3]), "y": float(box[4]), "z": float(box[5])},
                "color": {"r": r, "g": g, "b": b, "a": a},
            })
        scene = {"deletions": [], "entities": [{
            "timestamp": ts, "frame_id": "map", "id": "detections",
            "lifetime": {"sec": 0, "nsec": 150000000},
            "frame_locked": False, "cubes": cubes,
        }]}
        track_scene = {"deletions": [], "entities": [{
            "timestamp": ts, "frame_id": "map", "id": "tracks",
            "lifetime": {"sec": 0, "nsec": 150000000},
            "frame_locked": False, "cubes": [], "texts": [],
        }]}
        for t in tracks or ():
            r, g, b, _ = _COLORS.get(int(t.label), (1, 1, 1, 0.5))
            pos = {"x": float(t.position[0]), "y": float(t.position[1]), "z": float(t.position[2])}
            track_scene["entities"][0]["cubes"].append({
                "pose": {"position": pos, "orientation": _yaw_quat(t.yaw)},
                "size": {"x": float(t.size[0]), "y": float(t.size[1]), "z": float(t.size[2])},
                "color": {"r": r, "g": g, "b": b, "a": 0.25 if t.coasting else 0.6},
            })
            track_scene["entities"][0]["texts"].append({
                "pose": {"position": dict(pos, z=pos["z"] + float(t.size[2]) / 2 + 0.3),
                         "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                "billboard": True, "font_size": 14, "scale_invariant": True,
                "color": {"r": 1, "g": 1, "b": 1, "a": 1},
                "text": "%s %d %.1fm/s" % (t.name[:3], t.track_id, float(np.hypot(*t.velocity))),
            })
        p = T_world_body[:3, 3]
        pose = {"timestamp": ts, "frame_id": "map", "pose": {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "orientation": _quat(T_world_body[:3, :3]),
        }}
        log_t = _ns(stamp)
        payload = (
            (self._cloud, cloud),
            (self._scene, scene),
            (self._tracks, track_scene),
            (self._pose, pose),
        )
        for channel, msg in payload:
            self._w.add_message(channel, log_t, json.dumps(msg).encode(), log_t)
        self._fh.flush()
        self.frames += 1

    def close(self):
        self._w.finish()
        self._fh.close()
