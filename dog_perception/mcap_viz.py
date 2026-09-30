"""Write a Foxglove-readable MCAP: map-frame cloud, detection cubes, ego pose."""
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
        schema = json.dumps({"type": "object"}).encode()
        cloud = self._w.register_schema("foxglove.PointCloud", "jsonschema", schema)
        scene = self._w.register_schema("foxglove.SceneUpdate", "jsonschema", schema)
        pose = self._w.register_schema("foxglove.PoseInFrame", "jsonschema", schema)
        self._cloud = self._w.register_channel("/lidar", "json", cloud)
        self._scene = self._w.register_channel("/detections", "json", scene)
        self._pose = self._w.register_channel("/pose", "json", pose)
        self.frames = 0

    def add(self, stamp, xyz, intensity, detections, T_world_body):
        """xyz (N, 3) and boxes are already in the map / world frame."""
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
        p = T_world_body[:3, 3]
        pose = {"timestamp": ts, "frame_id": "map", "pose": {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "orientation": _quat(T_world_body[:3, :3]),
        }}
        log_t = _ns(stamp)
        payload = (
            (self._cloud, cloud),
            (self._scene, scene),
            (self._pose, pose),
        )
        for channel, msg in payload:
            self._w.add_message(channel, log_t, json.dumps(msg).encode(), log_t)
        self._fh.flush()
        self.frames += 1

    def close(self):
        self._w.finish()
        self._fh.close()
