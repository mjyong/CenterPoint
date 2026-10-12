"""Minimal Foxglove MCAP writer (JSON encoding) and message builders.

Messages are plain dicts in the Foxglove schemas (SceneUpdate, PointCloud,
FrameTransforms, CompressedImage, CameraCalibration, ImageAnnotations, Log)
plus free-form JSON topics (e.g. per-frame metrics for the Plot panel). Byte
fields are base64 strings, which Foxglove only decodes when the JSON schema
marks them with ``contentEncoding: base64``; those schemas are spelled out.
"""
import base64
import json

import numpy as np

from .geometry import R_to_quat

_STAMP = {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}
_SCHEMAS = {
    "foxglove.PointCloud": {
        "type": "object",
        "properties": {
            "timestamp": _STAMP, "frame_id": {"type": "string"}, "pose": {"type": "object"},
            "point_stride": {"type": "integer"},
            "fields": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "offset": {"type": "integer"}, "type": {"type": "integer"}}}},
            "data": {"type": "string", "contentEncoding": "base64"},
        },
    },
    "foxglove.CompressedImage": {
        "type": "object",
        "properties": {"timestamp": _STAMP, "frame_id": {"type": "string"},
                       "data": {"type": "string", "contentEncoding": "base64"}, "format": {"type": "string"}},
    },
}
FLOAT32 = 7


def stamp(t):
    sec = int(np.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return {"sec": sec, "nsec": nsec}


def duration(seconds):
    ns = max(0, int(round(float(seconds) * 1e9)))
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


def vec3(p):
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}


def quat_R(R):
    q = R_to_quat(R)
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def quat_yaw(yaw):
    return {"x": 0.0, "y": 0.0, "z": float(np.sin(yaw / 2)), "w": float(np.cos(yaw / 2))}


def pose(p, yaw=0.0, R=None):
    return {"position": vec3(p), "orientation": quat_R(R) if R is not None else quat_yaw(yaw)}


def color(rgba):
    r, g, b = rgba[:3]
    return {"r": float(r), "g": float(g), "b": float(b), "a": float(rgba[3]) if len(rgba) > 3 else 1.0}


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (a,)


# ------------------------------------------------------------------ primitives
def cube(center, size, yaw, rgba):
    return {"pose": pose(center, yaw), "size": vec3(size), "color": color(rgba)}


def line(points, rgba, thickness=0.08, kind="strip", scale_invariant=False):
    t = {"strip": 0, "loop": 1, "list": 2}[kind]
    return {"type": t, "pose": pose((0, 0, 0)), "thickness": float(thickness), "scale_invariant": scale_invariant,
            "points": [vec3(p) for p in np.asarray(points, float)], "color": color(rgba), "colors": [],
            "indices": []}


def text(position, txt, rgba=(1, 1, 1, 1), font_size=12.0, billboard=True):
    return {"pose": pose(position), "billboard": billboard, "font_size": float(font_size), "scale_invariant": True,
            "color": color(rgba), "text": txt}


def arrow(start, vec, rgba, diameter=0.12):
    v = np.asarray(vec, float)
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        return None
    yaw, pitch = np.arctan2(v[1], v[0]), -np.arctan2(v[2] if len(v) > 2 else 0.0, np.hypot(v[0], v[1]))
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("ZY", [yaw, pitch]).as_matrix()
    head = min(0.4 * n, 1.0)
    return {"pose": pose(start, R=R), "shaft_length": n - head, "shaft_diameter": diameter, "head_length": head,
            "head_diameter": 2.5 * diameter, "color": color(rgba)}


def sphere(center, diameter, rgba):
    return {"pose": pose(center), "size": vec3((diameter, diameter, diameter)), "color": color(rgba)}


def entity(eid, t, frame_id="map", lifetime=0.15, metadata=None, **prims):
    e = {"timestamp": stamp(t), "frame_id": frame_id, "id": eid, "lifetime": duration(lifetime),
         "frame_locked": False,
         "metadata": [{"key": str(k), "value": str(v)} for k, v in (metadata or {}).items()]}
    for k in ("arrows", "cubes", "spheres", "cylinders", "lines", "triangles", "texts", "models"):
        e[k] = [p for p in prims.get(k, []) if p is not None]
    return e


def box_corners(box):
    """(8, 3) corners of a standard box [x, y, z, l, w, h, yaw]."""
    x, y, z, l, w, h, yaw = box[:7]
    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([x + c * dx - s * dy, y + s * dx + c * dy, z + dz])


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),
             (0, 5), (1, 4)]          # last two: a cross on the front face shows the heading


def box_wire(box, rgba, thickness=0.06):
    c = box_corners(box)
    return line(np.array([c[i] for e in BOX_EDGES for i in e]), rgba, thickness, "list")


def dashed(points, dash=0.6, gap=0.4):
    """Points for a LINE_LIST that draws ``points`` (polyline) dashed."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.zeros((0, 3))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    out, a = [], 0.0
    while a < s[-1]:
        b = min(a + dash, s[-1])
        out += [_along(p, s, a), _along(p, s, b)]
        a = b + gap
    return np.array(out)


def _along(p, s, d):
    return np.array([np.interp(d, s, p[:, k]) for k in range(p.shape[1])])


def ellipse(center, cov2, n_sigma=1.0, z=0.0, n=24):
    """(n, 3) points of the n-sigma ellipse of a 2x2 covariance."""
    w, V = np.linalg.eigh(np.asarray(cov2, float))
    w = np.sqrt(np.maximum(w, 1e-9)) * n_sigma
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = (V @ (np.vstack([np.cos(a) * w[0], np.sin(a) * w[1]]))).T + np.asarray(center)[:2]
    return np.column_stack([pts, np.full(n, z)])


# ------------------------------------------------------------------ messages
def scene_update(entities, deletions=()):
    return {"deletions": list(deletions), "entities": list(entities)}


def point_cloud(t, frame_id, xyz, **fields):
    """Float32 fields x, y, z and any extra (N,) arrays."""
    cols = [np.asarray(xyz, np.float32).reshape(-1, 3)] + [np.asarray(v, np.float32).reshape(-1, 1)
                                                          for v in fields.values()]
    data = np.concatenate(cols, axis=1).astype("<f4")
    names = ["x", "y", "z"] + list(fields)
    return {"timestamp": stamp(t), "frame_id": frame_id, "pose": pose((0, 0, 0)), "point_stride": 4 * len(names),
            "fields": [{"name": n, "offset": 4 * i, "type": FLOAT32} for i, n in enumerate(names)],
            "data": base64.b64encode(data.tobytes()).decode("ascii")}


def frame_transforms(t, items):
    """items: [(parent, child, T 4x4)]."""
    return {"transforms": [{"timestamp": stamp(t), "parent_frame_id": p, "child_frame_id": c,
                            "translation": vec3(T[:3, 3]), "rotation": quat_R(T[:3, :3])} for p, c, T in items]}


def compressed_image(t, frame_id, data, fmt="jpeg"):
    return {"timestamp": stamp(t), "frame_id": frame_id, "format": fmt,
            "data": base64.b64encode(data).decode("ascii")}


def camera_calibration(t, frame_id, K, width, height):
    K = np.asarray(K, float)
    P = np.hstack([K, np.zeros((3, 1))])
    return {"timestamp": stamp(t), "frame_id": frame_id, "width": int(width), "height": int(height),
            "distortion_model": "", "D": [], "K": K.ravel().tolist(), "R": np.eye(3).ravel().tolist(),
            "P": P.ravel().tolist()}


def image_annotations(t, lines=(), texts=()):
    """lines: [(pts (N, 2) as LINE_LIST pairs, rgba, thickness)], texts: [(xy, str, rgba, font_size)]."""
    pts = [{"timestamp": stamp(t), "type": 4, "points": [{"x": float(x), "y": float(y)} for x, y in p],
            "outline_color": color(c), "outline_colors": [], "fill_color": color((0, 0, 0, 0)),
            "thickness": float(w)} for p, c, w in lines]
    txt = [{"timestamp": stamp(t), "position": {"x": float(xy[0]), "y": float(xy[1])}, "text": s,
            "font_size": float(fs), "text_color": color(c), "background_color": color((0, 0, 0, 0.55))}
           for xy, s, c, fs in texts]
    return {"circles": [], "points": pts, "texts": txt}


def log(t, message, level=2, name="eval"):
    return {"timestamp": stamp(t), "level": level, "message": message, "name": name, "file": "", "line": 0}


class McapWriter:
    """Topics are created on first write; ``schema`` is a Foxglove schema name
    or any other name for free-form JSON (e.g. metrics for the Plot panel)."""

    def __init__(self, path, compression="zstd"):
        from mcap.writer import CompressionType, Writer

        comp = {"zstd": CompressionType.ZSTD, "lz4": CompressionType.LZ4, None: CompressionType.NONE}[compression]
        self._fh = open(path, "wb")
        self._w = Writer(self._fh, compression=comp)
        self._w.start(profile="", library="dog_perception.foxglove")
        self._schemas, self._channels = {}, {}
        self.path = path
        self.messages = 0

    def write(self, topic, schema, t, msg):
        ch = self._channels.get(topic)
        if ch is None:
            sid = self._schemas.get(schema)
            if sid is None:
                body = _SCHEMAS.get(schema, {"type": "object"})
                sid = self._schemas[schema] = self._w.register_schema(schema, "jsonschema",
                                                                      json.dumps(dict(body, title=schema)).encode())
            ch = self._channels[topic] = self._w.register_channel(topic, "json", sid)
        ns = int(round(t * 1e9))
        self._w.add_message(ch, ns, json.dumps(msg, separators=(",", ":")).encode(), ns)
        self.messages += 1

    def close(self):
        self._w.finish()
        self._fh.close()
