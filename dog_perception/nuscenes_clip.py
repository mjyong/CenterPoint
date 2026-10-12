"""nuScenes scenes as a frame stream for the dog pipeline, plus ground truth.

Only the JSON tables and the sensor files are read (no nuscenes-devkit, no
maps). A frame is one LIDAR_TOP sweep:

* ``points``: this sweep + the 9 before it (det3d's nuScenes convention:
  sweeps moved into this sweep's lidar frame, close points removed from the
  older ones, time lag as 5th channel, padded by repetition at the start of a
  scene), expressed in the "D" frame the detector wrapper expects. D is the
  inverse of ``ModelFrame``, so the network sees exactly LIDAR_TOP, the frame
  it was trained in, and everything downstream (filters, tracker, predictor)
  runs unchanged;
* ``T_global_D`` / ``T_global_ego``: poses in the nuScenes global frame, which
  is the tracker's world frame;
* keyframes (2 Hz) carry the sample token, hence ground truth and camera images.

``hz=10`` takes every other 20 Hz sweep (keyframes always included), which is
the rate the tracker is tuned for; ``hz=2`` takes keyframes only.

Ground truth is mapped to the three dog classes the same way the detector's
output is (``NUSC_TO_DOG``); barrier / traffic cone / animal etc. get label -1.
"""
import json
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .detection.boxes import NUSC_TO_DOG, class_index
from .detection.frame_adapter import ModelFrame
from .geometry import inv_T, make_T, wrap_angle

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
# nuScenes detection-eval ranges of the merged classes (car / pedestrian / bicycle)
EVAL_RANGE = {"vehicle": 50.0, "pedestrian": 40.0, "cyclist": 40.0}
_TABLES = ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "sample_annotation",
           "instance", "category", "attribute", "visibility", "log")


def quat_wxyz_to_R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def pose_T(rec):
    return make_T(quat_wxyz_to_R(rec["rotation"]), rec["translation"])


def category_to_class(category_name):
    """nuScenes category -> nuScenes detection name -> dog class name ('' if none)."""
    c = category_name
    det = None
    if c.startswith("human.pedestrian") and c not in ("human.pedestrian.wheelchair", "human.pedestrian.stroller",
                                                       "human.pedestrian.personal_mobility"):
        det = "pedestrian"
    elif c == "vehicle.bicycle":
        det = "bicycle"
    elif c == "vehicle.motorcycle":
        det = "motorcycle"
    elif c in ("vehicle.bus.bendy", "vehicle.bus.rigid"):
        det = "bus"
    elif c == "vehicle.car":
        det = "car"
    elif c == "vehicle.construction":
        det = "construction_vehicle"
    elif c == "vehicle.trailer":
        det = "trailer"
    elif c == "vehicle.truck":
        det = "truck"
    return NUSC_TO_DOG.get(det, "")


class NuscTables:
    """The nuScenes JSON tables, indexed by token."""

    def __init__(self, dataroot, version="v1.0-mini"):
        self.dataroot = dataroot
        self.version = version
        meta = os.path.join(dataroot, version)
        if not os.path.isdir(meta):
            raise FileNotFoundError("no %s tables under %s" % (version, dataroot))
        self.t = {}
        for name in _TABLES:
            with open(os.path.join(meta, name + ".json")) as f:
                self.t[name] = {r["token"]: r for r in json.load(f)}
        self.scene_by_name = {r["name"]: r for r in self.t["scene"].values()}
        self.ann_by_sample = defaultdict(list)
        for a in self.t["sample_annotation"].values():
            self.ann_by_sample[a["sample_token"]].append(a)
        self.channel_of_cs = {tok: self.t["sensor"][cs["sensor_token"]]["channel"]
                              for tok, cs in self.t["calibrated_sensor"].items()}
        self.sd_by_sample = defaultdict(dict)        # sample -> channel -> keyframe sample_data
        for sd in self.t["sample_data"].values():
            if sd["is_key_frame"]:
                self.sd_by_sample[sd["sample_token"]][self.channel_of_cs[sd["calibrated_sensor_token"]]] = sd

    def get(self, table, token):
        return self.t[table][token]

    def path(self, sd):
        return os.path.join(self.dataroot, sd["filename"])

    def scene_names(self, which):
        """Scene names for a list / a split name ('mini_val', 'mini_train', 'val', ...)."""
        if isinstance(which, str):
            try:
                from nuscenes.utils.splits import create_splits_scenes
                splits = create_splits_scenes()
            except ImportError:
                splits = {"mini_val": ["scene-0103", "scene-0916"]}
            names = splits.get(which, which.split(","))
        else:
            names = list(which)
        return [n for n in names if n in self.scene_by_name]

    def box_velocity(self, ann, max_time_diff=1.5):
        """Global xy velocity of an annotation from its neighbours (devkit's rule)."""
        prev = self.t["sample_annotation"].get(ann["prev"]) if ann["prev"] else None
        nxt = self.t["sample_annotation"].get(ann["next"]) if ann["next"] else None
        if prev is None and nxt is None:
            return np.full(2, np.nan)
        first, last = (prev or ann), (nxt or ann)
        t0 = 1e-6 * self.t["sample"][first["sample_token"]]["timestamp"]
        t1 = 1e-6 * self.t["sample"][last["sample_token"]]["timestamp"]
        dt = t1 - t0
        if dt <= 0 or (prev is not None and nxt is not None and dt > 2 * max_time_diff) or \
                ((prev is None or nxt is None) and dt > max_time_diff):
            return np.full(2, np.nan)
        return (np.asarray(last["translation"][:2]) - np.asarray(first["translation"][:2])) / dt


@dataclass
class GroundTruth:
    """Boxes of one keyframe in the global frame (standard ``[x, y, z, l, w, h, yaw]``)."""
    boxes: np.ndarray
    velocities: np.ndarray        # (N, 2), NaN when unknown
    labels: np.ndarray            # dog class index, -1 = not one of the three classes
    ids: np.ndarray               # instance index within the scene (stable track id)
    num_pts: np.ndarray           # lidar points inside the box
    category: list
    attribute: list
    visibility: list


@dataclass
class Frame:
    stamp: float
    index: int
    sd_token: str
    is_key: bool
    sample_token: str
    points: np.ndarray            # (N, 5) D frame [x, y, z, intensity, time_lag]
    n_current: int                # points[:n_current] is this sweep alone
    T_global_D: np.ndarray
    T_global_ego: np.ndarray
    T_global_lidar: np.ndarray
    gt: GroundTruth = None
    cameras: dict = field(default_factory=dict)   # channel -> dict(path, stamp, T_global_cam, K, width, height)


class NuscScene:
    def __init__(self, tables, name, hz=10, nsweeps=10, cameras=("CAM_FRONT",)):
        self.tb = tables
        self.name = name
        self.scene = tables.scene_by_name[name]
        self.nsweeps = nsweeps
        self.cameras = tuple(cameras)
        self.frame = ModelFrame()
        # T_lidar_D: x_lidar = R x_D + [0, 0, z_offset]
        self.T_lidar_D = make_T(self.frame.R, [0.0, 0.0, self.frame.z_offset])
        first = tables.get("sample", self.scene["first_sample_token"])
        sd = tables.sd_by_sample[first["token"]]["LIDAR_TOP"]
        chain = [sd]
        while sd["next"]:
            sd = tables.get("sample_data", sd["next"])
            chain.append(sd)
        self.lidar_chain = chain
        self.selected = self._select(chain, hz)
        self._cache = OrderedDict()
        self.instances = self._instances()

    # ----------------------------------------------------------- frames
    @staticmethod
    def _select(chain, hz):
        if hz >= 20:
            return list(chain)
        key_t = [1e-6 * s["timestamp"] for s in chain if s["is_key_frame"]]
        out, last = [], -1e9
        for s in chain:
            t = 1e-6 * s["timestamp"]
            if s["is_key_frame"]:
                out.append(s)
                last = t
            elif hz > 2:
                nxt = [k for k in key_t if k > t]
                if t - last >= 0.75 / hz and (not nxt or nxt[0] - t >= 0.75 / hz):
                    out.append(s)
                    last = t
        return out

    def __len__(self):
        return len(self.selected)

    def _load(self, sd):
        tok = sd["token"]
        if tok not in self._cache:
            pts = np.fromfile(self.tb.path(sd), dtype=np.float32).reshape(-1, 5)[:, :4]
            self._cache[tok] = pts
            if len(self._cache) > 2 * self.nsweeps + 2:
                self._cache.popitem(last=False)
        return self._cache[tok]

    def T_global_lidar(self, sd):
        ego = self.tb.get("ego_pose", sd["ego_pose_token"])
        cs = self.tb.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return pose_T(ego) @ pose_T(cs), pose_T(ego)

    def multisweep(self, sd):
        """(N, 5) [x, y, z, intensity, time_lag] in the lidar frame of ``sd``."""
        T_ref, _ = self.T_global_lidar(sd)
        T_ref_inv = inv_T(T_ref)
        t_ref = 1e-6 * sd["timestamp"]
        clouds = [np.column_stack([self._load(sd), np.zeros(len(self._load(sd)), np.float32)])]
        cur, last = sd, None
        for _ in range(self.nsweeps - 1):
            if cur["prev"]:
                cur = self.tb.get("sample_data", cur["prev"])
                pts = self._load(cur)
                close = (np.abs(pts[:, 0]) < 1.0) & (np.abs(pts[:, 1]) < 1.0)
                pts = pts[~close]
                T, _ = self.T_global_lidar(cur)
                M = T_ref_inv @ T
                xyz = pts[:, :3] @ M[:3, :3].T + M[:3, 3]
                last = np.column_stack([xyz, pts[:, 3], np.full(len(pts), t_ref - 1e-6 * cur["timestamp"])])
            elif last is None:
                last = clouds[0]
            clouds.append(last)
        return np.vstack(clouds).astype(np.float32)

    def frames(self, start=0, stop=None):
        for i, sd in enumerate(self.selected[start:stop], start):
            pts_lidar = self.multisweep(sd)
            pts = self.frame.points_from_model(pts_lidar)
            T_gl, T_ge = self.T_global_lidar(sd)
            f = Frame(stamp=1e-6 * sd["timestamp"], index=i, sd_token=sd["token"], is_key=bool(sd["is_key_frame"]),
                      sample_token=sd["sample_token"] if sd["is_key_frame"] else "", points=pts,
                      n_current=len(self._load(sd)),
                      T_global_D=T_gl @ self.T_lidar_D, T_global_ego=T_ge, T_global_lidar=T_gl)
            if f.is_key:
                f.gt = self.ground_truth(f.sample_token)
                f.cameras = self.camera_data(f.sample_token)
            yield f

    # ----------------------------------------------------------- ground truth
    def _instances(self):
        """instance token -> dict(id, label, category, t (K,), center (K,3), yaw (K,), size (K,3))."""
        out = {}
        s = self.tb.get("sample", self.scene["first_sample_token"])
        while True:
            t = 1e-6 * s["timestamp"]
            for a in self.tb.ann_by_sample[s["token"]]:
                inst = out.get(a["instance_token"])
                if inst is None:
                    cat = self.tb.get("category", self.tb.get("instance", a["instance_token"])["category_token"])["name"]
                    inst = out[a["instance_token"]] = dict(id=len(out) + 1, category=cat,
                                                           label=class_index(category_to_class(cat)),
                                                           t=[], center=[], yaw=[], size=[])
                w, l, h = a["size"]
                inst["t"].append(t)
                inst["center"].append(a["translation"])
                inst["yaw"].append(_yaw(a["rotation"]))
                inst["size"].append([l, w, h])
            if not s["next"]:
                break
            s = self.tb.get("sample", s["next"])
        for inst in out.values():
            for k in ("t", "center", "yaw", "size"):
                inst[k] = np.asarray(inst[k], dtype=np.float64)
        return out

    def ground_truth(self, sample_token):
        anns = self.tb.ann_by_sample[sample_token]
        boxes, vel, labels, ids, npts, cats, attrs, vis = [], [], [], [], [], [], [], []
        for a in anns:
            inst = self.instances[a["instance_token"]]
            w, l, h = a["size"]
            boxes.append([*a["translation"], l, w, h, _yaw(a["rotation"])])
            vel.append(self.tb.box_velocity(a))
            labels.append(inst["label"])
            ids.append(inst["id"])
            npts.append(a["num_lidar_pts"])
            cats.append(inst["category"])
            attrs.append(",".join(self.tb.get("attribute", t)["name"] for t in a["attribute_tokens"]))
            vis.append(self.tb.get("visibility", a["visibility_token"])["level"] if a["visibility_token"] else "")
        n = len(boxes)
        return GroundTruth(np.asarray(boxes, np.float64).reshape(n, 7), np.asarray(vel, np.float64).reshape(n, 2),
                           np.asarray(labels, np.int64), np.asarray(ids, np.int64), np.asarray(npts, np.int64),
                           cats, attrs, vis)

    def instance_path(self, inst_id, t0, t1):
        """Annotated centers of instance ``inst_id`` with t0 <= t <= t1: (K, 3) [t, x, y]."""
        inst = self._by_id().get(inst_id)
        if inst is None:
            return np.zeros((0, 3))
        m = (inst["t"] >= t0 - 1e-6) & (inst["t"] <= t1 + 1e-6)
        return np.column_stack([inst["t"][m], inst["center"][m, :2]])

    def future(self, inst_id, t, times):
        """GT xy at ``t + times`` (linear between annotations), or None if the
        instance is not annotated over the whole horizon."""
        inst = self._by_id().get(inst_id)
        q = t + np.asarray(times)
        if inst is None or inst["t"][0] > t + 1e-3 or inst["t"][-1] < q[-1] - 0.05:
            return None
        tt = inst["t"]
        return np.column_stack([np.interp(q, tt, inst["center"][:, 0]), np.interp(q, tt, inst["center"][:, 1])])

    def interpolated(self, t):
        """GT boxes at an arbitrary time (between two annotations of the same
        instance), for drawing non-keyframes: (boxes (N, 7), labels, ids)."""
        boxes, labels, ids = [], [], []
        for inst in self.instances.values():
            tt = inst["t"]
            if len(tt) < 2 or t < tt[0] or t > tt[-1]:
                continue
            j = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
            a = (t - tt[j]) / max(tt[j + 1] - tt[j], 1e-6)
            c = (1 - a) * inst["center"][j] + a * inst["center"][j + 1]
            yaw = inst["yaw"][j] + a * wrap_angle(inst["yaw"][j + 1] - inst["yaw"][j])
            s = (1 - a) * inst["size"][j] + a * inst["size"][j + 1]
            boxes.append([*c, *s, yaw])
            labels.append(inst["label"])
            ids.append(inst["id"])
        n = len(boxes)
        return np.asarray(boxes, np.float64).reshape(n, 7), np.asarray(labels, np.int64), np.asarray(ids, np.int64)

    def _by_id(self):
        if not hasattr(self, "_id_index"):
            self._id_index = {v["id"]: v for v in self.instances.values()}
        return self._id_index

    # ----------------------------------------------------------- cameras
    def camera_data(self, sample_token):
        out = {}
        sds = self.tb.sd_by_sample[sample_token]
        for ch in self.cameras:
            sd = sds.get(ch)
            if sd is None or not os.path.exists(self.tb.path(sd)):
                continue
            ego = self.tb.get("ego_pose", sd["ego_pose_token"])
            cs = self.tb.get("calibrated_sensor", sd["calibrated_sensor_token"])
            out[ch] = dict(path=self.tb.path(sd), stamp=1e-6 * sd["timestamp"], T_global_cam=pose_T(ego) @ pose_T(cs),
                           T_ego_cam=pose_T(cs), K=np.asarray(cs["camera_intrinsic"], np.float64),
                           width=int(sd.get("width") or 1600), height=int(sd.get("height") or 900))
        return out


def _yaw(q_wxyz):
    R = quat_wxyz_to_R(q_wxyz)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def boxes_to_frame(boxes, T_new_old):
    """Standard boxes into another frame (heading from the rotated x axis)."""
    b = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 7)
    R = T_new_old[:3, :3]
    b[:, :3] = b[:, :3] @ R.T + T_new_old[:3, 3]
    b[:, 6] = wrap_angle(b[:, 6] + np.arctan2(R[1, 0], R[0, 0]))
    return b
