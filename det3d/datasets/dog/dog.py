"""Robot-dog dataset (Hesai XT32, multi-sweep, already preprocessed).

Every frame is stored exactly as the online detector sees it: the output of
``dog_perception.preprocess`` (deskewed, self-filtered, gravity aligned,
N sweeps with a dt channel) mapped into the network frame by
``dog_perception.detection.ModelFrame``. Training and deployment therefore
share one preprocessing implementation.

Info file (pickle, list of dicts):
    token       unique frame id
    lidar_path  path relative to root_path of a float32 .npy (N, 5) [x, y, z, intensity, dt]
    gt_boxes    (M, 9) det3d convention [x, y, z, w, l, h, vx, vy, r] in the network frame
    gt_names    (M,) class names
    gt_ignore   (M,) bool, optional: uncertain labels that get no loss and are not scored
    sequence    recording id (used for splitting)
    timestamp   seconds

Produced by ``tools/dog/export_dataset.py`` (from auto-labels or corrected labels).
"""
import os
import pickle

import numpy as np

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class DogDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, dt

    def __init__(self, info_path, root_path, nsweeps=5, cfg=None, pipeline=None, class_names=None,
                 test_mode=False, load_interval=1, class_balanced=True, empty_frames=0.0, **kwargs):
        """``empty_frames``: with ``class_balanced``, add frames without any (non-ignored) training
        object, this fraction of the resampled set, so background-only scenes are seen too.

        ``root_path`` is resolved once, here: data/dog_veh is a symlink, and a dataset exported to a
        new directory with the link moved over does not reach a training that is still running (the
        infos and GT-database infos it holds belong to the old one)."""
        root_path = os.path.realpath(root_path)
        self.load_interval = load_interval
        self.class_balanced = class_balanced
        self.empty_frames = empty_frames
        self.test_mode = test_mode
        self._class_names = class_names
        self.load_infos(info_path)   # before super().__init__, which calls len(self)
        super(DogDataset, self).__init__(root_path, info_path, pipeline, test_mode=test_mode,
                                         class_names=class_names)
        self.nsweeps = nsweeps
        self._num_point_features = DogDataset.NumPointFeatures

    def load_infos(self, info_path):
        with open(info_path, "rb") as f:
            infos = pickle.load(f)[:: self.load_interval]
        if self.test_mode or not self.class_balanced:
            self._infos = infos
            return
        # CBGS-style frame resampling so rare classes (cyclists) are not drowned by pedestrians
        names = [set(np.asarray(i["gt_names"])[~np.asarray(i.get("gt_ignore", np.zeros(len(i["gt_names"]), bool)),
                                                             dtype=bool)]) for i in infos]
        per_cls = {n: [i for i, s in zip(infos, names) if n in s] for n in self._class_names}
        total = sum(len(v) for v in per_cls.values())
        self._infos = []
        if total == 0:
            self._infos = infos
            return
        frac = 1.0 / len(self._class_names)
        for name, lst in per_cls.items():
            if not lst:
                continue
            ratio = frac / (len(lst) / total)
            self._infos += np.random.choice(lst, int(len(lst) * ratio)).tolist()
        empty = [i for i, s in zip(infos, names) if not s & set(self._class_names)]
        n_empty = min(len(empty), int(round(self.empty_frames * len(self._infos))))
        if n_empty:
            self._infos += np.random.choice(empty, n_empty, replace=False).tolist()

    def __len__(self):
        return len(self._infos)

    def get_sensor_data(self, idx):
        info = self._infos[idx]
        res = {
            "lidar": {"type": "lidar", "points": None, "nsweeps": self.nsweeps, "annotations": None},
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": False,
        }
        data, _ = self.pipeline(res, info)
        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    @property
    def ground_truth_annotations(self):
        return [{"token": i["token"], "gt_boxes": i["gt_boxes"], "gt_names": i["gt_names"]} for i in self._infos]

    def evaluation(self, detections, output_dir=None, testset=False):
        from dog_perception.detection.boxes import det3d_to_standard
        from dog_perception.detection.evaluate import evaluate_detections

        names = list(self._class_names)
        preds, gts = [], []
        for info in self._infos:
            det = detections.get(info["token"])
            if det is None:
                continue
            box = det["box3d_lidar"]
            box = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
            score = det["scores"].cpu().numpy() if hasattr(det["scores"], "cpu") else np.asarray(det["scores"])
            label = det["label_preds"].cpu().numpy() if hasattr(det["label_preds"], "cpu") else np.asarray(det["label_preds"])
            pb, pv = det3d_to_standard(box)
            gb, gv = det3d_to_standard(info["gt_boxes"])
            glab = np.array([names.index(n) if n in names else -1 for n in info["gt_names"]], dtype=np.int64)
            preds.append(dict(boxes=pb, velocities=pv, scores=score, labels=label.astype(np.int64)))
            gts.append(dict(boxes=gb, velocities=gv, labels=glab,
                            ignore=np.asarray(info.get("gt_ignore", np.zeros(len(glab), bool)), dtype=bool)))
        res = evaluate_detections(preds, gts, names)
        merged = self._evaluate_merged(preds, gts, names)
        flat = {}
        for prefix, r in (("", res), ("merged/", merged or {})):
            for k, v in r.items():
                if isinstance(v, dict):
                    flat[prefix + k] = ", ".join("%s=%.3f" % (kk, vv) for kk, vv in v.items() if isinstance(vv, float))
                else:
                    flat[prefix + k] = "%.4f" % v
        return {"results": flat, "raw": res, "merged": merged}, None

    @staticmethod
    def _evaluate_merged(preds, gts, names):
        """Heads car / truck / bus: also score what the detector outputs, vehicle / pedestrian / cyclist,
        after the same cross-head merge (``classwise_circle_nms``) the detector wrapper applies."""
        from dog_perception.detection.boxes import CLASSES, SUBCLASS_TO_CLASS, Detections, classwise_circle_nms
        from dog_perception.detection.centerpoint import MERGED_NMS_RADIUS
        from dog_perception.detection.evaluate import evaluate_detections

        coarse = [SUBCLASS_TO_CLASS.get(n, n) for n in names]
        if coarse == names:
            return None
        lut = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in coarse], dtype=np.int64)
        to_coarse = lambda l: np.where(l >= 0, lut[np.clip(l, 0, None)], -1)   # noqa: E731
        mp, mg = [], []
        for p, g in zip(preds, gts):
            lab = to_coarse(p["labels"])
            d = Detections(p["boxes"], p["velocities"], p["scores"], lab).select(lab >= 0)
            d = classwise_circle_nms(d, MERGED_NMS_RADIUS)
            mp.append(dict(boxes=d.boxes, velocities=d.velocities, scores=d.scores, labels=d.labels))
            mg.append(dict(g, labels=to_coarse(g["labels"])))
        return evaluate_detections(mp, mg, list(CLASSES))
