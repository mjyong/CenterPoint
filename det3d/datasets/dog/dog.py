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
    sequence    recording id (used for splitting)
    timestamp   seconds

Produced by ``tools/dog/export_dataset.py`` (from auto-labels or corrected labels).
"""
import pickle

import numpy as np

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class DogDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, dt

    def __init__(self, info_path, root_path, nsweeps=5, cfg=None, pipeline=None, class_names=None,
                 test_mode=False, load_interval=1, class_balanced=True, **kwargs):
        self.load_interval = load_interval
        self.class_balanced = class_balanced
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
        per_cls = {n: [i for i in infos if n in set(i["gt_names"])] for n in self._class_names}
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
            gts.append(dict(boxes=gb, velocities=gv, labels=glab))
        res = evaluate_detections(preds, gts, names)
        flat = {}
        for k, v in res.items():
            if isinstance(v, dict):
                flat[k] = ", ".join("%s=%.3f" % (kk, vv) for kk, vv in v.items() if isinstance(vv, float))
            else:
                flat[k] = "%.4f" % v
        return {"results": flat, "raw": res}, None
