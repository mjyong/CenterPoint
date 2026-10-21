"""Fine-tuning plumbing: sim frames -> DogDataset -> GT database -> det3d training step."""
import os
import pickle

import numpy as np

from dog_perception.geometry import inv_T

from .conftest import requires_det3d


def _labels_from_sim(sim_frames, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    labels = {"sequence": "sim", "frames": []}
    for fr, info in zip(sim_frames["frames"], sim_frames["infos"]):
        np.save(os.path.join(frames_dir, "%.6f.npy" % fr.stamp), fr.points)
        T = inv_T(fr.T_world_det)
        dyaw = np.arctan2(T[1, 0], T[0, 0])
        objs = []
        for g in info["gt"]:
            if g["num_points"] < 5:
                continue
            c = T[:3, :3] @ g["center"] + T[:3, 3]
            objs.append(dict(track_id=g["id"], label=g["label"], box=np.array([*c, *g["size"], g["yaw"] + dyaw]),
                             velocity=T[:2, :2] @ g["velocity"], score=1.0, review=False, interpolated=False))
        labels["frames"].append(dict(stamp=fr.stamp, T_world_det=fr.T_world_det, objects=objs))
    return labels


@requires_det3d
def test_dog_dataset_gt_sampling_and_training_step(sim_frames, tmp_path):
    import torch
    from det3d.datasets import build_dataset
    from det3d.datasets.utils.create_gt_database import create_groundtruth_database
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis.train import example_to_device, parse_second_losses
    from det3d.torchie.parallel import collate_kitti
    from tools.dog.export_dataset import export_recording

    root = str(tmp_path / "dog")
    labels = _labels_from_sim(sim_frames, str(tmp_path / "run" / "frames"))
    infos = export_recording(str(tmp_path / "run" / "frames"), labels, root, "sim")
    assert len(infos) == 6 and sum(len(i["gt_names"]) for i in infos) > 10
    # head classes: the sim labels carry no vehicle subclass, so vehicles get one from their size
    names = set(np.concatenate([i["gt_names"] for i in infos]))
    assert names <= {"car", "truck", "bus", "pedestrian", "cyclist"} and "car" in names
    for split in ("train", "val"):
        with open(os.path.join(root, "infos_%s.pkl" % split), "wb") as f:
            pickle.dump(infos, f)
    create_groundtruth_database("DOG", root, os.path.join(root, "infos_train.pkl"), nsweeps=5)
    db = os.path.join(root, "dbinfos_train_5sweeps_withvelo.pkl")
    assert os.path.exists(db)

    cfg = Config.fromfile("configs/dog/dog_centerpoint_pp_xt32.py")
    for split in ("train", "val"):
        d = cfg.data[split]
        d.root_path, d.info_path, d.ann_file = root, os.path.join(root, "infos_%s.pkl" % split), None
    pre = next(p for p in cfg.data.train.pipeline if p.type == "Preprocess")
    pre.cfg.db_sampler.db_info_path = db
    pre.cfg.db_sampler.db_prep_steps[0].filter_by_min_num_points = dict(
        car=1, truck=1, bus=1, pedestrian=1, cyclist=1)       # no bus in the sim: sampler skips it
    np.random.seed(0)
    ds = build_dataset(cfg.data.train)
    batch = collate_kitti([ds[0], ds[1]])
    assert len(batch["hm"]) == 5 and batch["hm"][3].shape[1:] == (1, 102, 102)   # pedestrian task, stride 4

    torch.manual_seed(0)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.train()
    loss, log_vars = parse_second_losses(model(example_to_device(batch, torch.device("cpu")), return_loss=True))
    assert torch.isfinite(loss)
    loss.backward()

    # evaluation hook used by tools/dist_test.py: GT as detections -> perfect AP
    val = build_dataset(cfg.data.val)
    dets = {i["token"]: dict(box3d_lidar=torch.from_numpy(i["gt_boxes"]), scores=torch.ones(len(i["gt_boxes"])),
                             label_preds=torch.tensor([cfg.class_names.index(n) for n in i["gt_names"]]))
            for i in infos}
    res, _ = val.evaluation(dets)
    assert abs(res["raw"]["mAP"] - 1.0) < 1e-6
    # and the merged output classes (what the detector reports), after the cross-head merge
    assert abs(res["merged"]["vehicle"]["mAP"] - 1.0) < 1e-6 and "merged/vehicle" in res["results"]


@requires_det3d
def test_vehicle_only_dataset_with_ignore_boxes(sim_frames, tmp_path):
    """export_dataset.py --classes vehicle: car / truck by size, no pedestrians or cyclists, tracks
    still flagged for review become ignore boxes. Ignore boxes stay out of the GT database, are drawn
    on the heat map without a positive / regression target, and are not scored."""
    import torch
    from det3d.datasets import build_dataset
    from det3d.datasets.utils.create_gt_database import create_groundtruth_database
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis.train import example_to_device, parse_second_losses
    from det3d.torchie.parallel import collate_kitti
    from tools.dog.export_dataset import export_recording

    root = str(tmp_path / "dog")
    labels = _labels_from_sim(sim_frames, str(tmp_path / "run" / "frames"))
    vehicles = sorted({o["track_id"] for f in labels["frames"] for o in f["objects"] if o["label"] == "vehicle"})
    assert len(vehicles) >= 2
    for f in labels["frames"]:
        for o in f["objects"]:
            o["review"] = o["track_id"] == vehicles[0]          # one uncertain vehicle track
    infos = export_recording(str(tmp_path / "run" / "frames"), labels, root, "sim", classes="vehicle")
    names = np.concatenate([i["gt_names"] for i in infos])
    ignore = np.concatenate([i["gt_ignore"] for i in infos])
    assert set(names) <= {"car", "truck"} and ignore.any() and not ignore.all()
    n_veh = sum(o["label"] == "vehicle" for f in labels["frames"] for o in f["objects"])
    assert len(names) == n_veh
    for split in ("train", "val"):
        with open(os.path.join(root, "infos_%s.pkl" % split), "wb") as f:
            pickle.dump(infos, f)
    create_groundtruth_database("DOG", root, os.path.join(root, "infos_train.pkl"), nsweeps=5)
    with open(os.path.join(root, "dbinfos_train_5sweeps_withvelo.pkl"), "rb") as f:
        db = pickle.load(f)
    assert sum(len(v) for v in db.values()) == int((~ignore).sum())

    cfg = Config.fromfile("configs/dog/dog_centerpoint_pp_xt32_veh.py")
    for split in ("train", "val"):
        d = cfg.data[split]
        d.root_path, d.info_path, d.ann_file = root, os.path.join(root, "infos_%s.pkl" % split), None
    pre = next(p for p in cfg.data.train.pipeline if p.type == "Preprocess")
    pre.cfg.db_sampler = None
    pre.cfg.global_rot_noise, pre.cfg.global_scale_noise, pre.cfg.global_translate_std = [0, 0], [1, 1], 0
    cfg.data.train.class_balanced = False
    ds = build_dataset(cfg.data.train)
    np.random.seed(0)
    for k in range(len(infos)):
        ex = ds[k]
        n_ign, n_all = int(infos[k]["gt_ignore"].sum()), len(infos[k]["gt_names"])
        assert sum(int((h == 1.0).sum()) for h in ex["hm"]) == n_all       # every box drawn on the heat map
        assert sum(int(m.sum()) for m in ex["mask"]) == n_all - n_ign      # ignored ones get no target
    batch = collate_kitti([ds[0], ds[1]])
    assert len(batch["hm"]) == 2

    torch.manual_seed(0)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.train()
    loss, _ = parse_second_losses(model(example_to_device(batch, torch.device("cpu")), return_loss=True))
    assert torch.isfinite(loss)

    # scoring: predicting only the boxes that are not ignored is perfect; ignored ones are not counted
    val = build_dataset(cfg.data.val)
    dets = {}
    for i in infos:
        keep = ~i["gt_ignore"]
        dets[i["token"]] = dict(box3d_lidar=torch.from_numpy(i["gt_boxes"][keep]), scores=torch.ones(int(keep.sum())),
                                label_preds=torch.tensor([cfg.class_names.index(n) for n in i["gt_names"][keep]]))
    res, _ = val.evaluation(dets)
    assert abs(res["merged"]["vehicle"]["mAP"] - 1.0) < 1e-6
    assert res["merged"]["vehicle"]["num_gt"] == int((~ignore).sum())


def test_vehicle_training_labels():
    """--classes vehicle: head by size; bus -> truck; an unreviewed teacher 'truck' the size of a car,
    a track still flagged for review and a reviewer's 'ignore' become ignore boxes."""
    from tools.dog.export_dataset import train_label

    def obj(sub, l, h, **kw):
        return dict(dict(label="vehicle", subclass=sub, box=np.array([10, 0, 0, l, 2.0, h, 0.0]), review=False), **kw)

    assert train_label(obj("car", 4.6, 1.6), "vehicle") == ("car", False)
    assert train_label(obj("bus", 12.4, 4.0), "vehicle") == ("truck", False)
    assert train_label(obj("truck", 5.0, 2.6), "vehicle") == ("truck", False)          # high: truck by size
    assert train_label(obj("truck", 2.7, 2.2), "vehicle") == ("car", True)             # sizes disagree
    assert train_label(obj("truck", 2.7, 2.2, reviewed=True), "vehicle") == ("car", False)
    assert train_label(obj("car", 4.6, 1.6, review=True), "vehicle") == ("car", True)
    assert train_label(obj("car", 4.6, 1.6, review=True), "vehicle", "drop") == (None, False)
    assert train_label(obj("car", 4.6, 1.6, review=True), "vehicle", "keep") == ("car", False)
    assert train_label(obj("car", 4.6, 1.6, ignore=True, reviewed=True), "vehicle", "keep") == ("car", True)
    assert train_label(dict(obj(None, 0.6, 1.7), label="pedestrian"), "vehicle") == (None, False)
    # the 5-head layout keeps the teacher's subclass and ignores flagged tracks by default
    assert train_label(obj("bus", 12.4, 4.0, review=True)) == ("bus", True)
    assert train_label(dict(obj(None, 0.6, 1.7), label="pedestrian")) == ("pedestrian", False)


@requires_det3d
def test_class_balanced_sampling_adds_empty_frames(tmp_path):
    """Frames are resampled per class (ignore boxes do not count); empty_frames adds frames without
    any training object, as a fraction of the resampled set."""
    from det3d.datasets.dog.dog import DogDataset

    def info(tok, names, ignore=None):
        return dict(token=tok, gt_names=np.array(names, dtype="<U16"),
                    gt_ignore=np.array(ignore if ignore is not None else [False] * len(names), dtype=bool))

    infos = ([info("c%d" % k, ["car"]) for k in range(30)] + [info("t%d" % k, ["truck"]) for k in range(10)]
             + [info("i%d" % k, ["truck"], [True]) for k in range(10)] + [info("e%d" % k, []) for k in range(40)])
    path = str(tmp_path / "infos.pkl")
    with open(path, "wb") as f:
        pickle.dump(infos, f)
    ds = DogDataset.__new__(DogDataset)
    ds.load_interval, ds.class_balanced, ds.test_mode, ds._class_names = 1, True, False, ["car", "truck"]
    np.random.seed(0)
    for frac, n_empty in ((0.0, 0), (0.25, 10), (2.0, 50)):
        ds.empty_frames = frac
        ds.load_infos(path)
        toks = [i["token"][0] for i in ds._infos]
        assert toks.count("c") == 20 and toks.count("t") == 20          # 30 / 10 frames -> half each
        assert toks.count("i") + toks.count("e") == n_empty              # only-ignored frames count as empty
