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
    cfg.data.train.pipeline[2].cfg.db_sampler.db_info_path = db
    cfg.data.train.pipeline[2].cfg.db_sampler.db_prep_steps[0].filter_by_min_num_points = dict(
        vehicle=1, pedestrian=1, cyclist=1)
    np.random.seed(0)
    ds = build_dataset(cfg.data.train)
    batch = collate_kitti([ds[0], ds[1]])
    assert batch["hm"][1].shape[1:] == (1, 102, 102)     # pedestrian task heat-map at stride 4

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
