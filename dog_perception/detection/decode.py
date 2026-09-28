"""NumPy CenterHead decoder for exported (ONNX / BPU) models.

Mirrors ``CenterHead.predict`` + circle NMS so the deployed graph can stop
at the raw head maps and everything data-dependent (top-k, NMS) runs on the
ARM cores, which is how BPU/TensorRT deployments are usually split.
"""
import numpy as np

from .boxes import circle_nms

HEAD_ORDER = ("reg", "height", "dim", "rot", "vel", "hm")


def decode_task(maps, pc_range_xy, voxel_size_xy, out_size_factor, score_threshold,
                post_center_range, min_radius_sq, post_max_size=83, pre_max_size=1000):
    """Decode one task head.

    maps: dict of (C, H, W) arrays with keys ``HEAD_ORDER`` (``vel`` optional),
    ``hm`` given as logits. Returns (box9 (K, 9 or 7), scores (K,), labels (K,)).
    """
    hm = 1.0 / (1.0 + np.exp(-maps["hm"]))
    C, H, W = hm.shape
    hm = hm.reshape(C, -1)
    scores = hm.max(axis=0)
    labels = hm.argmax(axis=0)
    cand = np.nonzero(scores > score_threshold)[0]
    if pre_max_size and len(cand) > pre_max_size:
        cand = cand[np.argsort(-scores[cand], kind="stable")[:pre_max_size]]

    ys, xs = np.divmod(cand, W)
    flat = lambda k: maps[k].reshape(maps[k].shape[0], -1)[:, cand]
    reg, hei, dim, rot = flat("reg"), flat("height"), np.exp(flat("dim")), flat("rot")
    x = (xs + reg[0]) * out_size_factor * voxel_size_xy[0] + pc_range_xy[0]
    y = (ys + reg[1]) * out_size_factor * voxel_size_xy[1] + pc_range_xy[1]
    r = np.arctan2(rot[0], rot[1])
    cols = [x, y, hei[0], dim[0], dim[1], dim[2]]
    if "vel" in maps:
        v = flat("vel")
        cols += [v[0], v[1]]
    box = np.stack(cols + [r], axis=1)

    lo, hi = np.asarray(post_center_range[:3]), np.asarray(post_center_range[3:])
    m = (box[:, :3] >= lo).all(1) & (box[:, :3] <= hi).all(1)
    box, sc, lb = box[m], scores[cand][m], labels[cand][m]
    keep = circle_nms(box[:, :2], sc, np.sqrt(min_radius_sq))[:post_max_size]
    return box[keep], sc[keep], lb[keep]


def decode_centerpoint(task_maps, num_classes, test_cfg):
    """task_maps: list (per task) of dicts of (C, H, W) arrays. Returns merged
    (box9, scores, global labels) exactly like ``CenterHead.predict``."""
    boxes, scores, labels = [], [], []
    offset = 0
    for t, maps in enumerate(task_maps):
        b, s, l = decode_task(
            maps, test_cfg["pc_range"], test_cfg["voxel_size"], test_cfg["out_size_factor"],
            test_cfg["score_threshold"], test_cfg["post_center_limit_range"], test_cfg["min_radius"][t],
            test_cfg["nms"]["nms_post_max_size"], test_cfg.get("pre_max_size", 1000))
        boxes.append(b)
        scores.append(s)
        labels.append(l + offset)
        offset += num_classes[t]
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(labels)
