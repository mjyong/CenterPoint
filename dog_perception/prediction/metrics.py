import numpy as np


def displacement_metrics(pred_modes, probs, gt, valid=None, miss_threshold=2.0):
    """pred_modes (B, K, T, 2), probs (B, K), gt (B, T, 2), valid (B, T).

    Returns minADE/minFDE over K modes, ADE/FDE of the most likely mode and
    miss rate (min final displacement > threshold).
    """
    if valid is None:
        valid = np.ones(gt.shape[:2])
    err = np.linalg.norm(pred_modes - gt[:, None], axis=-1)               # (B, K, T)
    w = valid[:, None, :]
    ade = (err * w).sum(-1) / np.maximum(w.sum(-1), 1)
    last = np.array([np.nonzero(v)[0][-1] if v.any() else 0 for v in valid])
    fde = err[np.arange(len(gt)), :, last]
    top = np.argmax(probs, axis=1)
    idx = np.arange(len(gt))
    return dict(
        minADE=float(ade.min(1).mean()), minFDE=float(fde.min(1).mean()),
        ADE1=float(ade[idx, top].mean()), FDE1=float(fde[idx, top].mean()),
        MR=float((fde.min(1) > miss_threshold).mean()), num=int(len(gt)),
    )
