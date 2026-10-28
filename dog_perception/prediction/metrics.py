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


def along_across(points, start, gt):
    """Split the error of a predicted path against the true one. ``points`` (T, 2): predicted
    positions at the steps of ``gt`` (T, 2), the true positions after ``start`` (2,). The true path is
    the polyline start -> gt, its first and last segments extended. Returns (T,) ``along``: signed
    distance along the path from the true position to the prediction's foot point (+: ahead, the
    predicted speed too high), and (T,) ``across``: distance from the path (its shape: where a map
    would help). NaN when the true path has no length."""
    path = np.vstack([start, gt]).astype(np.float64)
    seg = np.diff(path, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s0 = np.r_[0.0, np.cumsum(L)]
    ok = np.nonzero(L > 1e-6)[0]
    along, across = np.full(len(points), np.nan), np.full(len(points), np.nan)
    if len(ok) == 0:
        return along, across
    for k, p in enumerate(np.asarray(points, np.float64)):
        best = (np.inf, 0.0)
        for j in ok:
            u = (p - path[j]) @ seg[j] / L[j] ** 2
            u = min(max(u, -np.inf if j == ok[0] else 0.0), np.inf if j == ok[-1] else 1.0)
            d = float(np.hypot(*(p - path[j] - u * seg[j])))
            if d < best[0]:
                best = (d, s0[j] + u * L[j])
        across[k], along[k] = best[0], best[1] - s0[k + 1]
    return along, across
