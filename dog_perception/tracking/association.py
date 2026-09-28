import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_match(cost, max_cost):
    """Globally greedy: repeatedly take the cheapest remaining pair.

    cost: (N_det, M_trk); pairs with cost > max_cost (scalar or (N, M)) are
    never matched. Returns (K, 2) [det_idx, trk_idx].
    """
    if cost.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    valid = cost <= max_cost
    di, ti = np.nonzero(valid)
    order = np.argsort(cost[di, ti], kind="stable")
    used_d, used_t, out = set(), set(), []
    for k in order:
        d, t = di[k], ti[k]
        if d in used_d or t in used_t:
            continue
        used_d.add(d)
        used_t.add(t)
        out.append((d, t))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def hungarian_match(cost, max_cost):
    if cost.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    valid = cost <= max_cost
    big = 1e6
    c = np.where(valid, cost, big)
    r, k = linear_sum_assignment(c)
    keep = valid[r, k]
    return np.stack([r[keep], k[keep]], axis=1).astype(np.int64)


def match(cost, max_cost, method="greedy"):
    return (hungarian_match if method == "hungarian" else greedy_match)(cost, max_cost)
