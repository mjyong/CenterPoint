"""Multi-object tracking metrics (CLEAR MOT + identity metrics), center distance.

Same matching rule as py-motmetrics / the nuScenes tracking benchmark (2 m
center distance): a ground-truth object keeps its previous track if that
track is still within the threshold, the rest is assigned with the Hungarian
method. ``MotAccumulator.update`` returns the per-frame events (matches,
FP, FN, ID switches) so they can be drawn while a clip is replayed; the
summary adds the identity metrics (IDF1), which need the whole sequence.

    MOTA = 1 - (FN + FP + IDSW) / GT        MOTP = mean center error of matches
    IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)  (best one-to-one GT id <-> track id map)
    FRAG = times a GT trajectory is picked up again after being lost
    MT / ML = GT trajectories tracked >= 80 % / <= 20 % of their frames
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


class MotAccumulator:
    def __init__(self, class_names, dist_threshold=2.0):
        self.class_names = list(class_names)
        self.thr = dist_threshold
        self.last = {}                                  # (cls, gt id) -> track id of the last match
        self.counts = defaultdict(lambda: defaultdict(float))
        self.gt_status = defaultdict(list)              # (cls, gt id) -> [tracked?] per frame present
        self.cooc = defaultdict(lambda: defaultdict(int))   # cls -> (gt id, trk id) -> frames within thr
        self.n_gt_obs = defaultdict(int)
        self.n_trk_obs = defaultdict(int)
        self.frames = 0

    def update(self, gt_ids, gt_xy, gt_labels, trk_ids, trk_xy, trk_labels):
        """One frame. Returns events: dict with ``matches`` [(gt id, trk id, dist, cls)],
        ``fp`` [trk id], ``fn`` [gt id], ``idsw`` [(gt id, old trk, new trk)]."""
        ev = dict(matches=[], fp=[], fn=[], idsw=[])
        gt_ids, gt_xy, gt_labels = np.asarray(gt_ids), np.asarray(gt_xy, float).reshape(-1, 2), np.asarray(gt_labels)
        trk_ids, trk_xy, trk_labels = (np.asarray(trk_ids), np.asarray(trk_xy, float).reshape(-1, 2),
                                       np.asarray(trk_labels))
        self.frames += 1
        for c in range(len(self.class_names)):
            gi, ti = np.nonzero(gt_labels == c)[0], np.nonzero(trk_labels == c)[0]
            cnt = self.counts[c]
            cnt["gt"] += len(gi)
            cnt["trk"] += len(ti)
            self.n_gt_obs[c] += len(gi)
            self.n_trk_obs[c] += len(ti)
            D = np.linalg.norm(gt_xy[gi, None, :] - trk_xy[None, ti, :], axis=2) if len(gi) and len(ti) \
                else np.zeros((len(gi), len(ti)))
            ok = D <= self.thr
            for a, b in zip(*np.nonzero(ok)):
                self.cooc[c][(int(gt_ids[gi[a]]), int(trk_ids[ti[b]]))] += 1
            pairs = []
            used_g, used_t = set(), set()
            # 1) keep last frame's correspondences that are still valid
            t_index = {int(trk_ids[t]): k for k, t in enumerate(ti)}
            for a, g in enumerate(gi):
                prev = self.last.get((c, int(gt_ids[g])))
                b = t_index.get(prev) if prev is not None else None
                if b is not None and ok[a, b] and b not in used_t:
                    pairs.append((a, b))
                    used_g.add(a)
                    used_t.add(b)
            # 2) Hungarian on the rest
            ra = [a for a in range(len(gi)) if a not in used_g]
            rb = [b for b in range(len(ti)) if b not in used_t]
            if ra and rb:
                sub = np.where(ok[np.ix_(ra, rb)], D[np.ix_(ra, rb)], 1e6)
                r, k = linear_sum_assignment(sub)
                for x, y in zip(r, k):
                    if sub[x, y] < 1e6:
                        pairs.append((ra[x], rb[y]))
                        used_g.add(ra[x])
                        used_t.add(rb[y])
            for a, b in pairs:
                g, t = int(gt_ids[gi[a]]), int(trk_ids[ti[b]])
                prev = self.last.get((c, g))
                if prev is not None and prev != t:
                    cnt["idsw"] += 1
                    ev["idsw"].append((g, prev, t))
                self.last[(c, g)] = t
                cnt["tp"] += 1
                cnt["dist"] += D[a, b]
                ev["matches"].append((g, t, float(D[a, b]), c))
            for a in range(len(gi)):
                tracked = a in used_g
                self.gt_status[(c, int(gt_ids[gi[a]]))].append(tracked)
                if not tracked:
                    cnt["fn"] += 1
                    ev["fn"].append(int(gt_ids[gi[a]]))
            for b in range(len(ti)):
                if b not in used_t:
                    cnt["fp"] += 1
                    ev["fp"].append(int(trk_ids[ti[b]]))
        return ev

    def summary(self):
        out = {}
        for c, name in enumerate(self.class_names):
            k = self.counts[c]
            n_gt = k["gt"]
            r = dict(num_gt=int(n_gt), num_trk=int(k["trk"]), TP=int(k["tp"]), FP=int(k["fp"]), FN=int(k["fn"]),
                     IDSW=int(k["idsw"]))
            r["MOTA"] = 1.0 - (k["fn"] + k["fp"] + k["idsw"]) / n_gt if n_gt else float("nan")
            r["MOTP"] = k["dist"] / k["tp"] if k["tp"] else float("nan")
            r["recall"] = k["tp"] / n_gt if n_gt else float("nan")
            r["precision"] = k["tp"] / k["trk"] if k["trk"] else float("nan")
            # identity metrics
            pairs = self.cooc[c]
            if pairs:
                gids = sorted({g for g, _ in pairs})
                tids = sorted({t for _, t in pairs})
                gi = {g: i for i, g in enumerate(gids)}
                ti = {t: i for i, t in enumerate(tids)}
                M = np.zeros((len(gids), len(tids)))
                for (g, t), v in pairs.items():
                    M[gi[g], ti[t]] = v
                a, b = linear_sum_assignment(-M)
                idtp = float(M[a, b].sum())
            else:
                idtp = 0.0
            idfn, idfp = self.n_gt_obs[c] - idtp, self.n_trk_obs[c] - idtp
            r["IDF1"] = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else float("nan")
            r["IDTP"] = idtp                            # IDF1 of several sequences: 2 sum(IDTP) / sum(GT + tracks)
            # trajectory-level
            seqs = [s for (cc, _), s in self.gt_status.items() if cc == c]
            frac = np.array([np.mean(s) for s in seqs]) if seqs else np.zeros(0)
            r["num_gt_tracks"] = len(seqs)
            r["MT"] = int((frac >= 0.8).sum())
            r["ML"] = int((frac <= 0.2).sum())
            r["FRAG"] = int(sum(max(0, _runs(s) - 1) for s in seqs))
            out[name] = r
        tot = {key: sum(out[n][key] for n in self.class_names) for key in
               ("num_gt", "num_trk", "TP", "FP", "FN", "IDSW", "MT", "ML", "FRAG", "num_gt_tracks")}
        tot["MOTA"] = 1.0 - (tot["FN"] + tot["FP"] + tot["IDSW"]) / tot["num_gt"] if tot["num_gt"] else float("nan")
        tp_dist = sum(self.counts[c]["dist"] for c in range(len(self.class_names)))
        tot["MOTP"] = tp_dist / tot["TP"] if tot["TP"] else float("nan")
        tot["recall"] = tot["TP"] / tot["num_gt"] if tot["num_gt"] else float("nan")
        tot["precision"] = tot["TP"] / tot["num_trk"] if tot["num_trk"] else float("nan")
        f1 = [out[n]["IDF1"] for n in self.class_names if out[n]["num_gt"]]
        tot["IDF1"] = float(np.mean(f1)) if f1 else float("nan")       # class average
        out["all"] = tot
        return out


def _runs(status):
    """Number of maximal runs of True in a boolean sequence."""
    n, prev = 0, False
    for s in status:
        if s and not prev:
            n += 1
        prev = s
    return n
