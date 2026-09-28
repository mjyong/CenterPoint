"""Tier 2 at runtime, plus training / evaluation helpers.

``LearnedPredictor`` falls back to the IMM rollout for tracks with too little
history, so it can replace tier 1 without gaps.
"""
import time

import numpy as np
import torch

from .features import FeatureConfig, agent_to_world, build_inputs
from .kinematic import IMMPredictor, Prediction, merge_modes
from .metrics import displacement_metrics
from .model import ModelConfig, TrajectoryPredictor, load_model, predictor_loss

INPUT_KEYS = ("agent_hist", "agent_class", "nbr_hist", "nbr_static", "nbr_mask", "raster")


def _to_torch(samples, idx, device):
    return {k: torch.from_numpy(np.ascontiguousarray(samples[k][idx])).float().to(device)
            for k in INPUT_KEYS if k in samples}


class LearnedPredictor:
    def __init__(self, model, feature_cfg=None, device="cpu", fallback=None, min_history=0.5):
        if isinstance(model, str):
            model, fc = load_model(model, map_location=device)
            feature_cfg = feature_cfg or (FeatureConfig(**fc) if fc else None)
        self.model = model.to(device).eval()
        self.fc = feature_cfg or FeatureConfig()
        self.device = device
        self.fallback = fallback or IMMPredictor(horizon=self.fc.fut_len * self.fc.fut_dt, step=self.fc.fut_dt)
        self.min_history = min_history
        self.last_ms = 0.0

    @torch.no_grad()
    def __call__(self, tracker, ego_hist=None, obstacles_xy=None):
        t0 = time.perf_counter()
        tracks = [t for t in tracker.tracks if t.confirmed]
        if not tracks:
            return []
        stamp = tracks[0].stamp
        states = {t.id: t.to_state() for t in tracks}
        hist = {i: s.history for i, s in states.items()}
        labels = {i: s.label for i, s in states.items()}
        yaws = {i: s.yaw for i, s in states.items()}
        focal = [i for i, s in states.items() if len(s.history) and s.history[-1, 0] - s.history[0, 0] >= self.min_history - 1e-6]
        preds = []
        if focal:
            inp, meta = build_inputs(hist, labels, yaws, focal, stamp, self.fc, ego_hist, obstacles_xy)
            batch = {k: torch.from_numpy(v).to(self.device) for k, v in inp.items()}
            out = self.model(batch)
            traj = out["traj"].cpu().numpy()
            scale = np.exp(out["log_scale"].cpu().numpy())
            probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
            times = self.fc.fut_dt * np.arange(1, self.fc.fut_len + 1)
            for b, tid in enumerate(focal):
                o, th = meta["origins"][b], meta["thetas"][b]
                modes = agent_to_world(traj[b], o, th)
                # Laplace scale b -> variance 2 b^2, rotated into the world frame
                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]])
                var = 2 * scale[b] ** 2
                covs = np.einsum("ij,ktj,jl->ktil", R, var, R.T)
                preds.append(Prediction(tid, labels[tid], stamp, times, modes, probs[b], covs, "learned"))
        done = set(focal)
        for t in tracks:
            if t.id not in done:
                preds.append(self.fallback.predict_imm(t.imm, t.id, t.label, t.stamp))
        self.last_ms = 1e3 * (time.perf_counter() - t0)
        return preds


def train_predictor(train, val=None, model_cfg=None, epochs=30, batch_size=256, lr=2e-3, device="cpu",
                    seed=0, log_every=5, verbose=True):
    """train / val: sample dicts from ``build_samples``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fc = model_cfg or ModelConfig(hist_len=train["agent_hist"].shape[1], fut_len=train["future"].shape[1],
                                  use_raster="raster" in train)
    model = TrajectoryPredictor(fc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, int(np.ceil(len(train["future"]) / batch_size))))
    n = len(train["future"])
    history = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = _to_torch(train, idx, device)
            fut = torch.from_numpy(train["future"][idx]).float().to(device)
            fv = torch.from_numpy(train["future_valid"][idx]).float().to(device)
            loss, _ = predictor_loss(model(batch), fut, fv)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        rec = {"epoch": ep, "train_loss": tot / n}
        if val is not None and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            rec.update(evaluate_learned(model, val, device))
        history.append(rec)
        if verbose and ("minADE" in rec):
            print("epoch %d loss %.3f | val minADE %.3f minFDE %.3f FDE1 %.3f MR %.3f"
                  % (ep, rec["train_loss"], rec["minADE"], rec["minFDE"], rec["FDE1"], rec["MR"]))
    return model.eval(), history


@torch.no_grad()
def evaluate_learned(model, samples, device="cpu", batch_size=1024):
    model.eval()
    trajs, probs = [], []
    for i in range(0, len(samples["future"]), batch_size):
        idx = np.arange(i, min(i + batch_size, len(samples["future"])))
        out = model(_to_torch(samples, idx, device))
        trajs.append(out["traj"].cpu().numpy())
        probs.append(torch.softmax(out["logits"], 1).cpu().numpy())
    return displacement_metrics(np.concatenate(trajs), np.concatenate(probs), samples["future"],
                                samples["future_valid"])


def evaluate_kinematic(samples, class_params=None, merge_dist=0.25, hist_dt=0.1, fut_dt=0.5):
    """Tier-1 baselines on the same samples as the learned model.

    ``imm``: the online IMM predictions logged during tracking (if present);
    ``imm_refit``: an IMM re-run over the sample's history (a proxy when the
    log has no online predictions -- it filters already-filtered states, so
    it lags a bit); ``cv``: last tracker velocity extrapolated.
    """
    from ..detection.boxes import CLASSES
    from ..tracking.imm import IMM, build_model
    from ..tracking.tracker import DEFAULT_CLASS_PARAMS

    class_params = class_params or DEFAULT_CLASS_PARAMS
    T = samples["future"].shape[1]
    imm_modes, imm_probs, cv_modes = [], [], []
    K = 0
    for b in range(len(samples["future"])):
        h = samples["agent_hist"][b]
        p = class_params[CLASSES[int(samples["label"][b])]]
        valid = np.nonzero(h[:, 5] > 0.5)[0]
        imm = IMM([build_model(s) for s in p.models], p.stay_prob)
        first = h[valid[0]]
        imm.initialize([first[0], first[1], first[2], first[3], 0.0],
                       np.diag([p.pos_std ** 2] * 2 + [p.vel_std ** 2] * 2 + [p.init_omega_std ** 2]))
        H = np.zeros((4, 5))
        H[:4, :4] = np.eye(4)
        R = np.diag([p.pos_std ** 2] * 2 + [p.vel_std ** 2] * 2)
        for k in valid[1:]:
            imm.predict(hist_dt)
            if h[k, 4] > 0.5:
                imm.update(h[k, :4], H, R)
        r = imm.rollout(T * fut_dt, fut_dt)
        m, _, pr = merge_modes(r["mode_means"][:, :, :2], r["mode_covs"][:, :, :2, :2], r["mode_probs"], merge_dist)
        imm_modes.append(m)
        imm_probs.append(pr)
        K = max(K, len(m))
        last = h[valid[-1]]
        cv_modes.append((last[:2] + last[2:4] * (fut_dt * np.arange(1, T + 1))[:, None])[None])
    pad_m = np.zeros((len(imm_modes), K, T, 2))
    pad_p = np.zeros((len(imm_modes), K))
    for b, (m, pr) in enumerate(zip(imm_modes, imm_probs)):
        pad_m[b, :len(m)], pad_p[b, :len(m)] = m, pr
        pad_m[b, len(m):] = m[np.argmax(pr)]     # duplicate the best mode into empty slots
    gt, fv = samples["future"], samples["future_valid"]
    cv = np.stack(cv_modes)
    res = {"imm_refit": displacement_metrics(pad_m, pad_p, gt, fv),
           "cv": displacement_metrics(cv, np.ones((len(cv), 1)), gt, fv)}
    if "imm_modes" in samples:
        # what tier 1 actually predicted online (logged by TrackLogger)
        ok = samples["imm_probs"].sum(1) > 0
        res["imm"] = displacement_metrics(samples["imm_modes"][ok], samples["imm_probs"][ok], gt[ok], fv[ok])
    else:
        res["imm"] = res["imm_refit"]
    return res
