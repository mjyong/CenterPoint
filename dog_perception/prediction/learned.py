"""Tier 2 at runtime, plus training / evaluation helpers.

``LearnedPredictor`` falls back to the IMM rollout for tracks with too little
history, so it can replace tier 1 without gaps.
"""
import time
from collections import defaultdict, deque
from dataclasses import replace

import numpy as np
import torch

from .dataset import detection_history, log_row
from .features import FeatureConfig, agent_to_world, build_inputs
from .kinematic import IMMPredictor, Prediction, merge_modes
from .metrics import displacement_metrics
from .model import ModelConfig, TrajectoryPredictor, load_model, predictor_loss

INPUT_KEYS = ("agent_hist", "agent_class", "nbr_hist", "nbr_static", "nbr_mask", "raster", "agent_det")
MODES = ("trajectory", "velocity", "imm")


def _to_torch(samples, idx, device):
    return {k: torch.from_numpy(np.ascontiguousarray(samples[k][idx])).float().to(device)
            for k in INPUT_KEYS if k in samples}


class LearnedPredictor:
    """Tier 2 at runtime: call it once per frame after the tracker.

    The inputs are cut from its own buffer of the reported tracks, row for row what ``TrackLogger``
    logs and ``build_samples`` cuts, so they match the training windows (the tracker's own history also
    holds the frames before a track is reported).

    ``mode``: what predicts. "trajectory": the network's modes; "velocity": constant velocity from the
    state head's velocity (one mode); "imm": the IMM, the network only estimates the state.
    With a state head, ``state`` holds {track id: (velocity world (2,), P(moving))} of the last call,
    and ``apply_motion`` turns P(moving) into the motion state of the reported tracks.
    """

    def __init__(self, model, feature_cfg=None, device="cpu", fallback=None, min_history=0.5, mode="trajectory",
                 motion_on=0.6, motion_off=0.4, stop_memory=10.0):
        if isinstance(model, str):
            model, fc = load_model(model, map_location=device)
            feature_cfg = feature_cfg or (FeatureConfig(**fc) if fc else None)
        self.model = model.to(device).eval()
        self.fc = feature_cfg or FeatureConfig()
        self.device = device
        self.fallback = fallback or IMMPredictor(horizon=self.fc.fut_len * self.fc.fut_dt, step=self.fc.fut_dt)
        self.min_history = min_history
        self.mode = mode
        self.motion_on, self.motion_off, self.stop_memory = motion_on, motion_off, stop_memory
        self.last_ms = 0.0
        self.reset()

    @property
    def has_state(self):
        return bool(self.model.cfg.state_head)

    def reset(self):
        """Forget the tracks (a new recording)."""
        self._rows = defaultdict(deque)      # track id -> log rows of the last hist_len steps
        self._motion = {}                    # track id -> (motion state, stamp last moving)
        self.state = {}

    def _histories(self, states, stamp):
        horizon = self.fc.hist_dt * (self.fc.hist_len - 1)
        for s in states:
            self._rows[s.track_id].append(log_row(stamp, s))
        for tid in list(self._rows):
            q = self._rows[tid]
            while q and q[0][0] < stamp - horizon - 1e-6:
                q.popleft()
            if not q:                                  # not reported for hist_len steps
                del self._rows[tid]
                self._motion.pop(tid, None)
        hist, dets = {}, {}
        for s in states:
            h = np.asarray(self._rows[s.track_id], dtype=np.float64)
            hist[s.track_id] = np.column_stack([h[:, 0], h[:, 3:7], h[:, 8]])
            dets[s.track_id] = detection_history(h)
        return hist, dets

    @torch.no_grad()
    def __call__(self, tracker, ego_hist=None, obstacles_xy=None):
        if self.mode not in MODES:
            raise ValueError("mode: one of %s, not %r" % (MODES, self.mode))
        if self.mode == "velocity" and not self.has_state:
            raise ValueError("mode 'velocity' needs a model with a state head (train_predictor.py --state-head)")
        t0 = time.perf_counter()
        tracks = tracker.reported_tracks()
        self.state = {}
        if not tracks:
            self._histories([], tracker.stamp if tracker.stamp is not None else 0.0)
            return []
        stamp = tracks[0].stamp
        states = [t.to_state() for t in tracks]
        hist, dets = self._histories(states, stamp)
        labels = {s.track_id: s.label for s in states}
        yaws = {s.track_id: s.yaw for s in states}
        pos = {s.track_id: s.position[:2] for s in states}
        focal = [i for i, h in hist.items() if h[-1, 0] - h[0, 0] >= self.min_history - 1e-6]
        preds, done = [], set()
        if focal:
            inp, meta = build_inputs(hist, labels, yaws, focal, stamp, self.fc, ego_hist, obstacles_xy, dets)
            batch = {k: torch.from_numpy(v).to(self.device) for k, v in inp.items()}
            out = self.model(batch)
            traj = out["traj"].cpu().numpy()
            scale = np.exp(out["log_scale"].cpu().numpy())
            probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
            if self.has_state:
                vel = out["vel"].cpu().numpy()
                p_moving = torch.sigmoid(out["moving_logit"]).cpu().numpy()
            times = self.fc.fut_dt * np.arange(1, self.fc.fut_len + 1)
            for b, tid in enumerate(focal):
                o, th = meta["origins"][b], meta["thetas"][b]
                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]])
                if self.has_state:
                    self.state[tid] = (R @ vel[b], float(p_moving[b]))
                if self.mode == "imm":
                    continue
                # Laplace scale b -> variance 2 b^2, rotated into the world frame
                covs = np.einsum("ij,ktj,jl->ktil", R, 2 * scale[b] ** 2, R.T)
                if self.mode == "trajectory":
                    preds.append(Prediction(tid, labels[tid], stamp, times, agent_to_world(traj[b], o, th), probs[b],
                                            covs, "learned"))
                else:
                    top = int(np.argmax(probs[b]))
                    path = pos[tid] + np.outer(times, self.state[tid][0])
                    preds.append(Prediction(tid, labels[tid], stamp, times, path[None], np.ones(1), covs[top][None],
                                            "learned-velocity"))
                done.add(tid)
        for t in tracks:
            if t.id not in done:
                preds.append(self.fallback.predict_imm(t.imm, t.id, t.label, t.stamp))
        self.last_ms = 1e3 * (time.perf_counter() - t0)
        return preds

    def apply_motion(self, track_states):
        """The reported tracks with the state head's motion state: moving from P(moving) >=
        ``motion_on``, back to stopped below ``motion_off``, static ``stop_memory`` s after the last
        motion. A track never estimated (too little history) keeps the tracker's state; one without an
        estimate this frame, or coasting, keeps its last state, as in the tracker."""
        out = []
        for s in track_states:
            st, prev = self.state.get(s.track_id), self._motion.get(s.track_id)
            if st is None or (s.coasting and prev is not None):
                out.append(s if prev is None else replace(s, motion=prev[0]))
                continue
            state, last = prev if prev is not None else (s.motion, s.stamp if s.motion == "moving" else None)
            if st[1] >= self.motion_on or (state == "moving" and st[1] > self.motion_off):
                state, last = "moving", s.stamp
            elif state in ("moving", "stopped") and last is not None and s.stamp - last < self.stop_memory:
                state = "stopped"
            else:
                state = "static"
            self._motion[s.track_id] = (state, last)
            out.append(replace(s, motion=state))
        return out


def train_predictor(train, val=None, model_cfg=None, epochs=30, batch_size=256, lr=2e-3, device="cpu",
                    seed=0, log_every=5, verbose=True):
    """train / val: sample dicts from ``build_samples``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fc = model_cfg or ModelConfig(hist_len=train["agent_hist"].shape[1], fut_len=train["future"].shape[1],
                                  use_raster="raster" in train)
    state = fc.state_head and "vel_now" in train
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
            st = {k: torch.from_numpy(train[key][idx]).float().to(device)
                  for k, key in (("vel", "vel_now"), ("moving", "moving_now"), ("state_valid", "state_valid"))} \
                if state else {}
            loss, _ = predictor_loss(model(batch), fut, fv, **st)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        rec = {"epoch": ep, "train_loss": tot / n}
        if val is not None and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            rec.update(evaluate_learned(model, val, device))
            if state and "vel_now" in val:
                rec.update({"state_" + k: v for k, v in evaluate_state(model, val, device)["learned"].items()})
        history.append(rec)
        if verbose and ("minADE" in rec):
            print("epoch %d loss %.3f | val minADE %.3f minFDE %.3f FDE1 %.3f MR %.3f"
                  % (ep, rec["train_loss"], rec["minADE"], rec["minFDE"], rec["FDE1"], rec["MR"])
                  + (" | vel err %.3f moving recall %.3f false moving %.4f" % (
                      rec["state_vel_err"], rec["state_moving_recall"], rec["state_false_moving"])
                     if "state_vel_err" in rec else ""))
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


@torch.no_grad()
def evaluate_state(model, samples, device="cpu", batch_size=1024, threshold=0.5):
    """The state now against the targets (``vel_now``, ``moving_now``, where ``state_valid``), for the
    state head (``learned``) and the tracker (``tracker``: its velocity, the last step of the agent
    history, and its logged motion state, when the log has it). Per estimator: ``vel_err`` (mean
    velocity error, m/s), ``moving_recall`` (moving windows called moving), ``false_moving`` (static
    windows called moving), ``num``."""
    ok = samples["state_valid"] > 0.5
    gt_v, gt_m = samples["vel_now"][ok], samples["moving_now"][ok] > 0.5

    def score(v, m):
        return dict(vel_err=float(np.hypot(*(v - gt_v).T).mean()) if len(gt_v) else float("nan"),
                    moving_recall=float(m[gt_m].mean()) if gt_m.any() else float("nan"),
                    false_moving=float(m[~gt_m].mean()) if (~gt_m).any() else float("nan"), num=int(ok.sum()))

    out = {}
    if model is not None and model.cfg.state_head:
        model.eval()
        vel, p = [], []
        for i in range(0, len(samples["future"]), batch_size):
            r = model(_to_torch(samples, np.arange(i, min(i + batch_size, len(samples["future"]))), device))
            vel.append(r["vel"].cpu().numpy())
            p.append(torch.sigmoid(r["moving_logit"]).cpu().numpy())
        out["learned"] = score(np.concatenate(vel)[ok], np.concatenate(p)[ok] >= threshold)
    motion = samples.get("tracker_motion")
    logged = motion is not None and (motion >= 0).all()          # -1: a log without the motion state
    out["tracker"] = score(samples["agent_hist"][ok, -1, 2:4], motion[ok] == 1 if logged else np.zeros(int(ok.sum()), bool))
    if not logged:
        out["tracker"].update(moving_recall=float("nan"), false_moving=float("nan"))
    return out


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
        r = imm.rollout(T * fut_dt, fut_dt, mixed=False)
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
