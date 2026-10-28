"""Tier 2: light multi-modal trajectory predictor (~0.2 M params).

    agent history --(GRU | small Transformer)--> a
    neighbour histories (incl. the robot) --same encoder--> n_i
    a attends over {n_i} + a learned null token        --> s   (social context)
    [optional] obstacle raster --tiny CNN-->              m
    MLP([a, s, class, m]) --> h ; K mode queries --> K x Tf x 2 (Laplace mean, scale) + K logits
    [optional, ``state_head``] h --MLP--> the agent's state now: velocity (2) + moving logit

Trained winner-takes-all: Laplace NLL on the mode closest to the ground truth
+ cross-entropy on which mode that was; with the state head also L1 on the velocity and binary
cross-entropy on moving. ``det_feats``: the agent encoder also reads the detections the tracker
matched (features.py ``agent_det``), so the state comes from the measurements, not only from the
tracker's filtered estimates (part of the tracker merged into the network, TRACKING_PREDICTION 3.3.8).
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .features import DET_FEATS, HIST_FEATS


@dataclass
class ModelConfig:
    hist_len: int = 20
    fut_len: int = 6
    num_modes: int = 6
    num_classes: int = 3
    d_model: int = 64
    encoder: str = "gru"       # "gru" | "transformer"
    num_layers: int = 2
    use_raster: bool = False
    raster_size: int = 64
    det_feats: int = 0         # DET_FEATS: the agent encoder also reads ``agent_det``
    state_head: bool = False   # also output the velocity now and a moving logit


class TemporalEncoder(nn.Module):
    def __init__(self, cfg, in_dim):
        super().__init__()
        d = cfg.d_model
        self.kind = cfg.encoder
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))
        if self.kind == "gru":
            self.rnn = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.zeros(1, cfg.hist_len, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, cfg.num_layers)

    def forward(self, x):
        """x: (B, T, F) with the last feature = valid flag. Returns (B, d)."""
        valid = x[..., -1] > 0.5
        h = self.inp(x)
        if self.kind == "gru":
            # invalid steps are the oldest ones (before the track was born) and are
            # zero-filled, so the final hidden state summarises the valid tail
            out, _ = self.rnn(h)
            return out[:, -1]
        pad = ~valid
        pad[:, -1] = False     # keep at least one key per sequence
        out = self.tf(h + self.pos, src_key_padding_mask=pad)
        return out[:, -1]


class TrajectoryPredictor(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        d, C = cfg.d_model, cfg.num_classes
        if cfg.det_feats not in (0, DET_FEATS):
            raise ValueError("det_feats: 0 or %d" % DET_FEATS)
        self.agent_enc = TemporalEncoder(cfg, HIST_FEATS + cfg.det_feats)
        self.nbr_enc = TemporalEncoder(cfg, HIST_FEATS)
        self.nbr_static = nn.Linear(C + 1, d)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.cls_emb = nn.Linear(C, d)
        ctx = 3 * d
        if cfg.use_raster:
            self.map_enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, d, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            ctx += d
        self.fuse = nn.Sequential(nn.Linear(ctx, 2 * d), nn.ReLU(), nn.Linear(2 * d, 2 * d), nn.ReLU())
        self.mode_q = nn.Parameter(torch.randn(cfg.num_modes, d) * 0.1)
        self.traj_head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, cfg.fut_len * 4))
        self.cls_head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        if cfg.state_head:
            self.state = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 3))

    def forward(self, batch):
        cfg = self.cfg
        x = batch["agent_hist"]
        if cfg.det_feats:                                                   # valid flag stays last
            x = torch.cat([batch["agent_det"], x], dim=-1)
        a = self.agent_enc(x)                                               # (B, d)
        B, Nn, T, Fd = batch["nbr_hist"].shape
        n = self.nbr_enc(batch["nbr_hist"].reshape(B * Nn, T, Fd)).reshape(B, Nn, -1)
        n = n + self.nbr_static(batch["nbr_static"])
        keys = torch.cat([self.null_token.expand(B, 1, -1), n], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=n.device), batch["nbr_mask"] < 0.5], dim=1)
        s, _ = self.attn(a[:, None], keys, keys, key_padding_mask=pad)
        feats = [a, s[:, 0], self.cls_emb(batch["agent_class"])]
        if cfg.use_raster:
            feats.append(self.map_enc(batch["raster"]))
        h = self.fuse(torch.cat(feats, dim=-1))                               # (B, 2d)
        q = self.mode_q[None].expand(B, -1, -1)
        z = torch.cat([h[:, None].expand(-1, cfg.num_modes, -1), q], dim=-1)  # (B, K, 3d)
        out = self.traj_head(z).reshape(B, cfg.num_modes, cfg.fut_len, 4)
        res = dict(traj=out[..., :2], log_scale=out[..., 2:].clamp(-4.0, 3.0), logits=self.cls_head(z)[..., 0])
        if cfg.state_head:
            st = self.state(h)
            res.update(vel=st[:, :2], moving_logit=st[:, 2])                  # agent frame, now
        return res


def predictor_loss(out, fut, fut_valid, cls_weight=0.5, vel=None, moving=None, state_valid=None,
                   state_weight=1.0):
    """fut (B, Tf, 2) agent frame; fut_valid (B, Tf). With a state head: ``vel`` (B, 2) agent frame
    and ``moving`` (B,) 0 / 1, the targets now, where ``state_valid`` (B,) (``build_samples``' vel_now,
    moving_now, state_valid)."""
    w = fut_valid[:, None, :]                                               # (B, 1, Tf)
    err = torch.linalg.norm(out["traj"] - fut[:, None], dim=-1)             # (B, K, Tf)
    ade = (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)
    best = ade.argmin(dim=1)
    idx = torch.arange(len(best), device=best.device)
    mu, ls = out["traj"][idx, best], out["log_scale"][idx, best]
    nll = (torch.abs(fut - mu) * torch.exp(-ls) + ls + math.log(2.0)).sum(-1)  # (B, Tf)
    reg = (nll * fut_valid).sum() / fut_valid.sum().clamp(min=1.0)
    cls = F.cross_entropy(out["logits"], best)
    loss, parts = reg + cls_weight * cls, dict(nll=reg.item(), cls=cls.item(), min_ade=ade.min(1)[0].mean().item())
    if "vel" in out and vel is not None:
        w = torch.ones_like(moving) if state_valid is None else state_valid
        n = w.sum().clamp(min=1.0)
        v = (torch.abs(out["vel"] - vel).sum(-1) * w).sum() / n
        m = (F.binary_cross_entropy_with_logits(out["moving_logit"], moving, reduction="none") * w).sum() / n
        loss = loss + state_weight * (v + m)
        parts.update(vel=v.item(), moving=m.item())
    return loss, parts


def save_model(model, path, feature_cfg=None):
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(model.cfg),
                "feature_cfg": asdict(feature_cfg) if feature_cfg is not None else None}, path)


def load_model(path, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(path, map_location=map_location)
    model = TrajectoryPredictor(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("feature_cfg")
