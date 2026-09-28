"""Tier 2: light multi-modal trajectory predictor (~0.2 M params).

    agent history --(GRU | small Transformer)--> a
    neighbour histories (incl. the robot) --same encoder--> n_i
    a attends over {n_i} + a learned null token        --> s   (social context)
    [optional] obstacle raster --tiny CNN-->              m
    MLP([a, s, class, m]) --> h ; K mode queries --> K x Tf x 2 (Laplace mean, scale) + K logits

Trained winner-takes-all: Laplace NLL on the mode closest to the ground truth
+ cross-entropy on which mode that was.
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .features import HIST_FEATS


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
        self.agent_enc = TemporalEncoder(cfg, HIST_FEATS)
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

    def forward(self, batch):
        cfg = self.cfg
        a = self.agent_enc(batch["agent_hist"])                              # (B, d)
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
        return dict(traj=out[..., :2], log_scale=out[..., 2:].clamp(-4.0, 3.0),
                    logits=self.cls_head(z)[..., 0])


def predictor_loss(out, fut, fut_valid, cls_weight=0.5):
    """fut (B, Tf, 2) agent frame; fut_valid (B, Tf)."""
    w = fut_valid[:, None, :]                                               # (B, 1, Tf)
    err = torch.linalg.norm(out["traj"] - fut[:, None], dim=-1)             # (B, K, Tf)
    ade = (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)
    best = ade.argmin(dim=1)
    idx = torch.arange(len(best), device=best.device)
    mu, ls = out["traj"][idx, best], out["log_scale"][idx, best]
    nll = (torch.abs(fut - mu) * torch.exp(-ls) + ls + math.log(2.0)).sum(-1)  # (B, Tf)
    reg = (nll * fut_valid).sum() / fut_valid.sum().clamp(min=1.0)
    cls = F.cross_entropy(out["logits"], best)
    return reg + cls_weight * cls, dict(nll=reg.item(), cls=cls.item(), min_ade=ade.min(1)[0].mean().item())


def save_model(model, path, feature_cfg=None):
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(model.cfg),
                "feature_cfg": asdict(feature_cfg) if feature_cfg is not None else None}, path)


def load_model(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model = TrajectoryPredictor(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("feature_cfg")
