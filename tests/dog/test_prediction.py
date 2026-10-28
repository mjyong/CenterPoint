import numpy as np
import pytest
import torch

from dog_perception.demo import oracle_tracking_log
from dog_perception.prediction import (FeatureConfig, SampleConfig, TrackLogger, build_inputs, build_samples,
                                       merge_modes, rts_smooth)
from dog_perception.prediction.dataset import ROW
from dog_perception.prediction.metrics import along_across
from dog_perception.prediction.learned import (LearnedPredictor, evaluate_kinematic, evaluate_learned,
                                               train_predictor)
from dog_perception.prediction.model import ModelConfig, TrajectoryPredictor, load_model, predictor_loss, save_model
from dog_perception.detection.boxes import Detections
from dog_perception.tracking import MultiObjectTracker, TrackerConfig


@pytest.fixture(scope="module")
def logs():
    return [oracle_tracking_log(seed=s, duration=25.0, num_pedestrians=8) for s in range(3)]


def test_rts_smoother_reduces_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 5, 0.1)
    truth = np.column_stack([t * 1.2, np.sin(t) * 2])
    noisy = truth + rng.normal(0, 0.15, truth.shape)
    sm = rts_smooth(t, noisy)
    assert np.sqrt(np.mean((sm - truth) ** 2)) < 0.6 * np.sqrt(np.mean((noisy - truth) ** 2))


def test_merge_modes_collapses_identical_paths():
    m = np.zeros((2, 3, 2))
    m[:, :, 0] = np.arange(3)
    c = np.stack([np.tile(np.eye(2) * 0.1, (3, 1, 1)), np.tile(np.eye(2) * 1.0, (3, 1, 1))])
    mm, cc, pp = merge_modes(m, c, np.array([0.7, 0.3]), 0.2)
    assert len(mm) == 1 and np.isclose(pp[0], 1.0) and np.isclose(cc[0, 0, 0, 0], 0.7 * 0.1 + 0.3 * 1.0)


def test_agent_frame_normalisation():
    fc = FeatureConfig()
    t = np.arange(0, 2.0, 0.1)
    # agent moving along +y (world); in its own frame it must move along +x and end at the origin
    h = np.column_stack([t, np.zeros_like(t), t * 1.0, np.zeros_like(t), np.ones_like(t), np.ones_like(t)])
    other = np.column_stack([t, np.full_like(t, 3.0), t * 1.0, np.zeros_like(t), np.ones_like(t), np.ones_like(t)])
    inp, meta = build_inputs({1: h, 2: other}, {1: 1, 2: 0}, {1: 0.0, 2: 0.0}, [1], t[-1], fc)
    ah = inp["agent_hist"][0]
    assert np.allclose(ah[-1, :2], 0, atol=1e-6) and np.all(np.diff(ah[:, 0]) > 0) and np.allclose(ah[:, 1], 0, atol=1e-6)
    assert np.allclose(ah[-1, 2:4], [1.0, 0.0], atol=1e-6)
    assert inp["nbr_mask"][0].sum() == 1 and np.allclose(inp["nbr_hist"][0, 0, -1, :2], [0.0, -3.0], atol=1e-6)
    assert np.isclose(meta["thetas"][0], np.pi / 2)


@pytest.mark.slow
def test_build_samples_and_online_imm(logs):
    s = build_samples(logs[0], SampleConfig())
    n = len(s["future"])
    assert n > 100
    assert s["agent_hist"].shape == (n, 20, 6) and s["nbr_hist"].shape == (n, 16, 20, 6)
    assert s["future"].shape == (n, 6, 2) and s["imm_modes"].shape == (n, 2, 6, 2)
    ev = evaluate_kinematic(s)
    assert ev["imm"]["num"] > 0 and ev["imm"]["minADE"] < 2.0


@pytest.mark.parametrize("encoder,raster", [("gru", False), ("transformer", True)])
def test_model_forward_backward(encoder, raster):
    cfg = ModelConfig(encoder=encoder, use_raster=raster)
    m = TrajectoryPredictor(cfg)
    B = 4
    batch = dict(agent_hist=torch.randn(B, 20, 6), agent_class=torch.eye(3)[[0, 1, 2, 1]],
                 nbr_hist=torch.randn(B, 16, 20, 6), nbr_static=torch.zeros(B, 16, 4),
                 nbr_mask=torch.zeros(B, 16))
    batch["agent_hist"][..., 5] = 1.0
    batch["nbr_mask"][0, :3] = 1.0
    if raster:
        batch["raster"] = torch.zeros(B, 1, 64, 64)
    out = m(batch)
    assert out["traj"].shape == (B, 6, 6, 2) and out["logits"].shape == (B, 6)
    loss, info = predictor_loss(out, torch.randn(B, 6, 2), torch.ones(B, 6))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters() if p.requires_grad and p.grad is not None)


@pytest.mark.slow
def test_training_converges_and_runtime_wrapper(logs, tmp_path):
    train = build_samples(logs[0])
    for l in logs[1:2]:
        extra = build_samples(l)
        train = {k: np.concatenate([train[k], extra[k]]) for k in train}
    val = build_samples(logs[2])
    model, hist = train_predictor(train, val, epochs=8, verbose=False, log_every=100)
    res = evaluate_learned(model, val)
    assert hist[-1]["train_loss"] < 0.7 * hist[0]["train_loss"]
    assert np.isfinite(res["minADE"]) and res["minADE"] < 2.0
    path = str(tmp_path / "m.pt")
    save_model(model, path, FeatureConfig())
    m2, fc = load_model(path)
    assert fc["hist_len"] == 20

    # runtime wrapper on a live tracker, with IMM fallback for young tracks
    trk = MultiObjectTracker(TrackerConfig(classes=("pedestrian",)))
    for k in range(15):
        t = 0.1 * (k + 1)
        d = Detections(np.array([[k * 0.1, 0.0, 0.9, 0.6, 0.6, 1.7, 0.0], [5.0, 5.0 - k * 0.1, 0.9, 0.6, 0.6, 1.7, 0.0]]),
                       np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([0.9, 0.9]), np.array([1, 1]), "world")
        trk.step(d, t)
    trk.step(Detections(np.array([[20.0, 0.0, 0.9, 0.6, 0.6, 1.7, 0.0]]), np.zeros((1, 2)), np.array([0.9]),
                        np.array([1]), "world"), 1.6)
    trk.step(Detections(np.array([[20.0, 0.0, 0.9, 0.6, 0.6, 1.7, 0.0]]), np.zeros((1, 2)), np.array([0.9]),
                        np.array([1]), "world"), 1.7)
    lp = LearnedPredictor(path)
    preds = lp(trk)
    src = sorted(p.source for p in preds)
    assert "learned" in src and "imm" in src
    for p in preds:
        assert p.modes.shape[1:] == (6, 2) and np.isclose(p.probs.sum(), 1.0)


def test_mirror_and_moving_windows(logs):
    from dog_perception.prediction import mirror_samples, moving_mask, select_samples
    s = build_samples(logs[0], SampleConfig())
    m = mirror_samples(s)
    assert np.allclose(m["future"][..., 1], -s["future"][..., 1]) and np.allclose(m["future"][..., 0], s["future"][..., 0])
    assert np.allclose(m["agent_hist"][..., 3], -s["agent_hist"][..., 3])
    assert np.allclose(m["nbr_hist"][..., 1], -s["nbr_hist"][..., 1])
    mov = moving_mask(s, 1.5)
    assert 0 < mov.sum() < len(mov)
    part = select_samples(s, mov)
    assert len(part["future"]) == mov.sum() and (np.hypot(*part["future"][:, -1].T) >= 1.5).all()


@pytest.mark.slow
def test_train_predictor_tool_reports_moving_and_static(tmp_path):
    import json
    import os
    import subprocess
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = tmp_path / "pred" / "gru.pt"
    subprocess.run([sys.executable, os.path.join(root, "tools", "dog", "train_predictor.py"), "--sim", "2",
                    "--epochs", "1", "--static-keep", "0.3", "--mirror", "--out", str(out)], check=True,
                   env=dict(os.environ, PYTHONPATH=root), capture_output=True)
    res = json.load(open(tmp_path / "pred" / "gru_eval.json"))
    assert {"all", "moving", "static"} <= set(res["eval"]) and "learned" in res["eval"]["moving"]
    assert out.exists()


def test_along_across_splits_speed_from_path():
    gt = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])          # one step along +x, then a left turn
    along, across = along_across(np.array([[0.9, 1.0], [1.5, 0.5], [2.0, 0.0]]), np.zeros(2), gt)
    assert np.allclose(along, [1.0, -0.5, -2.0]) and np.allclose(across, [0.1, 0.5, 1.0])
    # beyond its ends the path runs on along the first / last segment
    along, across = along_across(np.array([[4.0, 1.0], [-1.0, 0.5]]), np.zeros(2), np.array([[1.0, 0.0], [3.0, 0.0]]))
    assert np.allclose(along, [3.0, -4.0]) and np.allclose(across, [1.0, 0.5])
    assert np.isnan(along_across(np.ones((1, 2)), np.zeros(2), np.zeros((1, 2)))).all()   # no path: a parked car


def test_logged_detections_and_measured_targets():
    """The track log keeps the matched detection of every observed frame; the targets can be smoothed
    from those instead of the tracker's positions. Logs written before keep working with "filtered"."""
    rng = np.random.default_rng(0)
    trk, log, fed = MultiObjectTracker(TrackerConfig()), TrackLogger(), {}
    for k in range(80):
        t = 0.1 * k
        if 40 <= k < 43:                                       # missed for 0.3 s: the track coasts
            dets = Detections(stamp=t)
        else:
            box = np.array([[5.0 + 2.0 * t + rng.normal(0, 0.1), rng.normal(0, 0.1), 0.8, 4.5, 1.9, 1.6, 0.0]])
            fed[round(t, 6)] = box[0, :2]
            dets = Detections(box, np.array([[2.0, 0.0]]), np.array([0.9]), np.zeros(1, np.int64), "world", t)
        log.add(t, trk.step(dets, t))
    d = log.to_dict()
    rows = d["rows"]
    assert rows.shape[1] == len(ROW) == 15
    coast = rows[:, 8] < 0.5
    assert coast.sum() == 3 and np.isnan(rows[coast, 9:]).all()
    assert all(np.allclose(r[9:11], fed[round(r[0], 6)]) for r in rows[~coast])   # the detections themselves
    assert np.allclose(rows[~coast, 12], 2.0) and np.allclose(rows[~coast, 14], 0.9)
    filtered, measured = build_samples(d, SampleConfig(targets="filtered")), build_samples(d, SampleConfig())
    assert len(measured["future"]) == len(filtered["future"]) > 10
    assert not np.allclose(measured["future"], filtered["future"])
    old = dict(d, rows=rows[:, :9])
    assert np.allclose(build_samples(old, SampleConfig(targets="filtered"))["future"], filtered["future"])
    with pytest.raises(ValueError):
        build_samples(old)                                     # measured, the default
