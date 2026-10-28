"""Train / evaluate the tier-2 predictor on tracker logs and compare with tier 1.

Real data: every ``run_sequence.py`` run writes ``<out>/track_log.npz``, and
``pipeline_eval.py --track-logs DIR`` writes one per scene from saved detections with the current
tracker settings (CPU). Pass several (split into train / val by *recording* or scene, windows overlap):

    python tools/dog/train_predictor.py --logs rec_*/track_log.npz --val-logs rec_9/track_log.npz \
        --out work_dirs/pred/model.pt

The windows are cut from every track, and most tracks are parked vehicles: on the 0921 logs only a
small share of the windows move. ``--static-keep 0.3`` keeps 30 % of the windows that move less than
``--moving-disp`` m in 3 s (all moving ones stay); ``--mirror`` adds every training window mirrored
left / right. The validation windows are never subsampled or mirrored, and the comparison with tier 1
(the logged online IMM predictions, and constant velocity) is printed for all, moving and static
windows: a parked vehicle predicted to drive off would raise false alarms.

The targets are smoothed from the matched detections' centers (``--targets measured``, the default;
logs written since 2026-10-28) or from the tracker's positions (``filtered``, for older logs); on 0921
the measured ones are 3x closer to the labels (T23). The validation windows use the same targets, so
the numbers printed here are against that pseudo ground truth. ``pipeline_eval.py`` scores against the
reviewed labels, and its "Training targets" table says how close each source is.

Part of the tracker merged into the network (TRACKING_PREDICTION.md 3.3.8): ``--state-head`` also
trains an estimate of the state now (velocity, moving: the auto-label's rule on the targets), and
``--det-inputs`` feeds the agent encoder the detections the tracker matched, not only its filtered
estimates (logs written since 2026-10-28). The state is then compared with the tracker's (its velocity
and logged motion state) on the validation windows.

Without data yet, ``--sim N`` generates N simulated one-minute logs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.prediction import (FeatureConfig, SampleConfig, build_samples, concat_samples,  # noqa: E402
                                       mirror_samples, moving_mask, select_samples)
from dog_perception.prediction.learned import (evaluate_kinematic, evaluate_learned, evaluate_state,  # noqa: E402
                                               train_predictor)
from dog_perception.prediction.model import ModelConfig, save_model  # noqa: E402


def load_logs(patterns):
    files = sorted(f for p in patterns for f in glob.glob(p))
    return [dict(np.load(f)) for f in files], files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--val-logs", nargs="*", default=[])
    ap.add_argument("--sim", type=int, default=0, help="generate N simulated logs (last one is validation)")
    ap.add_argument("--encoder", default="gru", choices=["gru", "transformer"])
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--static-keep", type=float, default=1.0,
                    help="share of the training windows moving less than --moving-disp to keep")
    ap.add_argument("--moving-disp", type=float, default=1.5, help="[m] in the 3 s future: a moving window")
    ap.add_argument("--mirror", action="store_true", help="add every training window mirrored left / right")
    ap.add_argument("--targets", default="measured", choices=["measured", "filtered"],
                    help="smooth the targets from the matched detections or the tracker's positions (old logs)")
    ap.add_argument("--state-head", action="store_true", help="also estimate the velocity and moving now")
    ap.add_argument("--det-inputs", action="store_true", help="the agent encoder also reads the matched detections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/pred/model.pt")
    args = ap.parse_args()

    fc = FeatureConfig(use_detections=args.det_inputs)
    sc = SampleConfig(features=fc, targets=args.targets)
    if args.sim:
        from dog_perception.demo import oracle_tracking_log
        logs = [oracle_tracking_log(seed=100 + i, duration=60.0) for i in range(args.sim)]
        train_logs, val_logs = logs[:-1], logs[-1:]
    else:
        train_logs, _ = load_logs(args.logs)
        val_logs, _ = load_logs(args.val_logs)
        if not val_logs:
            train_logs, val_logs = train_logs[:-1], train_logs[-1:]
    train = concat_samples([build_samples(l, sc) for l in train_logs])
    val = concat_samples([build_samples(l, sc) for l in val_logs])
    mov = moving_mask(train, args.moving_disp)
    print("windows: train %d (%d moving), val %d (%d moving)" % (len(mov), mov.sum(), len(val["future"]),
                                                                 moving_mask(val, args.moving_disp).sum()))
    if args.static_keep < 1.0:
        keep = mov | (np.random.default_rng(args.seed).random(len(mov)) < args.static_keep)
        train = select_samples(train, keep)
        print("kept %d training windows (all %d moving, %.0f %% of the rest)" % (keep.sum(), mov.sum(),
                                                                             100 * args.static_keep))
    if args.mirror:
        train = concat_samples([train, mirror_samples(train)])
        print("mirrored: %d training windows" % len(train["future"]))

    mc = ModelConfig(hist_len=fc.hist_len, fut_len=fc.fut_len, num_modes=args.modes, encoder=args.encoder,
                     det_feats=6 if args.det_inputs else 0, state_head=args.state_head)
    model, _ = train_predictor(train, val, mc, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               device=args.device, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_model(model, args.out, fc)

    vmov = moving_mask(val, args.moving_disp)
    res = {}
    print("\n| windows | predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m | n |")
    print("|---|---|---|---|---|---|---|---|")
    for subset, mask in (("all", np.ones(len(vmov), bool)), ("moving", vmov), ("static", ~vmov)):
        if not mask.any():
            continue
        part = select_samples(val, mask)
        r = evaluate_kinematic(part)
        r["learned"] = evaluate_learned(model, part, args.device)
        res[subset] = r
        for name in ("cv", "imm", "learned"):
            m = r[name]
            print("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %d |" % (subset, name, m["minADE"], m["minFDE"],
                                                                            m["ADE1"], m["FDE1"], m["MR"], m["num"]))
    if args.state_head:
        print("\nState now (targets: velocity over +-0.5 s and the auto-label's moving rule on the pseudo ground "
              "truth; tracker: its velocity and logged motion state, moving = 'moving'; learned: P(moving) >= 0.5)")
        print("\n| windows | estimator | vel err m/s | moving recall | false moving | n |")
        print("|---|---|---|---|---|---|")
        for subset, mask in (("all", np.ones(len(vmov), bool)), ("moving", vmov), ("static", ~vmov)):
            if not mask.any():
                continue
            r = evaluate_state(model, select_samples(val, mask), args.device)
            res[subset]["state"] = r
            for name in ("tracker", "learned"):
                m = r[name]
                print("| %s | %s | %.3f | %.3f | %.4f | %d |" % (subset, name, m["vel_err"], m["moving_recall"],
                                                                 m["false_moving"], m["num"]))
    with open(os.path.splitext(args.out)[0] + "_eval.json", "w") as f:
        json.dump(dict(args={k: v for k, v in vars(args).items()}, eval=res), f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
