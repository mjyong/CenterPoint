"""Train / evaluate the tier-2 predictor on tracker logs and compare with tier 1.

Real data: every ``run_sequence.py`` run writes ``<out>/track_log.npz``;
pass several of them (split into train / val by *recording*, windows overlap):

    python tools/dog/train_predictor.py --logs rec_*/track_log.npz --val-logs rec_9/track_log.npz \
        --out work_dirs/pred/model.pt

Without data yet, ``--sim N`` generates N simulated one-minute logs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.prediction import FeatureConfig, SampleConfig, build_samples, concat_samples  # noqa: E402
from dog_perception.prediction.learned import evaluate_kinematic, evaluate_learned, train_predictor  # noqa: E402
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
    ap.add_argument("--out", default="work_dirs/pred/model.pt")
    args = ap.parse_args()

    fc = FeatureConfig()
    sc = SampleConfig(features=fc)
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
    print("samples: train %d, val %d" % (len(train["future"]), len(val["future"])))

    mc = ModelConfig(hist_len=fc.hist_len, fut_len=fc.fut_len, num_modes=args.modes, encoder=args.encoder)
    model, _ = train_predictor(train, val, mc, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               device=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_model(model, args.out, fc)

    res = evaluate_kinematic(val)
    res["learned"] = evaluate_learned(model, val, args.device)
    print("\n| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |")
    print("|---|---|---|---|---|---|")
    for name in ("cv", "imm", "learned"):
        m = res[name]
        print("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (name, m["minADE"], m["minFDE"], m["ADE1"], m["FDE1"], m["MR"]))
    with open(os.path.splitext(args.out)[0] + "_eval.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
