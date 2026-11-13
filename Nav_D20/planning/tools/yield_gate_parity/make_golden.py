"""Golden data for the C++ yield gate (src/algorithm/yield_gate/): scenarios run through the Python reference
(dog_perception/yield_gate.py), inputs and outputs written as a token file that parity.cc reads back.

    python Nav_D20/planning/tools/yield_gate_parity/make_golden.py --out /tmp/yield_golden.txt
    python Nav_D20/planning/tools/yield_gate_parity/make_golden.py --out /tmp/yield_1104.txt \
        --scenes $(for d in data/labels/1104_*/labels_sustech2.pkl; do basename $(dirname $d); done)

Scenarios: hand-made ones (a car crossing, far, passing, a side road, standing and blocking vehicles, a young
track, stale perception, the crosswalk, a lost vehicle) for each rule; random ones (moving and standing vehicles
with multi-mode predictions and covariances, the robot moving or on the crosswalk); with --scenes, the 1104
tracker output the offline evaluation uses (tools/dog/yield_eval.py), a few arrival times per path.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from dataclasses import fields  # noqa: E402

from dog_perception.yield_gate import YieldConfig, YieldGate, assess  # noqa: E402

MOTION = {"static": 0, "moving": 1, "stopped": 2}
PATH = np.array([[0.0, 0.0], [0.0, 8.0]])


def num(v):
    if v is None:
        return "nan"
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(v)


class Writer:
    def __init__(self, f):
        self.f = f
        self.frames = 0

    def w(self, *tokens):
        self.f.write(" ".join(str(t) for t in tokens) + "\n")

    def config(self, cfg):
        items = []
        for fd in fields(YieldConfig):
            v = getattr(cfg, fd.name)
            if fd.name == "classes":
                items.append("classes %d %s" % (len(v), " ".join(v)))
            elif isinstance(v, str):
                items.append("%s %s" % (fd.name, v))
            else:
                items.append("%s %s" % (fd.name, num(v)))
        self.w("CONFIG", len(items), " ".join(items))

    def frame(self, name, path, objects, now, stamp, dog_moving, on_crosswalk, a, d):
        self.frames += 1
        p = np.asarray(path, dtype=float)
        self.w("FRAME", name, num(now), num(stamp), int(dog_moving), int(on_crosswalk), len(p),
               " ".join("%s %s" % (num(x), num(y)) for x, y in p[:, :2]), len(objects))
        for o in objects:
            size = np.asarray(o["size"], dtype=float)
            cov = o.get("covariance")
            cov = None if cov is None else np.asarray(cov, dtype=float)[:2, :2]
            times = o.get("pred_times")
            times = [] if times is None else list(np.asarray(times, dtype=float))
            toks = ["OBJ", int(o["id"]), o["label"], MOTION[o.get("motion", "static")],
                    num(o["position"][0]), num(o["position"][1]), num(o["velocity"][0]), num(o["velocity"][1]),
                    num(o.get("yaw", 0.0)), num(size[0]), num(size[1]),
                    num(o.get("age")), num(o.get("standing_for"))]
            toks += [1 if cov is not None else 0] + ([num(v) for v in cov.ravel()] if cov is not None else [])
            toks += [len(times)] + [num(t) for t in times]
            if times:
                modes = np.asarray(o["pred_modes"], dtype=float)
                probs = np.asarray(o["pred_probs"], dtype=float)
                covs = o.get("pred_covs")
                toks += [len(probs)]
                for k in range(len(probs)):
                    toks += [num(probs[k])] + [num(v) for v in modes[k, :, :2].ravel()]
                    if covs is not None:
                        toks += [1] + [num(v) for v in np.asarray(covs[k], dtype=float)[:, :2, :2].ravel()]
                    else:
                        toks += [0]
            self.w(*toks)
        self.w("ASSESS", int(a.stale), int(a.in_lane), num(a.min_gap), len(a.conflicts),
               " ".join("%d %s %s %s" % (c["id"], num(c["gap"]), num(c["s"]), c["kind"]) for c in a.conflicts),
               len(a.standing),
               " ".join("%d %s %d" % (st["id"], num(st["standing_for"]), int(st["blocking"])) for st in a.standing),
               len(a.approaching), " ".join(str(i) for i in sorted(a.approaching)),
               len(a.present), " ".join(str(i) for i in sorted(a.present)))
        self.w("DECISION", int(d.go), d.reason, len(d.ids), " ".join(str(i) for i in d.ids), num(d.gap))


def run(writer, name, cfg, frames):
    """frames: [(path, objects, now, stamp, dog_moving, on_crosswalk)], one gate across them."""
    writer.config(cfg)
    writer.w("RESET", name)
    gate = YieldGate(cfg)
    for path, objs, now, stamp, moving, cw in frames:
        a = assess(path, objs, now, stamp, cfg)
        d = gate.update(a, now, dog_moving=moving, on_crosswalk=cw)
        writer.frame(name, path, objs, now, stamp, moving, cw, a, d)


def car(x, y, vx, vy=0.0, oid=1, motion="moving", yaw=None, **kw):
    return dict(id=oid, label="vehicle", position=np.array([x, y]), velocity=np.array([vx, vy]),
                yaw=math.atan2(vy, vx) if yaw is None else yaw, size=np.array([4.5, 1.9, 1.6]), motion=motion, **kw)


def hand_made(writer):
    v30 = 30 / 3.6
    for rule in ("wait", "gap", "stop"):
        cfg = YieldConfig(yield_rule=rule)
        # a car crossing at 30 km/h from 30 m, until it has passed
        run(writer, "crossing_" + rule, cfg,
            [(PATH, [car(-30.0 + v30 * t, 4.0, v30)], t, t, False, False) for t in np.arange(0.0, 12.0, 0.1)])
        run(writer, "far_passed_" + rule, cfg, [
            (PATH, [car(10.0, 4.0, v30, oid=1), car(-100.0, 4.0, v30, oid=2), car(-30.0, 30.0, v30, oid=3),
                    dict(car(-30.0, 4.0, v30, oid=4), label="pedestrian")], 0.0, 0.0, False, False)])
        run(writer, "side_road_" + rule, cfg, [(PATH, [car(-15.0, -6.0 + 1.5 * t, 0.0, 1.5)], t, t, False, False)
                                               for t in np.arange(0.0, 6.0, 0.1)])
        run(writer, "slow_far_" + rule, cfg, [(PATH, [car(-35.0 + 2.0 * t, 4.0, 2.0)], t, t, False, False)
                                              for t in np.arange(0.0, 25.0, 0.2)])
        across = car(-1.0, 4.0, 0.0, motion="stopped", yaw=0.0)
        run(writer, "blocked_" + rule, cfg, [(PATH, [across], t, t, False, False) for t in np.arange(0.0, 18.0, 0.1)])
        run(writer, "standing_nose_" + rule, cfg,
            [(PATH, [car(-6.0, 4.0, 0.0, motion="stopped", yaw=0.0)], t, t, False, False)
             for t in np.arange(0.0, 5.0, 0.1)])
        run(writer, "static_aged_" + rule, cfg,
            [(PATH, [car(-6.0, 4.0, 0.0, motion="static", yaw=0.0, age=20.0)], 0.0, 0.0, False, False)])
        run(writer, "young_" + rule, cfg, [(PATH, [car(-8.0, 20.0, 0.0, 0.5, age=0.3 + t)], t, t, False, False)
                                           for t in np.arange(0.0, 3.0, 0.1)])
        run(writer, "stale_" + rule, cfg, [(PATH, [], t, 0.0 if t < 1.0 else t, False, False)
                                           for t in np.arange(0.0, 4.0, 0.1)] +
            [(PATH, [], 4.0, None, False, False)])
        run(writer, "committed_" + rule, cfg, [(PATH, [car(-12.0, 0.2, v30)], 0.0, 0.0, True, False),
                                               (PATH, [car(-12.0, 0.2, v30)], 0.1, 0.1, False, False)])
        run(writer, "crosswalk_" + rule, cfg, [(PATH, [car(-30.0, 4.0, v30)], 0.0, 0.0, True, True),
                                               (PATH, [], 1.0, 0.0, False, True)])
        run(writer, "lost_" + rule, cfg, [(PATH, [car(-30.0, 4.0, v30)], 0.0, 0.0, False, False)] +
            [(PATH, [], t, t, False, False) for t in np.arange(0.1, 8.0, 0.1)])
        run(writer, "message_age_" + rule, cfg, [(PATH, [car(-30.0, 4.0, v30)], 0.25, 0.0, False, False)])
    run(writer, "min_watch", YieldConfig(min_watch=3.0),
        [(PATH, [], t, t, False, False) for t in np.arange(0.0, 5.0, 0.1)])
    bent = np.array([[0.0, 0.0], [0.0, 3.0], [2.0, 6.0], [2.0, 12.0]])
    run(writer, "bent_path", YieldConfig(), [(bent, [car(-25.0 + 6.0 * t, 5.0, 6.0)], t, t, False, False)
                                              for t in np.arange(0.0, 8.0, 0.2)])


def random_object(rng, oid, t):
    kind = rng.choice(["moving", "moving", "moving", "stopped", "static"])
    label = rng.choice(["vehicle", "vehicle", "cyclist", "pedestrian"])
    p = rng.uniform([-40, -20], [40, 40])
    if kind == "moving":
        sp = rng.uniform(0.2, 12.0)
        th = rng.uniform(-np.pi, np.pi)
        v = sp * np.array([np.cos(th), np.sin(th)])
    else:
        v = rng.normal(0, 0.1, 2)
    o = dict(id=oid, label=label, position=p, velocity=v, yaw=float(rng.uniform(-np.pi, np.pi)),
             size=np.array([rng.uniform(1.5, 12.0), rng.uniform(0.6, 2.6), 1.6]), motion=kind)
    if rng.random() < 0.8:
        o["age"] = float(rng.uniform(0.05, 20.0))
    if rng.random() < 0.5:
        o["standing_for"] = float(rng.uniform(0.0, 20.0))
    if rng.random() < 0.7:
        a = rng.normal(0, 0.5, (4, 4))
        o["covariance"] = a @ a.T
    if kind == "moving" and rng.random() < 0.8:
        T = int(rng.integers(5, 31))
        times = 0.1 * np.arange(1, T + 1)
        K = int(rng.integers(1, 4))
        probs = rng.dirichlet(np.ones(K))
        modes = np.zeros((K, T, 2))
        covs = np.zeros((K, T, 2, 2))
        for k in range(K):
            w = rng.normal(0, 0.3)
            th0 = math.atan2(v[1], v[0])
            sp = float(np.hypot(*v)) * rng.uniform(0.6, 1.2)
            x, y, th = p[0], p[1], th0
            for i in range(T):
                th += w * 0.1
                x += sp * 0.1 * math.cos(th)
                y += sp * 0.1 * math.sin(th)
                modes[k, i] = (x, y)
                s2 = (0.2 + 0.4 * times[i]) ** 2
                c = rng.uniform(-0.3, 0.3) * s2
                covs[k, i] = [[s2, c], [c, s2 * rng.uniform(0.5, 1.5)]]
        o.update(pred_times=times, pred_modes=modes, pred_probs=probs, pred_covs=covs if rng.random() < 0.7 else None)
    return o


def random_scenes(writer, n, seed):
    rng = np.random.default_rng(seed)
    for sc in range(n):
        rule = rng.choice(["wait", "wait", "gap", "stop"])
        cfg = YieldConfig(yield_rule=str(rule), min_watch=float(rng.choice([0.0, 0.0, 2.0])))
        if rng.random() < 0.5:
            path = np.array([[0.0, 0.0], [0.0, rng.uniform(4.0, 30.0)]])
        else:
            path = np.cumsum(np.vstack([[0.0, 0.0], rng.uniform([-2, 1], [2, 4], (int(rng.integers(2, 6)), 2))]), 0)
        objs = [random_object(rng, i + 1, 0.0) for i in range(int(rng.integers(0, 7)))]
        frames = []
        t = 0.0
        for k in range(int(rng.integers(1, 40))):
            stamp = t - rng.choice([0.0, 0.05, 0.2, 0.5]) if rng.random() < 0.95 else None
            moved = []
            for o in objs:
                o = dict(o)
                o["position"] = o["position"] + o["velocity"] * t
                if o.get("pred_times") is not None:
                    o["pred_modes"] = o["pred_modes"] + o["velocity"] * t
                if o.get("age") is not None:
                    o["age"] = o["age"] + t
                if rng.random() > 0.1:                 # sometimes not seen in a frame
                    moved.append(o)
            frames.append((path, moved, t, stamp, bool(rng.random() < 0.2), bool(rng.random() < 0.05)))
            t += float(rng.choice([0.1, 0.1, 0.5, 1.0]))
        run(writer, "random_%03d" % sc, cfg, frames)


def eval_scenes(writer, scenes, results, per_scene, fan, length):
    """The 1104 tracker output as tools/dog/yield_eval.py feeds the gate: fan paths in the det frame of the first good
    frame, perception stamps plus the evaluation's latency; a few 60-frame stretches per path."""
    import pickle
    from dog_perception.yield_eval import EvalConfig, det_to_world, fan_paths, track_objects
    ev = EvalConfig()
    cfg = YieldConfig()
    paths = fan_paths(fan, length)
    for scene in scenes:
        path_file = results.format(scene=scene)
        if not os.path.isfile(path_file):
            print("%s: missing, skipped" % path_file)
            continue
        with open(path_file, "rb") as f:
            res = pickle.load(f)
        good = [r for r in res if not r.get("pose_bad")]
        if not good:
            continue
        T0 = np.asarray(good[0]["T_world_det"])
        frames = track_objects(res)
        for name, p in paths.items():
            path = det_to_world(p, T0)
            for k0 in np.linspace(0, max(0, len(frames) - 60), per_scene).astype(int):
                sel = [(path, objs, st + ev.latency, st, False, False) for st, objs in frames[k0:k0 + 60]]
                run(writer, "%s_%s_%d" % (scene, name, k0), cfg, sel)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=300, help="random scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--results", default="data/runs_online/{scene}/pillar/results.pkl")
    ap.add_argument("--per-scene", type=int, default=3)
    ap.add_argument("--fan", type=int, default=8)
    ap.add_argument("--length", type=float, default=8.0)
    args = ap.parse_args(argv)
    with open(args.out, "w") as f:
        w = Writer(f)
        hand_made(w)
        random_scenes(w, args.random, args.seed)
        if args.scenes:
            eval_scenes(w, args.scenes, args.results, args.per_scene, args.fan, args.length)
        w.w("END")
    print("%s: %d frames" % (args.out, w.frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
