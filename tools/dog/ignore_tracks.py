"""Turn label tracks into ignore boxes: neither a vehicle the detector must find nor background.

The forklifts that unload goods need not be detected (2026-10-26). They are picked by the median
box size of their tracks in the reviewed keyframe labels (the forklift checked in Foxglove:
3.4 x 1.5 x 2.0 m; others 2.4-3.2 m long, 1.4-1.8 m wide, 2.0-2.5 m high), or named as
scene:track, and marked ignore (``ignore_reason``) in the keyframe labels and in the 10 Hz labels
propagated from them. propagate_keyframes renumbers the boxes a reviewer added, so 10 Hz ids are
matched to keyframe ids by position at the keyframes. Ignore boxes are left out of AP, MOT and the
training loss alike (export_dataset.py, compare_detectors.py, pipeline_eval.py).

    python tools/dog/ignore_tracks.py --scenes $ALL --keyframes "data/labels/{scene}/labels_sustech2.pkl" \\
        --also "data/labels/{scene}/labels_full2.pkl" "data/labels/{scene}/labels_train2.pkl" \\
        --length 2.3 3.6 --width 1.3 1.85 --height 1.95 2.6 --reason forklift --dry-run

Files are rewritten in place (they are in git); --dry-run only lists the tracks. Running it again
changes nothing.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.pipeline_eval import _key, _to_world, match_dense_ids  # noqa: E402


def select_tracks(keyframes, length, width, height, label="vehicle"):
    """{track id: dict(keyframes, lwh, center (world), moving)} of the not yet ignored ``label`` tracks
    whose median box size lies in the (lo, hi) ranges."""
    obs = {}
    for f in keyframes["frames"]:
        for o in f["objects"]:
            if o["label"] == label:
                obs.setdefault(o["track_id"], []).append((o, _to_world(o["box"], f["T_world_det"])))
    out = {}
    for tid, v in obs.items():
        if all(o.get("ignore") for o, _ in v):
            continue
        lwh = np.median([o["box"][3:6] for o, _ in v], 0)
        if all(lo <= x <= hi for x, (lo, hi) in zip(lwh, (length, width, height))):
            out[tid] = dict(keyframes=len(v), lwh=lwh, center=np.median([w[:2] for _, w in v], 0),
                            moving=any(o.get("moving") for o, _ in v))
    return out


def mark_ignore(labels, ids, reason, id_of=None, label="vehicle"):
    """``labels`` with the objects of tracks ``ids`` turned into ignore boxes, and how many changed.
    ``id_of(track id)``: the keyframe id of a track (None: the track id itself)."""
    n = 0
    frames = []
    for f in labels["frames"]:
        objs = []
        for o in f["objects"]:
            tid = o["track_id"] if id_of is None else id_of(o["track_id"])
            if o["label"] == label and tid in ids and not (o.get("ignore") and o.get("ignore_reason") == reason):
                o = dict(o, ignore=True, ignore_reason=reason)
                n += 1
            objs.append(o)
        frames.append(dict(f, objects=objs))
    return dict(labels, frames=frames), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--keyframes", required=True, help="reviewed keyframe labels, {scene} for the scene name")
    ap.add_argument("--also", nargs="*", default=[], help="10 Hz labels propagated from them (missing files skipped)")
    for k in ("length", "width", "height"):
        ap.add_argument("--" + k, type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--tracks", nargs="*", default=[], metavar="SCENE:ID", help="tracks to ignore besides the size rule")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="SCENE:ID", help="tracks the size rule must not take")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    by_size = all(getattr(args, k) is not None for k in ("length", "width", "height"))
    named = {(s.split(":")[0], int(s.split(":")[1])) for s in args.tracks}
    excluded = {(s.split(":")[0], int(s.split(":")[1])) for s in args.exclude}
    total = {}
    for scene in args.scenes:
        path = args.keyframes.replace("{scene}", scene)
        with open(path, "rb") as f:
            key = pickle.load(f)
        sel = select_tracks(key, args.length, args.width, args.height) if by_size else {}
        ids = {t for t in sel if (scene, t) not in excluded} | {t for s, t in named if s == scene}
        for t in sorted(ids):
            i = sel.get(t)
            print("%s track %d: %s" % (scene, t, "named" if i is None else
                                       "%d keyframes, %.2f x %.2f x %.2f m, at (%.1f, %.1f)%s" % (
                                           i["keyframes"], *i["lwh"], *i["center"], ", moving" if i["moving"] else "")))
        if not ids:
            continue
        key2, n = mark_ignore(key, ids, args.reason)
        changes = [(path, key2, n)]
        added = {o["track_id"] for f in key["frames"] for o in f["objects"] if o.get("added")}
        kf = {_key(f["stamp"]): f for f in key["frames"]}
        for pattern in args.also:
            p = pattern.replace("{scene}", scene)
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                dense = pickle.load(f)
            m = match_dense_ids(kf, {_key(f["stamp"]): f for f in dense["frames"]})
            # a 10 Hz track never at a keyframe keeps its id unless that id was renumbered (an added box)
            id_of = lambda t: m.get(t, t if t not in added else None)   # noqa: E731
            changes.append((p, *mark_ignore(dense, ids, args.reason, id_of)))
        for p, labels, n in changes:
            total[os.path.basename(p)] = total.get(os.path.basename(p), 0) + n
            print("  %s: %d boxes -> ignore" % (p, n))
            if not args.dry_run and n:
                with open(p, "wb") as f:
                    pickle.dump(labels, f)
    print("total:", ", ".join("%s %d" % kv for kv in total.items()) or "nothing", "(dry run)" if args.dry_run else "")


if __name__ == "__main__":
    main()
