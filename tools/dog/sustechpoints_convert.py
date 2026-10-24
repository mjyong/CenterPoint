"""Auto-labels <-> SUSTechPOINTS (box-level correction and adding missed objects).

SUSTechPOINTS is vendored in third_party/SUSTechPOINTS (see its VENDORED.md); scenes go to its data/.

    # export: labels + point clouds -> a SUSTechPOINTS scene (third_party/SUSTechPOINTS/data/rec_001);
    # --split train / val groups the scene list in the vendored tool
    python tools/dog/sustechpoints_convert.py export --labels data/labels/rec_001/labels_reviewed.pkl \
        --frames work_dirs/rec_001/frames --scene rec_001 --every 5 --split train

    # label in SUSTechPOINTS: open http://127.0.0.1:8081 (needs cherrypy + jinja2; TensorFlow only for its
    # auto-annotation button). Same as "cd third_party/SUSTechPOINTS && python main.py", with --host / --port.
    python tools/dog/sustechpoints_convert.py serve

    # then import:
    python tools/dog/sustechpoints_convert.py import --scene rec_001 --out data/labels/rec_001/labels_sustech.pkl

    # validation labels made from the teacher's boxes favour the teacher: add what other models detect and no
    # box accounts for as red candidates (reviewed scenes too; {scene} stands for each scene), review, import
    python tools/dog/sustechpoints_convert.py candidates --scenes rec_001 rec_002 --min-score 0.3 \
        --results v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl nusc_pp=data/runs_nusc_pp/{scene}/pillar/results.pkl

    # the imported labels feed export_dataset.py like any labels.pkl (keyframes only); for a training scene,
    # first spread the reviewed keyframes over every frame:
    python tools/dog/sustechpoints_convert.py propagate --full data/labels/rec_001/labels.pkl \
        --keyframes data/labels/rec_001/labels_sustech.pkl --out data/labels/rec_001/labels_full.pkl
    python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/labels/rec_001/labels_full.pkl ...

Frames and boxes are in the detection frame (gravity aligned, ground at z ~ 0). Types map back with
dog_perception.sustechpoints.TYPE_TO_SUBCLASS (Car / Van -> car, Truck and construction machines ->
truck, Bus / Tram / LongVehicle -> bus; override with --type-map Trimotorcycle=car); velocities
are recomputed from the annotated boxes. See docs/DOG_FINETUNE.md.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.detection.boxes import CLASSES  # noqa: E402
from dog_perception.sustechpoints import (add_candidates, export_scene, import_scene,  # noqa: E402
                                          propagate_keyframes, write_detections)
from dog_perception.visibility import annotate_points  # noqa: E402

VENDORED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "SUSTechPOINTS"))


def serve(app, host, port):
    """Run SUSTechPOINTS' cherrypy app (``python main.py`` with --host / --port). The vendored copy runs
    without TensorFlow (see third_party/SUSTechPOINTS/VENDORED.md)."""
    import cherrypy                    # before chdir: its autoreload restarts from the cwd at import time

    app = os.path.abspath(os.path.expanduser(app))
    os.makedirs(os.path.join(app, "data"), exist_ok=True)
    os.chdir(app)                      # the app resolves data/, public/ and server.conf from the cwd
    sys.path.insert(0, app)
    import main as sustech
    # as cherrypy.quickstart(Root(), "/", config="server.conf"), but host / port from the command line win
    cherrypy.config.update("server.conf")
    cherrypy.config.update({"server.socket_host": host, "server.socket_port": port})
    cherrypy.tree.mount(sustech.Root(), "/", config="server.conf")
    scenes = sorted(d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d)))
    print("SUSTechPOINTS on http://%s:%d (scenes: %s)" % (host, port, ", ".join(scenes) or "none yet"))
    cherrypy.engine.signals.subscribe()
    cherrypy.engine.start()
    cherrypy.engine.block()


def candidates(args):
    """``candidates`` and ``detections`` over the scenes, with each model's results.pkl of the scene."""
    specs = [kv.split("=", 1) for kv in args.results]
    total = Counter()
    for scene in args.scenes:
        results = {}
        for name, path in specs:
            with open(path.replace("{scene}", scene), "rb") as f:
                results[name] = pickle.load(f)
        scene_dir = os.path.join(args.root, scene)
        if args.cmd == "detections" or not args.dry_run:
            n = write_detections(scene_dir, results, args.dets_min_score)
            print("%s: %d model boxes for Ctrl + drag -> %s" % (scene, n, os.path.join(scene_dir, "dog_dets")))
        if args.cmd == "detections":
            continue
        if args.dry_run:
            tmp = tempfile.mkdtemp()
            work = os.path.join(tmp, scene)
            shutil.copytree(scene_dir, work, ignore=shutil.ignore_patterns("lidar", "label_backup", "dog_dets"))
            os.symlink(os.path.abspath(os.path.join(scene_dir, "lidar")), os.path.join(work, "lidar"))
            counts = add_candidates(work, results, args.min_score, args.match_dist, min_points=args.min_points,
                                    backup=False)
            shutil.rmtree(tmp)
        else:
            counts = add_candidates(scene_dir, results, args.min_score, args.match_dist, min_points=args.min_points)
        total.update(counts)
        print("%s: %d candidate boxes (%d objects); %s" % (
            scene, counts.get("candidate boxes", 0), counts.get("candidate objects", 0),
            ", ".join("%s %d" % kv for kv in sorted(counts.items()) if not kv[0].startswith("candidate"))))
    if args.cmd == "detections":
        return
    print("total: %d candidate boxes (%d objects); %s" % (
        total["candidate boxes"], total["candidate objects"],
        ", ".join("%s %d" % kv for kv in sorted(total.items()) if not kv[0].startswith("candidate"))))
    if args.dry_run:
        print("dry run: nothing written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="labels.pkl + frames -> SUSTechPOINTS scene")
    e.add_argument("--labels", required=True, help="labels.pkl (auto_label.py or review_labels.py apply)")
    e.add_argument("--frames", required=True, help="run_sequence.py --save-frames output (<run>/frames)")
    e.add_argument("--out", default=os.path.join(VENDORED, "data"), help="SUSTechPOINTS data/ directory")
    e.add_argument("--scene", required=True, help="scene name (directory under --out)")
    e.add_argument("--every", type=int, default=1, help="export every n-th frame (5: 2 Hz from 10 Hz)")
    e.add_argument("--start", type=float, default=None, help="seconds since the recording start")
    e.add_argument("--end", type=float, default=None)
    e.add_argument("--vehicles-only", action="store_true",
                   help="export only the vehicle labels (the imported labels then hold vehicles only)")
    e.add_argument("--current-sweep", action="store_true",
                   help="only the current sweep (default: the accumulated sweeps the model sees; older sweeps are "
                        "drawn dimmer in the vendored tool)")
    e.add_argument("--force", action="store_true",
                   help="write over a scene whose labels were changed in SUSTechPOINTS (the review is lost)")
    e.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="written to the scene's desc.json: the tool lists validation and training scenes apart")
    i = sub.add_parser("import", help="SUSTechPOINTS scene -> labels.pkl")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="scene name in third_party/SUSTechPOINTS/data")
    src.add_argument("--scene-dir", help="scene directory anywhere else")
    i.add_argument("--out", required=True)
    i.add_argument("--type-map", nargs="*", default=[], metavar="TYPE=CLASS",
                   help="extra / changed type mappings, CLASS in car, truck, bus, pedestrian, cyclist "
                        "(or vehicle without a subclass), or 'skip'")
    i.add_argument("--static-speed", type=float, default=0.2, help="[m/s] slower boxes get zero velocity")
    i.add_argument("--frames", default=None,
                   help="the scene's run_sequence.py frames: count the points in each box (occluded / far boxes "
                        "are then not scored by compare_detectors.py, as in export_dataset.py)")
    g = sub.add_parser("propagate", help="keyframes reviewed in SUSTechPOINTS -> labels at every frame (training scenes)")
    g.add_argument("--full", required=True, help="the scene's labels at every frame (labels.pkl or labels_reviewed.pkl)")
    g.add_argument("--keyframes", required=True, help="import of the reviewed keyframes (labels_sustech.pkl)")
    g.add_argument("--out", required=True)
    g.add_argument("--max-gap", type=float, default=1.0, help="[s] frames between keyframes further apart are left out")
    g.add_argument("--classes", nargs="+", choices=CLASSES, default=None,
                   help="label classes the review covered (default: the export's, e.g. vehicle for --vehicles-only); "
                        "--full objects of other classes are copied as they are")
    c = sub.add_parser("candidates", help="other models' detections no box accounts for -> red boxes to check "
                                          "(also writes the detections for Ctrl + drag, as 'detections')")
    d = sub.add_parser("detections", help="the models' boxes per frame: Ctrl + drag in the tool snaps to them")
    for p in (c, d):
        p.add_argument("--root", default=os.path.join(VENDORED, "data"), help="SUSTechPOINTS data/ directory")
        p.add_argument("--scenes", nargs="+", required=True, help="scene names under --root")
        p.add_argument("--results", nargs="+", required=True, metavar="NAME=PATH",
                       help="run_sequence.py results.pkl per model; {scene} in PATH stands for the scene name")
        p.add_argument("--dets-min-score", type=float, default=0.15, help="lowest score of the boxes for Ctrl + drag")
    c.add_argument("--min-score", type=float, default=0.3)
    c.add_argument("--match-dist", type=float, default=2.0,
                   help="[m] a detection this close to a label (or a deleted box) is not a candidate")
    c.add_argument("--min-points", type=int, default=1, help="points (all sweeps) a candidate needs to be judged")
    c.add_argument("--dry-run", action="store_true", help="count only; the scenes are not changed")
    st = sub.add_parser("stats", help="frames, boxes and boxes still marked review per scene")
    st.add_argument("--root", default=os.path.join(VENDORED, "data"), help="SUSTechPOINTS data/ directory")
    v = sub.add_parser("serve", help="start SUSTechPOINTS (works without TensorFlow)")
    v.add_argument("--app", default=VENDORED, help="SUSTechPOINTS checkout (default: the vendored copy)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    if args.cmd == "serve":
        serve(args.app, args.host, args.port)
    elif args.cmd in ("candidates", "detections"):
        candidates(args)
    elif args.cmd == "stats":
        totals = {}
        print("scene split frames boxes review")
        for name in sorted(os.listdir(args.root)):
            d = os.path.join(args.root, name)
            if not os.path.isdir(os.path.join(d, "label")):
                continue
            desc = os.path.join(d, "desc.json")
            split = json.load(open(desc)).get("split", "") if os.path.exists(desc) else ""
            labels = [json.load(open(os.path.join(d, "label", f))) or [] for f in sorted(os.listdir(os.path.join(d, "label")))]
            n = [len(labels), sum(len(x) for x in labels),
                 sum("review" in str(a.get("obj_attr") or "") for x in labels for a in x)]
            print(name, split or "-", *n)
            totals[split or "-"] = [a + b for a, b in zip(totals.get(split or "-", [0, 0, 0]), n)]
        for split, n in sorted(totals.items()):
            print("total %s: %d frames, %d boxes, %d marked review" % (split, *n))
    elif args.cmd == "propagate":
        with open(args.full, "rb") as f:
            full = pickle.load(f)
        with open(args.keyframes, "rb") as f:
            keyed = pickle.load(f)
        labels, counts = propagate_keyframes(full, keyed, max_gap=args.max_gap, classes=args.classes)
        with open(args.out, "wb") as f:
            pickle.dump(labels, f)
        print("%d of %d frames, %s -> %s" % (len(labels["frames"]), len(full["frames"]), counts, args.out))
    elif args.cmd == "export":
        with open(args.labels, "rb") as f:
            labels = pickle.load(f)
        n = export_scene(labels, args.frames, args.out, args.scene, args.every, args.start, args.end,
                         args.current_sweep, split=args.split, classes="vehicle" if args.vehicles_only else "coarse",
                         overwrite=args.force, label_classes=["vehicle"] if args.vehicles_only else None)
        print("%d frames -> %s" % (n, os.path.join(args.out, args.scene)))
    else:
        tmap = dict(kv.split("=", 1) for kv in args.type_map)
        scene_dir = args.scene_dir or os.path.join(VENDORED, "data", args.scene)
        labels, counts = import_scene(scene_dir, tmap, args.static_speed)
        if args.frames:
            for fr in labels["frames"]:
                cloud = os.path.join(args.frames, "%.6f.npy" % fr["stamp"])
                if fr["objects"] and os.path.exists(cloud):
                    fr["objects"] = annotate_points(fr["objects"], np.load(cloud))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(labels, f)
        n_tracks = len({o["track_id"] for fr in labels["frames"] for o in fr["objects"]})
        print("%d frames, %d tracks, boxes %s -> %s" % (len(labels["frames"]), n_tracks, counts, args.out))


if __name__ == "__main__":
    main()
