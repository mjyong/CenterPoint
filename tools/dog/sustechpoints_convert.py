"""Auto-labels <-> SUSTechPOINTS (box-level correction and adding missed objects).

SUSTechPOINTS is vendored in third_party/SUSTechPOINTS (see its VENDORED.md); scenes go to its data/.

    # export: labels + point clouds -> a SUSTechPOINTS scene (third_party/SUSTechPOINTS/data/rec_001)
    python tools/dog/sustechpoints_convert.py export --labels data/labels/rec_001/labels_reviewed.pkl \
        --frames work_dirs/rec_001/frames --scene rec_001 --every 5

    # label in SUSTechPOINTS: open http://127.0.0.1:8081 (needs cherrypy + jinja2; TensorFlow only for its
    # auto-annotation button). Same as "cd third_party/SUSTechPOINTS && python main.py", with --host / --port.
    python tools/dog/sustechpoints_convert.py serve

    # then import:
    python tools/dog/sustechpoints_convert.py import --scene rec_001 --out data/labels/rec_001/labels_sustech.pkl

    # the imported labels feed export_dataset.py like any labels.pkl
    python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/labels/rec_001/labels_sustech.pkl ...

Frames and boxes are in the detection frame (gravity aligned, ground at z ~ 0). Types map back with
dog_perception.sustechpoints.TYPE_TO_SUBCLASS (Car / Van -> car, Truck and construction machines ->
truck, Bus / Tram / LongVehicle -> bus; override with --type-map Trimotorcycle=car); velocities
are recomputed from the annotated boxes. See docs/DOG_FINETUNE.md.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dog_perception.sustechpoints import export_scene, import_scene  # noqa: E402

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
    e.add_argument("--current-sweep", action="store_true",
                   help="only the current sweep (default: the accumulated sweeps the model sees)")
    i = sub.add_parser("import", help="SUSTechPOINTS scene -> labels.pkl")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="scene name in third_party/SUSTechPOINTS/data")
    src.add_argument("--scene-dir", help="scene directory anywhere else")
    i.add_argument("--out", required=True)
    i.add_argument("--type-map", nargs="*", default=[], metavar="TYPE=CLASS",
                   help="extra / changed type mappings, CLASS in car, truck, bus, pedestrian, cyclist "
                        "(or vehicle without a subclass), or 'skip'")
    i.add_argument("--static-speed", type=float, default=0.2, help="[m/s] slower boxes get zero velocity")
    v = sub.add_parser("serve", help="start SUSTechPOINTS (works without TensorFlow)")
    v.add_argument("--app", default=VENDORED, help="SUSTechPOINTS checkout (default: the vendored copy)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    if args.cmd == "serve":
        serve(args.app, args.host, args.port)
    elif args.cmd == "export":
        with open(args.labels, "rb") as f:
            labels = pickle.load(f)
        n = export_scene(labels, args.frames, args.out, args.scene, args.every, args.start, args.end,
                         args.current_sweep)
        print("%d frames -> %s" % (n, os.path.join(args.out, args.scene)))
    else:
        tmap = dict(kv.split("=", 1) for kv in args.type_map)
        scene_dir = args.scene_dir or os.path.join(VENDORED, "data", args.scene)
        labels, counts = import_scene(scene_dir, tmap, args.static_speed)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(labels, f)
        n_tracks = len({o["track_id"] for fr in labels["frames"] for o in fr["objects"]})
        print("%d frames, %d tracks, boxes %s -> %s" % (len(labels["frames"]), n_tracks, counts, args.out))


if __name__ == "__main__":
    main()
