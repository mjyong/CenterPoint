"""Teacher inference + auto-labelling of many sequences on all free GPUs (docs/BAG_SPLIT.md section 4).

    python tools/dog/batch_label.py --split data/survey/split/split.csv --seq-root data/seq \
        --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth [--gpus auto] [--per-gpu 1] [--dry-run]

1. GPUs: ``--gpus auto`` asks nvidia-smi for every card's memory and utilisation and takes the
   free ones: at least ``--min-free-mb`` free (default 8 GB, enough for voxel + TTA), at most
   ``--max-mem-mb`` in use, and at most ``--max-util`` % busy. Another process sitting on the card
   does not by itself make it busy. ``--gpus 0,2`` takes those cards without this check. A
   ``CUDA_VISIBLE_DEVICES`` already set limits the choice. Indices are nvidia-smi's
   (``CUDA_DEVICE_ORDER=PCI_BUS_ID`` for the workers).
2. Scenes: the ``split.csv`` rows that are not ``drop`` (or ``--seq``), longest first so the
   cards finish together.
3. One worker per free card (``--per-gpu`` workers per card when one process leaves the card
   idle, e.g. waiting on the CPU preprocessing) takes the next scene and runs
   ``run_sequence.py --detectors voxel --tta --save-frames`` on it (``CUDA_VISIBLE_DEVICES`` = its
   card); ``auto_label.py`` then runs on the CPU (``--label-jobs`` at once) while the card goes on
   with the next scene.
4. Resumable: scenes with ``<runs>/<scene>/voxel/results.pkl`` are not run again, scenes with
   ``<labels>/<scene>/labels.pkl`` not labelled again (``--force`` redoes them). Each step logs to
   ``run_sequence.log`` / ``auto_label.log`` next to its output; a failed scene is reported and
   the others go on.
5. ``<labels>/batch_summary.csv``: per scene the card, seconds, frames, frames left out (pose not
   trusted), boxes, tracks and tracks flagged for review.
"""
import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"]
APPS = ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"]
BUS = ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"]


def parse_gpus(text):
    """nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu (csv, no header,
    no units) -> [dict(index, name, mem_used, mem_total, util)]."""
    out = []
    for line in text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 5:
            continue
        num = lambda x: float(x) if re.match(r"^[0-9.]+$", x) else float("nan")   # noqa: E731  "[N/A]"
        out.append(dict(index=int(f[0]), name=f[1], mem_used=num(f[2]), mem_total=num(f[3]), util=num(f[4])))
    return out


def pick_gpus(gpus, max_mem_mb=16000.0, max_util=20.0, busy=(), visible=None, min_free_mb=8192.0):
    """Indices of the free cards: at least ``min_free_mb`` free, at most ``max_mem_mb`` in use,
    utilisation at most ``max_util`` (unknown utilisation counts as busy). A compute process
    (``busy``) does not by itself exclude the card. ``visible`` limits the choice."""
    del busy  # presence is printed by the caller; memory left and utilisation decide
    free = []
    for g in gpus:
        if visible is not None and g["index"] not in visible:
            continue
        free_mb = g["mem_total"] - g["mem_used"]
        if not (g["mem_used"] <= max_mem_mb and free_mb >= min_free_mb and g["util"] <= max_util):
            continue
        free.append(g["index"])
    return free


def query_gpus():
    """(gpus, indices with a compute process) from nvidia-smi; ([], set()) without it."""
    try:
        gpus = parse_gpus(subprocess.run(QUERY, check=True, capture_output=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        return [], set()
    busy = set()
    try:
        bus = {}
        for line in subprocess.run(BUS, check=True, capture_output=True, text=True).stdout.strip().splitlines():
            i, b = [x.strip() for x in line.split(",")[:2]]
            bus[b.lower()] = int(i)
        for line in subprocess.run(APPS, check=True, capture_output=True, text=True).stdout.strip().splitlines():
            b = line.split(",")[0].strip().lower()
            if b in bus:
                busy.add(bus[b])
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return gpus, busy


def visible_devices():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None or not v.strip():
        return None
    try:
        return {int(x) for x in v.split(",") if x.strip()}
    except ValueError:                             # UUIDs: leave the choice to nvidia-smi
        return None


def scenes_from_split(path, seq_root):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") != "drop"]
    return [os.path.join(seq_root, r["name"]) for r in rows]


def scene_size(seq):
    d = os.path.join(seq, "scans")
    return len(os.listdir(d)) if os.path.isdir(d) else 0


LABEL_LINE = re.compile(r"labelled (\d+) frames \((\d+) left out[^)]*\), (\d+) boxes, (\d+) tracks \((\d+) flagged")


def parse_label_line(text):
    m = LABEL_LINE.search(text or "")
    if not m:
        return {}
    return dict(zip(("frames", "left_out", "boxes", "tracks", "review"), map(int, m.groups())))


def run_batch(scenes, slots, infer, label, label_jobs=2, log=print):
    """Run ``infer(scene, slot)`` on the slots in parallel (each slot one scene at a time) and
    ``label(scene)`` after it on ``label_jobs`` CPU threads. Both return (ok, info dict).
    Returns {scene: dict(slot, ok, infer_s, label_s, ...)}."""
    todo = queue.Queue()
    for s in scenes:
        todo.put(s)
    results, lock = {}, threading.Lock()
    pool = ThreadPoolExecutor(max(1, label_jobs))
    pending = []

    def do_label(scene):
        t0 = time.time()
        ok, info = label(scene)
        with lock:
            results[scene].update(info, label_ok=ok, label_s=round(time.time() - t0, 1))
        log("%-26s labelled %s" % (os.path.basename(scene), "ok" if ok else "FAILED"))

    def worker(slot):
        while True:
            try:
                scene = todo.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            log("%-26s -> %s" % (os.path.basename(scene), slot))
            ok, info = infer(scene, slot)
            with lock:
                results[scene] = dict(info, slot=slot, infer_ok=ok, infer_s=round(time.time() - t0, 1))
            log("%-26s inference %s on %s in %.0f s" % (os.path.basename(scene), "ok" if ok else "FAILED", slot,
                                                         time.time() - t0))
            if ok:
                with lock:
                    pending.append(pool.submit(do_label, scene))

    threads = [threading.Thread(target=worker, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for f in list(pending):
        f.result()
    pool.shutdown()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=None, help="split.csv (rows not 'drop'); or --seq")
    ap.add_argument("--seq", nargs="*", default=None, help="sequence directories instead of --split")
    ap.add_argument("--seq-root", default="data/seq")
    ap.add_argument("--runs", default="data/runs", help="run_sequence.py outputs: <runs>/<scene>")
    ap.add_argument("--labels", default="data/labels", help="auto_label.py outputs: <labels>/<scene>/labels.pkl")
    ap.add_argument("--ckpt-voxel", default="work_dirs/pretrained/nusc_voxel.pth")
    ap.add_argument("--config-voxel", default=None)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--gpus", default="auto", help="'auto' (free cards), a list like 0,2, or 'cpu'")
    ap.add_argument("--per-gpu", type=int, default=1, help="workers per card")
    ap.add_argument("--max-mem-mb", type=float, default=16000.0, help="auto: a card using more memory than this is busy")
    ap.add_argument("--min-free-mb", type=float, default=8192.0, help="auto: a card with less than this free is busy")
    ap.add_argument("--max-util", type=float, default=20.0, help="auto: a card busier than this [%%] is busy")
    ap.add_argument("--label-jobs", type=int, default=0, help="auto_label.py at once (0: cores // 4, at least 1)")
    ap.add_argument("--force", action="store_true", help="redo scenes that already have results / labels")
    ap.add_argument("--dry-run", action="store_true", help="show the cards and the plan, run nothing")
    args = ap.parse_args()

    if args.seq:
        scenes = list(args.seq)
    elif args.split:
        scenes = scenes_from_split(args.split, args.seq_root)
    else:
        ap.error("need --split or --seq")
    scenes = sorted(scenes, key=scene_size, reverse=True)

    if args.gpus == "cpu":
        cards = ["cpu"]
    elif args.gpus == "auto":
        gpus, busy = query_gpus()
        if not gpus:
            raise SystemExit("nvidia-smi found no GPU (use --gpus cpu to run on the CPU)")
        vis = visible_devices()
        free = pick_gpus(gpus, args.max_mem_mb, args.max_util, busy, vis, args.min_free_mb)
        print("%-4s %-28s %10s %6s %6s  %s" % ("gpu", "name", "memory MB", "util", "procs", "use"))
        for g in gpus:
            print("%-4d %-28s %5.0f/%-5.0f %5.0f%% %6s  %s" % (
                g["index"], g["name"][:28], g["mem_used"], g["mem_total"], g["util"],
                "yes" if g["index"] in busy else "-", "FREE" if g["index"] in free else "busy"))
        if not free:
            raise SystemExit("no free GPU (at most %.0f MB in use, at least %.0f MB free, at most %.0f%% busy)" % (
                args.max_mem_mb, args.min_free_mb, args.max_util))
        cards = ["cuda:%d" % i for i in free]
    else:
        cards = ["cuda:%d" % int(x) for x in args.gpus.split(",") if x.strip()]
    slots = ["%s#%d" % (c, k) for c in cards for k in range(max(1, args.per_gpu))]
    label_jobs = args.label_jobs or max(1, (os.cpu_count() or 1) // 4)
    print("%d scenes on %d worker(s): %s; auto_label %d at once" % (len(scenes), len(slots), " ".join(slots),
                                                                    label_jobs))
    if args.dry_run:
        for k, s in enumerate(scenes):
            print("  %-26s %4d scans  -> %s (first round)" % (os.path.basename(s), scene_size(s), slots[k % len(slots)])
                  if k < len(slots) else "  %-26s %4d scans  -> next free worker" % (os.path.basename(s), scene_size(s)))
        return

    def infer(scene, slot):
        name = os.path.basename(os.path.normpath(scene))
        out = os.path.join(args.runs, name)
        if not args.force and os.path.exists(os.path.join(out, "voxel", "results.pkl")):
            return True, dict(skipped_inference=True)
        os.makedirs(out, exist_ok=True)
        card = slot.split("#")[0]
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, os.path.join(ROOT, "tools/dog/run_sequence.py"), "--seq", scene, "--detectors", "voxel",
               "--ckpt-voxel", args.ckpt_voxel, "--save-frames", "--out", out]
        if args.config_voxel:
            cmd += ["--config-voxel", args.config_voxel]
        if not args.no_tta:
            cmd += ["--tta"]
        if card == "cpu":
            cmd += ["--device", "cpu"]
        else:
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=card.split(":")[1])
            cmd += ["--device", "cuda:0"]
        with open(os.path.join(out, "run_sequence.log"), "w") as fh:
            ok = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode == 0
        return ok, {}

    def label(scene):
        name = os.path.basename(os.path.normpath(scene))
        out = os.path.join(args.labels, name)
        dst = os.path.join(out, "labels.pkl")
        logf = os.path.join(out, "auto_label.log")
        if not args.force and os.path.exists(dst) and os.path.exists(logf):
            return True, dict(parse_label_line(open(logf).read()), skipped_label=True)
        os.makedirs(out, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, os.path.join(ROOT, "tools/dog/auto_label.py"),
               "--results", os.path.join(args.runs, name, "voxel", "results.pkl"), "--out", dst]
        with open(logf, "w") as fh:
            ok = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode == 0
        return ok, parse_label_line(open(logf).read())

    t0 = time.time()
    res = run_batch(scenes, slots, infer, label, label_jobs)
    os.makedirs(args.labels, exist_ok=True)
    cols = ["scene", "slot", "infer_ok", "infer_s", "label_ok", "label_s", "frames", "left_out", "boxes", "tracks",
            "review"]
    with open(os.path.join(args.labels, "batch_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in scenes:
            w.writerow(dict(res.get(s, {}), scene=os.path.basename(os.path.normpath(s))))
    failed = [os.path.basename(s) for s in scenes if not res.get(s, {}).get("infer_ok")
              or not res.get(s, {}).get("label_ok")]
    print("%d scenes in %.0f s on %d worker(s); %d failed%s -> %s" % (
        len(scenes), time.time() - t0, len(slots), len(failed), (": " + " ".join(failed)) if failed else "",
        os.path.join(args.labels, "batch_summary.csv")))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
