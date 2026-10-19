"""tools/dog/batch_label.py: free-GPU choice from nvidia-smi and the worker / CPU-pool schedule."""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dog"))

import batch_label as bl  # noqa: E402

SMI = """0, NVIDIA GeForce RTX 4090, 23500, 24564, 98
1, NVIDIA GeForce RTX 4090, 310, 24564, 0
2, NVIDIA GeForce RTX 4090, 1200, 24564, 3
3, NVIDIA GeForce RTX 4090, 280, 24564, [N/A]
"""


def test_free_gpus_from_nvidia_smi():
    gpus = bl.parse_gpus(SMI)
    assert [g["index"] for g in gpus] == [0, 1, 2, 3] and gpus[0]["mem_used"] == 23500
    assert bl.pick_gpus(gpus) == [1, 2]                              # 0 full and busy, 3 utilisation unknown
    assert bl.pick_gpus(gpus, busy={2}) == [1, 2]                    # a resident process does not exclude 2
    assert bl.pick_gpus(gpus, max_mem_mb=1000) == [1]
    assert bl.pick_gpus(gpus, visible={2, 3}) == [2]                 # CUDA_VISIBLE_DEVICES already set
    shared = bl.parse_gpus("4, NVIDIA GeForce RTX 4090, 14000, 23028, 0\n"
                           "5, NVIDIA GeForce RTX 4090, 21000, 23028, 0\n")
    assert bl.pick_gpus(shared, busy={4, 5}) == [4]                  # 5 has under 8 GB free


def test_run_batch_keeps_every_worker_busy_and_labels_on_the_side():
    seen, lock = [], threading.Lock()

    def infer(scene, slot):
        time.sleep(0.05)
        with lock:
            seen.append((scene, slot))
        return scene != "bad", {}

    def label(scene):
        time.sleep(0.02)
        return True, dict(frames=10, boxes=3)

    scenes = ["s%d" % k for k in range(7)] + ["bad"]
    t0 = time.time()
    res = bl.run_batch(scenes, ["cuda:1#0", "cuda:2#0"], infer, label, label_jobs=2, log=lambda *a: None)
    assert time.time() - t0 < 0.05 * 8 * 0.75                       # two cards in parallel
    assert {s for s, _ in seen} == set(scenes) and {w for _, w in seen} == {"cuda:1#0", "cuda:2#0"}
    assert all(res[s]["label_ok"] and res[s]["frames"] == 10 for s in scenes if s != "bad")
    assert not res["bad"]["infer_ok"] and "label_ok" not in res["bad"]


def test_label_line_and_dry_run(tmp_path):
    line = ("labelled 302 frames (5 left out: pose not trusted), 1840 boxes, 37 tracks (6 flagged for review) "
            "-> x/labels.pkl, x/labels_tracks.csv")
    assert bl.parse_label_line(line) == dict(frames=302, left_out=5, boxes=1840, tracks=37, review=6)
    seqs = []
    for k, n in enumerate((3, 9, 5)):
        d = tmp_path / ("sc%d" % k) / "scans"
        d.mkdir(parents=True)
        for i in range(n):
            (d / ("%06d.npz" % i)).write_bytes(b"")
        seqs.append(str(d.parent))
    out = subprocess.run([sys.executable, "tools/dog/batch_label.py", "--seq", *seqs, "--gpus", "0,3", "--per-gpu", "2",
                          "--dry-run"], check=True, capture_output=True, text=True).stdout
    assert "3 scenes on 4 worker(s): cuda:0#0 cuda:0#1 cuda:3#0 cuda:3#1" in out
    assert out.index("sc1") < out.index("sc2") < out.index("sc0")       # longest first
