"""Warm-start a dog model from a 10-class nuScenes CenterPoint checkpoint.

Instead of re-initialising the heads, each dog task inherits the nuScenes
task head that already knows the object (all regression branches + the
matching heat-map channel):

    car        <- task 0 (car)                           hm channel 0
    truck      <- task 1 (truck, construction_vehicle)   hm channel 0 (truck)
    bus        <- task 2 (bus, trailer)                  hm channel 0 (bus)
    pedestrian <- task 5 (pedestrian, cone)              hm channel 0
    cyclist    <- task 4 (motorcycle, bike)              hm channel 1 (bicycle)
    vehicle    <- task 0 (car)        older 3-task layout (configs/dog/*_3cls.py)

The dog tasks (one class each) are read from ``--config``. Backbone / neck /
shared conv are copied unchanged. Pillar and Voxel nuScenes models have the
same head layout.

    python tools/dog/convert_nusc_ckpt.py --src nusc_pp.pth --dst work_dirs/dog_pp_init.pth \
        [--config configs/dog/dog_centerpoint_pp_xt32.py]
"""
import argparse
import os
import re
import sys

import torch

# dog class -> (nuScenes task index, heat-map channel)
NUSC_HEAD = {"car": (0, 0), "truck": (1, 0), "bus": (2, 0), "pedestrian": (5, 0), "cyclist": (4, 1),
             "vehicle": (0, 0)}


def mapping_for(task_class_names):
    """[[class] per dog task] -> {dog task index: (nuScenes task index, [heat-map channels])}."""
    out = {}
    for i, names in enumerate(task_class_names):
        if len(names) != 1 or names[0] not in NUSC_HEAD:
            raise ValueError("no nuScenes head for dog task %d %s" % (i, names))
        src, ch = NUSC_HEAD[names[0]]
        out[i] = (src, [ch])
    return out


DEFAULT_MAPPING = mapping_for([["car"], ["truck"], ["bus"], ["pedestrian"], ["cyclist"]])


def convert_state_dict(sd, mapping=None):
    mapping = mapping or DEFAULT_MAPPING
    out = {}
    pat = re.compile(r"^bbox_head\.tasks\.(\d+)\.(\w+)\.(\d+)\.(.+)$")
    last_hm = {}
    for k in sd:
        m = pat.match(k)
        if m and m.group(2) == "hm":
            t, i = int(m.group(1)), int(m.group(3))
            last_hm[t] = max(last_hm.get(t, -1), i)
    for k, v in sd.items():
        if not k.startswith("bbox_head.tasks."):
            out[k] = v
    for dst, (src, chans) in mapping.items():
        prefix = "bbox_head.tasks.%d." % src
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            nk = "bbox_head.tasks.%d." % dst + k[len(prefix):]
            m = pat.match(k)
            if m and m.group(2) == "hm" and int(m.group(3)) == last_hm[src] and m.group(4) in ("weight", "bias"):
                v = v[chans].clone()
            out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--config", default="configs/dog/dog_centerpoint_pp_xt32.py",
                    help="dog config whose head tasks to build (the *_3cls configs for the older layout)")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from det3d.torchie import Config
    tasks = [t["class_names"] for t in Config.fromfile(args.config).model.bbox_head.tasks]
    try:
        ckpt = torch.load(args.src, map_location="cpu", weights_only=False)   # det3d checkpoints hold numpy meta
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    new = convert_state_dict(sd, mapping_for(tasks))
    torch.save({"state_dict": new, "meta": {"converted_from": args.src, "tasks": tasks}}, args.dst)
    print("saved %s (%d tensors, tasks %s)" % (args.dst, len(new), ", ".join("/".join(t) for t in tasks)))


if __name__ == "__main__":
    main()
