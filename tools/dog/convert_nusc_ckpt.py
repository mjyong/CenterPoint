"""Warm-start a 3-class dog model from a 10-class nuScenes CenterPoint checkpoint.

Instead of re-initialising the heads, each dog task inherits the nuScenes
task head that already knows the object (all regression branches + the
matching heat-map channel):

    vehicle    <- task 0 (car)               hm channel 0
    pedestrian <- task 5 (pedestrian, cone)  hm channel 0
    cyclist    <- task 4 (motorcycle, bike)  hm channel 1 (bicycle)

Backbone / neck / shared conv are copied unchanged.

    python tools/dog/convert_nusc_ckpt.py --src nusc_pp.pth --dst work_dirs/dog_pp_init.pth
"""
import argparse
import re

import torch

# dog task index -> (nuScenes task index, heat-map channels to keep)
DEFAULT_MAPPING = {0: (0, [0]), 1: (5, [0]), 2: (4, [1])}


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
    args = ap.parse_args()
    ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    new = convert_state_dict(sd)
    torch.save({"state_dict": new, "meta": {"converted_from": args.src}}, args.dst)
    print("saved %s (%d tensors)" % (args.dst, len(new)))


if __name__ == "__main__":
    main()
