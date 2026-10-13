# SUSTechPOINTS (vendored)

3D point-cloud annotation tool used to correct auto-labels box by box and to add missed objects
(docs/DOG_FINETUNE.md, section 10). Kept in this repository so it runs after a plain `git pull` and can
be changed together with the converter (`tools/dog/sustechpoints_convert.py`).

- Upstream: https://github.com/naurril/SUSTechPOINTS, branch `dev-auto-annotate`
- Version: commit `50fa18871a6d46507ff9057175669235885f79f9` (2024-06-13, the latest when vendored on 2026-10-13)
- License: GPL-3.0 (`LICENSE`, unchanged). This directory stays under GPL-3.0; the rest of the repository
  only exchanges files with it (PCD / JSON) or starts it as a separate program.

## Left out (fetch from upstream if needed)

- `data/example/` (15 MB sample scene): our scenes are exported into `data/` instead
- images and animations under `doc/` (13 MB), referenced by the README files; the README and doc `.md`
  files are kept
- the auto-annotation model `algos/models/deep_annotation_inference.h5` is not in upstream either
  (`algos/models/readme.txt`)

## Local changes

| date | file | change |
|---|---|---|
| 2026-10-13 | `main.py` | TensorFlow / the model file are optional: without them the server starts, `/predict_rotation` (used when a new box snaps to its points) returns the yaw of the points' principal axis (numpy, 180 deg ambiguous: `g` flips) and `/auto_annotate` answers 503; everything else works |
| 2026-10-13 | `.gitignore` | `data/*` instead of `data`, so `data/README.md` is kept; exported scenes stay ignored |
| 2026-10-13 | `data/README.md` | new: what goes into `data/` |
| 2026-10-13 | `scene_reader.py` | only directories in `data/` are scenes (otherwise `data/README.md` shows up as one) |

Mark further changes in this table and in the changed file (GPL-3.0 section 5a asks for a notice with
the date).

## Run

```bash
pip install cherrypy jinja2            # TensorFlow only for auto-annotation
python tools/dog/sustechpoints_convert.py serve          # from the repository root, http://127.0.0.1:8081
# or, as upstream: cd third_party/SUSTechPOINTS && python main.py   (port 8081, all interfaces)
```

## Updating from upstream

```bash
git clone https://github.com/naurril/SUSTechPOINTS /tmp/sustechpoints
diff -ru --exclude=.git --exclude=data /tmp/sustechpoints third_party/SUSTechPOINTS   # review
```

Copy the upstream changes over, re-apply the local changes above, update the commit here, then check
the round trip: `python -m pytest tests/dog/test_sustechpoints.py` and an export / serve / import by
hand (docs/DOG_FINETUNE.md, section 10).
