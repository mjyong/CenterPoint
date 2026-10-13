Scenes go here: one directory per scene with `lidar/<frame>.pcd` and `label/<frame>.json`.

    python tools/dog/sustechpoints_convert.py export --labels <labels.pkl> --frames <run>/frames --scene rec_001

writes `rec_001/` here. Scenes are not committed (see the repository's .gitignore).
