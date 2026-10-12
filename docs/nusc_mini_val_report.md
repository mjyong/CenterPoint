# nuScenes clip evaluation

- data: v1.0-mini, scenes scene-0103, scene-0916 (393 frames at 10 Hz, 81 keyframes evaluated)
- detector: `work_dirs/pretrained/nusc_pp.pth` (`configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py`), classes merged to vehicle / pedestrian / cyclist
- matching: center distance in BEV; GT counted when it has lidar points and lies within the class range (vehicle 50 m, pedestrian / cyclist 40 m), predictions and tracks filtered the same way

## Detection

AP over the detector output (score >= 0.1) at 0.5 / 1 / 2 / 4 m; error metrics on true positives at 2 m (ATE m, ASE 1-IoU, AOE rad, AVE m/s).

| class | AP@0.5 | AP@1 | AP@2 | AP@4 | mAP | ATE | ASE | AOE | AVE | AP@2 0-20m | 20-30m | 30-50m | #GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vehicle | 0.661 | 0.702 | 0.710 | 0.727 | **0.700** | 0.22 | 0.17 | 0.27 | 0.18 | 0.935 | 0.728 | 0.341 | 2001 |
| pedestrian | 0.751 | 0.766 | 0.801 | 0.844 | **0.790** | 0.21 | 0.27 | 0.46 | 0.24 | 0.866 | 0.775 | 0.545 | 1057 |
| cyclist | 0.445 | 0.489 | 0.510 | 0.547 | **0.498** | 0.38 | 0.27 | 1.08 | 0.16 | 0.701 | 0.320 | 0.226 | 246 |

mAP over the three classes: **0.663**

At the reporting thresholds (vehicle 0.35, pedestrian 0.30, cyclist 0.40):

| class | precision | recall | TP | FP | #GT |
|---|---|---|---|---|---|
| vehicle | 0.956 | 0.703 | 1406 | 64 | 2001 |
| pedestrian | 0.795 | 0.798 | 843 | 218 | 1057 |
| cyclist | 0.737 | 0.411 | 101 | 36 | 246 |

## Tracking

Reported tracks (confirmed, coasting included) vs GT instances, 2 m.

| class | MOTA | MOTP (m) | IDF1 | recall | precision | IDSW | FRAG | MT | ML | GT tracks | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vehicle | **0.689** | 0.24 | 0.792 | 0.767 | 0.924 | 29 | 36 | 51 | 16 | 119 | 1534 | 126 | 467 |
| pedestrian | **0.531** | 0.23 | 0.771 | 0.883 | 0.722 | 12 | 6 | 51 | 3 | 67 | 933 | 360 | 124 |
| cyclist | **0.398** | 0.41 | 0.633 | 0.516 | 0.819 | 1 | 7 | 5 | 4 | 17 | 127 | 28 | 119 |
| all | **0.617** | 0.24 | 0.732 | 0.785 | 0.835 | 42 | 49 | 107 | 23 | 203 | 2594 | 514 | 710 |

### Where the tracking errors come from

False tracks (FP) by cause:

| cause | vehicle | pedestrian | cyclist |
|---|---|---|---|
| second track on a tracked object (< 1 m from its track) | 0 | 39 | 0 |
| next to a tracked object of its class | 0 | 85 | 0 |
| other class on a GT object | 8 | 14 | 5 |
| on a GT that is not evaluated | 47 | 8 | 2 |
| coasting after the object was lost | 40 | 74 | 5 |
| no object nearby | 31 | 140 | 16 |

Missed GT objects (FN): was there a reported detection within 2 m? How far, how many lidar points?

| | vehicle | pedestrian | cyclist |
|---|---|---|---|
| detected, not tracked | 51 | 71 | 23 |
| not detected | 416 | 53 | 96 |
| 0-20 m | 25 | 32 | 42 |
| 20-30 m | 116 | 50 | 51 |
| 30-50 m | 326 | 42 | 26 |
| 1-9 pts | 440 | 116 | 89 |
| 10-49 pts | 24 | 7 | 29 |
| >=50 pts | 3 | 1 | 1 |

## Prediction

Tracks matched to a GT object with 3 s of annotated future; 6 steps of 0.5 s. IMM = tier 1 (all modes with their probabilities), CV = constant velocity from the track. minADE / minFDE: best mode; ADE1 / FDE1: most likely mode; MR: no mode ends within 2 m.

**all objects** (1913 samples)

| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| IMM | 0.50 | 0.78 | 0.52 | 0.81 | 0.078 |
| CV | 0.52 | 0.82 | 0.52 | 0.82 | 0.084 |

**moving objects (>= 1.5 m in 3 s)** (554 samples)

| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| IMM | 0.68 | 1.28 | 0.73 | 1.36 | 0.166 |
| CV | 0.74 | 1.39 | 0.74 | 1.39 | 0.186 |

**vehicle, moving** (71 samples)

| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| IMM | 2.40 | 4.92 | 2.47 | 5.11 | 0.831 |
| CV | 2.65 | 5.46 | 2.65 | 5.46 | 0.887 |

**pedestrian, moving** (478 samples)

| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| IMM | 0.43 | 0.74 | 0.47 | 0.80 | 0.069 |
| CV | 0.46 | 0.79 | 0.46 | 0.79 | 0.082 |

**cyclist, moving** (5 samples)

| predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| IMM | 0.71 | 1.17 | 0.78 | 1.33 | 0.000 |
| CV | 0.77 | 1.32 | 0.77 | 1.32 | 0.200 |

## Snapshots

Bird's-eye view at the middle keyframe of each scene (ego at the center, heading up). Green / red outlines: GT tracked / missed; filled boxes: reported tracks (by class) with their id; colored lines: the two most likely predicted modes (opacity = probability); black dashes: GT 3 s future.

![scene-0103_bev.png](scene-0103_bev.png)
![scene-0916_bev.png](scene-0916_bev.png)

## Timing (median ms per frame, this machine)

| stage | ms |
|---|---|
| load | 47.1 |
| det_network | 1585.7 |
| detect | 1699.1 |
| track | 36.7 |
| predict | 32.2 |
| visualize | 157.7 |

## Output

- `work_dirs/nusc_eval/scene-0103.mcap`: 2138 messages, 105 MB
- `work_dirs/nusc_eval/scene-0916.mcap`: 2239 messages, 115 MB

Open the MCAP in Foxglove: 3D panel (follow frame `ego`), Image panel on `/cam/CAM_FRONT/image` with `/cam/CAM_FRONT/annotations`, Plot panel on `/metrics/frame.mota_cum`, Log panel on `/log`.
