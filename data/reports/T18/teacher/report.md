# Pipeline evaluation

- scenes (10): 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- 2884 frames, 580 keyframes scored (4.8 min)
- detections: `results` from `data/runs/{scene}/voxel/results.pkl`
- parameters: defaults
- scored: vehicles that are not ignore boxes, within 40 m and with current-sweep points; a detection or track on any other labelled box is not counted. Label tracks stitched into one vehicle: 19.
- intervals: 90 % over 500 resamples of the scenes

## Main metrics

| metric | value | interval |
|---|---|---|
| detections: recall within near range | **0.898** | 0.826 – 0.922 |
| detections: precision within near range | **0.948** | 0.875 – 0.970 |
| detections: F1 within near range | **0.922** | 0.851 – 0.943 |
| detections: false per keyframe | **0.217** | 0.161 – 0.280 |
| tracks: MOTA | **0.812** | 0.660 – 0.867 |
| tracks: IDF1 | **0.880** | 0.793 – 0.915 |
| tracks: ID switches / min | **1.655** | 0.620 – 3.125 |
| tracks: false tracks / min | **13.447** | 8.750 – 18.155 |
| tracks: confirm delay p90 [s] | **0.500** | 0.400 – 1.200 |
| motion: relevant moving stretches missed | **1** | 0.000 – 3.000 |
| motion: parked reported moving / h | **24.826** | 0.000 – 49.823 |
| prediction: FDE 3 s, moving [m] | **1.958** | 1.657 – 2.528 |
| prediction: miss rate 3 s, moving | **0.393** | 0.298 – 0.569 |

## Detections at the reporting threshold

Vehicle threshold 0.35 (`tracker.high_score`), 2 m match, keyframes. Near: within 30 m.

| range | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0-30 m | 0.948 | 0.898 | 1247 | 68 | 142 |
| 30-40 m | 0.794 | 0.651 | 224 | 58 | 120 |

False detections per keyframe 0.217. AP over all scores: AP@2.0 0.927, AP@4.0 0.937, AP@2.0 0-20 m 0.977, 20-40 m 0.897.

## Tracking

Reported tracks (confirmed, coasting included) against the scored vehicles, keyframes, 2 m.

| MOTA | MOTP m | IDF1 | recall | precision | IDSW | IDSW / min | FRAG | MT | ML | vehicles | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.812** | 0.20 | **0.880** | 0.887 | 0.926 | 8 | 1.66 | 26 | 39 | 4 | 59 | 1537 | 122 | 196 |

- false tracks (a track false on consecutive keyframes counts once): 13.45 per minute
- a vehicle coming into 30 m waits for its first matched track: median 0.0 s, p90 0.5 s (17 vehicles; 1 never tracked; 0.5 s resolution)

### Where the errors are

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 2 |
| beside a vehicle (2-4 m off) | 6 |
| coasting after the vehicle was lost | 6 |
| nothing labelled nearby | 108 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 41 |
| not detected | 155 |
| 0-10 m | 3 |
| 10-20 m | 19 |
| 20-30 m | 77 |
| 30-40 m | 97 |
| 1-9 pts | 14 |
| 10-49 pts | 52 |
| >=50 pts | 130 |

Blind sector (142-218 deg behind the robot) and the 20 deg bands beside it:

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 1317 | 127 | 0.096 | 1296 | 104 | 0.080 |
| edge | 402 | 67 | 0.167 | 347 | 13 | 0.037 |
| blind | 14 | 2 | 0.143 | 16 | 5 | 0.312 |

## Motion state

Online state against the labels' moving flags (tools/dog/motion_eval.py). Relevant: >= 1 m/s and within 30 m.

| moving recall | false moving | parked reported moving | false episodes (per h) | moving stretches missed | latency median / p90 s | relevant missed | relevant latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.725 | 0.0053 | 2 / 40 | 2 (24.8) | 9 / 21 | 0.50 / 1.45 | **1 / 8** | 0.50 / 1.40 |

## Prediction

Tracks matched to a vehicle with 3 s of labelled future (keyframes every 0.5 s). Moving: >= 1.5 m in 3 s; turning: heading changes > 15 deg. ADE1 / FDE1: the most likely mode; minADE / minFDE: the best; MR: no mode ends within 2 m. IMM is what runs online, CV keeps the track's velocity.

**all matched** (1205 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.29 | 0.33 | 0.29 | 0.33 | 0.015 |
| IMM | 2 s | 0.37 | 0.49 | 0.37 | 0.49 | 0.051 |
| IMM | 3 s | 0.45 | 0.66 | 0.45 | 0.66 | 0.085 |
| CV | 1 s | 0.29 | 0.33 | 0.29 | 0.33 | 0.015 |
| CV | 2 s | 0.37 | 0.49 | 0.37 | 0.49 | 0.051 |
| CV | 3 s | 0.45 | 0.66 | 0.45 | 0.66 | 0.085 |

**moving** (135 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.62 | 0.75 | 0.62 | 0.75 | 0.037 |
| IMM | 2 s | 0.91 | 1.35 | 0.91 | 1.35 | 0.230 |
| IMM | 3 s | 1.21 | 1.96 | 1.21 | 1.96 | 0.393 |
| CV | 1 s | 0.62 | 0.75 | 0.62 | 0.75 | 0.037 |
| CV | 2 s | 0.91 | 1.35 | 0.91 | 1.35 | 0.230 |
| CV | 3 s | 1.21 | 1.96 | 1.21 | 1.96 | 0.393 |

**moving, straight** (120 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.63 | 0.76 | 0.63 | 0.76 | 0.033 |
| IMM | 2 s | 0.91 | 1.33 | 0.91 | 1.33 | 0.233 |
| IMM | 3 s | 1.20 | 1.94 | 1.20 | 1.94 | 0.383 |
| CV | 1 s | 0.63 | 0.76 | 0.63 | 0.76 | 0.033 |
| CV | 2 s | 0.91 | 1.33 | 0.91 | 1.33 | 0.233 |
| CV | 3 s | 1.20 | 1.94 | 1.20 | 1.94 | 0.383 |

**moving, turning** (15 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.55 | 0.68 | 0.55 | 0.68 | 0.067 |
| IMM | 2 s | 0.92 | 1.50 | 0.92 | 1.50 | 0.200 |
| IMM | 3 s | 1.28 | 2.12 | 1.28 | 2.12 | 0.467 |
| CV | 1 s | 0.55 | 0.68 | 0.55 | 0.68 | 0.067 |
| CV | 2 s | 0.92 | 1.50 | 0.92 | 1.50 | 0.200 |
| CV | 3 s | 1.28 | 2.13 | 1.28 | 2.13 | 0.467 |

**moving, < 3 m/s** (125 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.63 | 0.77 | 0.63 | 0.77 | 0.040 |
| IMM | 2 s | 0.93 | 1.38 | 0.93 | 1.38 | 0.248 |
| IMM | 3 s | 1.24 | 2.01 | 1.24 | 2.01 | 0.408 |
| CV | 1 s | 0.63 | 0.77 | 0.63 | 0.77 | 0.040 |
| CV | 2 s | 0.93 | 1.38 | 0.93 | 1.38 | 0.248 |
| CV | 3 s | 1.24 | 2.01 | 1.24 | 2.01 | 0.408 |

**moving, >= 6 m/s** (10 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.44 | 0.55 | 0.44 | 0.55 | 0.000 |
| IMM | 2 s | 0.65 | 0.95 | 0.65 | 0.95 | 0.000 |
| IMM | 3 s | 0.83 | 1.30 | 0.83 | 1.30 | 0.200 |
| CV | 1 s | 0.44 | 0.55 | 0.44 | 0.55 | 0.000 |
| CV | 2 s | 0.64 | 0.95 | 0.64 | 0.95 | 0.000 |
| CV | 3 s | 0.83 | 1.30 | 0.83 | 1.30 | 0.200 |

## Hazard proxy

A moving vehicle that comes within 5 m of the robot within 3 s (labelled future, the robot's own path) should raise an alarm: a moving track whose most likely predicted path, or its position now, comes that close. Few such vehicles in 0921; the numbers are for the controlled scenes to come (DATA_COLLECTION_PLAN.md 3.2).

- events 6, reported 2, lead before the closest approach: median 1.5 s, min 1.0 s
- alarms without such a vehicle: 0 (0.0 per hour)

## Timing (this machine, ms per frame: median / p90)

| stage | median | p90 |
|---|---|---|
| filter | 0.00 | 0.00 |
| track | 1.78 | 6.73 |
| predict | 1.04 | 3.88 |

## Issues (`issues.csv`)

| kind | count |
|---|---|
| ID switch | 8 |
| bad prediction | 24 |
| false moving | 2 |
| false track | 23 |
| hazard not reported | 4 |
| lost vehicle | 14 |
| missed moving vehicle | 1 |

## Snapshots

Middle keyframe of each scene, robot at the center heading up. Green / red outlines: scored vehicles tracked / missed; grey: not scored; filled: reported tracks; lines: the two most likely predicted modes; dashes: labelled 3 s future; shaded: the blind sector.

![0921_164127_s000_bev.png](0921_164127_s000_bev.png)
![0921_164127_s001_bev.png](0921_164127_s001_bev.png)
![0921_164127_s002_bev.png](0921_164127_s002_bev.png)
![0921_164127_s003_bev.png](0921_164127_s003_bev.png)
![0921_164127_s004_bev.png](0921_164127_s004_bev.png)
![0921_164127_s005_bev.png](0921_164127_s005_bev.png)
![0921_165659_s003_bev.png](0921_165659_s003_bev.png)
![0921_165659_s004_bev.png](0921_165659_s004_bev.png)
![0921_165659_s005_bev.png](0921_165659_s005_bev.png)
![0921_165659_s006_bev.png](0921_165659_s006_bev.png)
