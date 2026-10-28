# Pipeline evaluation

- scenes (25): 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- 7479 frames, 1476 keyframes scored (12.3 min)
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- parameters: defaults
- scored: vehicles that are not ignore boxes, within 40 m and with current-sweep points; a detection or track on any other labelled box is not counted. Label tracks stitched into one vehicle: 80.
- intervals: 90 % over 0 resamples of the scenes

## Main metrics

| metric | value | interval |
|---|---|---|
| detections: recall within near range | **0.808** | - |
| detections: precision within near range | **0.809** | - |
| detections: F1 within near range | **0.808** | - |
| detections: false per keyframe | **0.497** | - |
| tracks: MOTA | **0.658** | - |
| tracks: IDF1 | **0.785** | - |
| tracks: ID switches / min | **2.439** | - |
| tracks: false tracks / min | **22.193** | - |
| tracks: confirm delay p90 [s] | **1.052** | - |
| motion: relevant moving stretches missed | **0** | - |
| motion: parked reported moving / h | **39.021** | - |
| prediction: FDE 3 s, moving [m] | **2.998** | - |
| prediction: miss rate 3 s, moving | **0.636** | - |

## Detections at the reporting threshold

Vehicle threshold 0.40 (`tracker.high_score`), 2 m match, keyframes. Near: within 30 m.

| range | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0-30 m | 0.809 | 0.808 | 1945 | 460 | 462 |
| 30-40 m | 0.762 | 0.727 | 872 | 273 | 327 |

False detections per keyframe 0.497. AP over all scores: AP@2.0 0.826, AP@4.0 0.839, AP@2.0 0-20 m 0.872, 20-40 m 0.771.

## Tracking

Reported tracks (confirmed, coasting included) against the scored vehicles, keyframes, 2 m.

| MOTA | MOTP m | IDF1 | recall | precision | IDSW | IDSW / min | FRAG | MT | ML | vehicles | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.658** | 0.22 | **0.785** | 0.842 | 0.827 | 30 | 2.44 | 108 | 99 | 15 | 147 | 3038 | 634 | 568 |

- false tracks (a track false on consecutive keyframes counts once): 22.19 per minute
- a vehicle coming into 30 m waits for its first matched track: median 0.0 s, p90 1.1 s (60 vehicles; 8 never tracked; 0.5 s resolution)

### Where the errors are

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 0 |
| beside a vehicle (2-4 m off) | 25 |
| coasting after the vehicle was lost | 65 |
| nothing labelled nearby | 544 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 112 |
| not detected | 456 |
| 0-10 m | 93 |
| 10-20 m | 93 |
| 20-30 m | 170 |
| 30-40 m | 212 |
| 1-9 pts | 93 |
| 10-49 pts | 177 |
| >=50 pts | 298 |

Blind sector (142-218 deg behind the robot) and the 20 deg bands beside it:

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 3192 | 487 | 0.153 | 3290 | 579 | 0.176 |
| edge | 324 | 56 | 0.173 | 317 | 48 | 0.151 |
| blind | 90 | 25 | 0.278 | 65 | 7 | 0.108 |

## Motion state

Online state against the labels' moving flags (tools/dog/motion_eval.py). Relevant: >= 1 m/s and within 30 m.

| moving recall | false moving | parked reported moving | false episodes (per h) | moving stretches missed | latency median / p90 s | relevant missed | relevant latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.875 | 0.0211 | 7 / 119 | 8 (39.0) | 8 / 24 | 0.50 / 0.50 | **0 / 7** | 0.50 / 0.70 |

## Prediction

Tracks matched to a vehicle with 3 s of labelled future (keyframes every 0.5 s). Moving: >= 1.5 m in 3 s; turning: heading changes > 15 deg. ADE1 / FDE1: the most likely mode; minADE / minFDE: the best; MR: no mode ends within 2 m. IMM is what runs online (IMM: tier 1, learned: tier 2), CV keeps the track's velocity. Moving vehicles: the most likely mode's error split against the labelled path, along: mean distance along it from the true position (speed, starting, stopping), bias: its mean signed value (+: ahead, predicted too fast), across: mean distance from the path (its shape: what a road map could fix).

**all matched** (2212 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.33 | 0.37 | 0.33 | 0.37 | 0.024 | - | - | - |
| IMM | 2 s | 0.42 | 0.57 | 0.42 | 0.57 | 0.068 | - | - | - |
| IMM | 3 s | 0.53 | 0.79 | 0.52 | 0.78 | 0.116 | - | - | - |
| CV | 1 s | 0.33 | 0.37 | 0.33 | 0.37 | 0.024 | - | - | - |
| CV | 2 s | 0.42 | 0.57 | 0.42 | 0.57 | 0.071 | - | - | - |
| CV | 3 s | 0.53 | 0.79 | 0.53 | 0.79 | 0.118 | - | - | - |

**moving** (165 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.95 | 1.14 | 0.95 | 1.13 | 0.145 | 1.06 | 0.64 | 0.29 |
| IMM | 2 s | 1.37 | 2.00 | 1.35 | 1.96 | 0.376 | 1.74 | 0.86 | 0.66 |
| IMM | 3 s | 1.83 | 3.00 | 1.79 | 2.91 | 0.636 | 2.50 | 1.01 | 1.09 |
| CV | 1 s | 0.96 | 1.14 | 0.96 | 1.14 | 0.145 | 1.05 | 0.63 | 0.30 |
| CV | 2 s | 1.37 | 2.02 | 1.37 | 2.02 | 0.412 | 1.72 | 0.84 | 0.70 |
| CV | 3 s | 1.84 | 3.03 | 1.84 | 3.03 | 0.661 | 2.46 | 0.96 | 1.18 |

**moving, straight** (123 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.99 | 1.16 | 0.99 | 1.16 | 0.171 | 1.13 | 0.81 | 0.24 |
| IMM | 2 s | 1.37 | 1.95 | 1.36 | 1.93 | 0.341 | 1.78 | 1.21 | 0.52 |
| IMM | 3 s | 1.78 | 2.84 | 1.77 | 2.80 | 0.569 | 2.51 | 1.55 | 0.84 |
| CV | 1 s | 0.99 | 1.16 | 0.99 | 1.16 | 0.171 | 1.13 | 0.81 | 0.25 |
| CV | 2 s | 1.37 | 1.95 | 1.37 | 1.95 | 0.358 | 1.77 | 1.20 | 0.53 |
| CV | 3 s | 1.78 | 2.85 | 1.78 | 2.85 | 0.577 | 2.50 | 1.55 | 0.87 |

**moving, turning** (42 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.85 | 1.07 | 0.83 | 1.03 | 0.071 | 0.83 | 0.14 | 0.45 |
| IMM | 2 s | 1.36 | 2.16 | 1.31 | 2.06 | 0.476 | 1.63 | -0.17 | 1.09 |
| IMM | 3 s | 1.95 | 3.46 | 1.85 | 3.21 | 0.833 | 2.48 | -0.60 | 1.84 |
| CV | 1 s | 0.86 | 1.08 | 0.86 | 1.08 | 0.071 | 0.82 | 0.12 | 0.48 |
| CV | 2 s | 1.38 | 2.21 | 1.38 | 2.21 | 0.571 | 1.56 | -0.24 | 1.20 |
| CV | 3 s | 2.00 | 3.55 | 2.00 | 3.55 | 0.905 | 2.33 | -0.75 | 2.09 |

**moving, < 3 m/s** (117 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.92 | 1.10 | 0.91 | 1.10 | 0.128 | 1.02 | 0.55 | 0.31 |
| IMM | 2 s | 1.36 | 2.05 | 1.34 | 2.02 | 0.402 | 1.76 | 0.71 | 0.71 |
| IMM | 3 s | 1.86 | 3.13 | 1.83 | 3.05 | 0.701 | 2.60 | 0.84 | 1.15 |
| CV | 1 s | 0.92 | 1.11 | 0.92 | 1.11 | 0.128 | 1.01 | 0.55 | 0.33 |
| CV | 2 s | 1.37 | 2.07 | 1.37 | 2.07 | 0.427 | 1.73 | 0.68 | 0.76 |
| CV | 3 s | 1.88 | 3.17 | 1.88 | 3.17 | 0.726 | 2.54 | 0.78 | 1.26 |

**moving, 3-6 m/s** (48 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 1.05 | 1.22 | 1.04 | 1.20 | 0.188 | 1.15 | 0.84 | 0.24 |
| IMM | 2 s | 1.39 | 1.89 | 1.36 | 1.84 | 0.312 | 1.71 | 1.20 | 0.55 |
| IMM | 3 s | 1.75 | 2.68 | 1.69 | 2.56 | 0.479 | 2.28 | 1.42 | 0.96 |
| CV | 1 s | 1.05 | 1.22 | 1.05 | 1.22 | 0.188 | 1.15 | 0.84 | 0.25 |
| CV | 2 s | 1.38 | 1.89 | 1.38 | 1.89 | 0.375 | 1.70 | 1.20 | 0.56 |
| CV | 3 s | 1.75 | 2.68 | 1.75 | 2.68 | 0.500 | 2.27 | 1.42 | 0.98 |

## Training targets

The tier-2 training targets (prediction/dataset.py) of the same samples, cut from this run's track log the way train_predictor.py cuts them, against the labelled future: a pseudo ground truth, smoothed afterwards from the tracker's positions (filtered) or from the matched detections' centers (measured). windows: samples whose track has a training window there. Errors in m; > 1 m: share; along / bias / across as above (moving).

| targets | samples | windows | horizon | mean | p90 | > 1 m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| filtered | all matched | 1847 / 2212 | 1 s | 0.17 | 0.40 | 0.010 | - | - | - |
| filtered | all matched | 1847 / 2212 | 2 s | 0.17 | 0.41 | 0.011 | - | - | - |
| filtered | all matched | 1847 / 2212 | 3 s | 0.18 | 0.43 | 0.012 | - | - | - |
| filtered | moving | 132 / 165 | 1 s | 0.41 | 0.83 | 0.038 | 0.39 | 0.27 | 0.09 |
| filtered | moving | 132 / 165 | 2 s | 0.43 | 0.83 | 0.030 | 0.40 | 0.27 | 0.10 |
| filtered | moving | 132 / 165 | 3 s | 0.47 | 0.89 | 0.068 | 0.44 | 0.31 | 0.11 |
| filtered | static | 1715 / 2047 | 1 s | 0.15 | 0.33 | 0.008 | - | - | - |
| filtered | static | 1715 / 2047 | 2 s | 0.15 | 0.32 | 0.009 | - | - | - |
| filtered | static | 1715 / 2047 | 3 s | 0.16 | 0.35 | 0.008 | - | - | - |
| measured | all matched | 1847 / 2212 | 1 s | 0.13 | 0.28 | 0.006 | - | - | - |
| measured | all matched | 1847 / 2212 | 2 s | 0.13 | 0.28 | 0.005 | - | - | - |
| measured | all matched | 1847 / 2212 | 3 s | 0.15 | 0.33 | 0.008 | - | - | - |
| measured | moving | 132 / 165 | 1 s | 0.16 | 0.34 | 0.008 | 0.15 | -0.04 | 0.03 |
| measured | moving | 132 / 165 | 2 s | 0.16 | 0.37 | 0.000 | 0.15 | -0.04 | 0.04 |
| measured | moving | 132 / 165 | 3 s | 0.18 | 0.40 | 0.008 | 0.17 | -0.03 | 0.05 |
| measured | static | 1715 / 2047 | 1 s | 0.13 | 0.27 | 0.006 | - | - | - |
| measured | static | 1715 / 2047 | 2 s | 0.13 | 0.26 | 0.006 | - | - | - |
| measured | static | 1715 / 2047 | 3 s | 0.14 | 0.32 | 0.008 | - | - | - |

## Hazard proxy

A moving vehicle that comes within 5 m of the robot within 3 s (labelled future, the robot's own path) should raise an alarm: a moving track whose most likely predicted path, or its position now, comes that close. Few such vehicles in 0921; the numbers are for the controlled scenes to come (DATA_COLLECTION_PLAN.md 3.2).

- events 2, reported 1, lead before the closest approach: median 4.5 s, min 4.5 s
- alarms without such a vehicle: 9 (43.9 per hour)

## Timing (this machine, ms per frame: median / p90)

| stage | median | p90 |
|---|---|---|
| filter | 1.21 | 3.26 |
| track | 1.52 | 4.28 |
| predict | 0.76 | 2.85 |

## Issues (`issues.csv`)

| kind | count |
|---|---|
| ID switch | 30 |
| bad prediction | 72 |
| false moving | 8 |
| false track | 131 |
| hazard not reported | 1 |
| lost vehicle | 47 |

## Snapshots

Middle keyframe of each scene, robot at the center heading up. Green / red outlines: scored vehicles tracked / missed; grey: not scored; filled: reported tracks; lines: the two most likely predicted modes; dashes: labelled 3 s future; shaded: the blind sector.

![0921_165659_s000_bev.png](0921_165659_s000_bev.png)
![0921_165659_s001_bev.png](0921_165659_s001_bev.png)
![0921_165659_s002_bev.png](0921_165659_s002_bev.png)
![0921_165659_s007_bev.png](0921_165659_s007_bev.png)
![0921_165659_s008_bev.png](0921_165659_s008_bev.png)
![0921_165659_s009_bev.png](0921_165659_s009_bev.png)
![0921_165659_s010_bev.png](0921_165659_s010_bev.png)
![0921_165659_s011_bev.png](0921_165659_s011_bev.png)
![0921_165659_s012_bev.png](0921_165659_s012_bev.png)
![0921_165659_s013_bev.png](0921_165659_s013_bev.png)
![0921_165659_s014_bev.png](0921_165659_s014_bev.png)
![0921_165659_s015_bev.png](0921_165659_s015_bev.png)
![0921_165659_s016_bev.png](0921_165659_s016_bev.png)
![0921_165659_s017_bev.png](0921_165659_s017_bev.png)
![0921_165659_s018_bev.png](0921_165659_s018_bev.png)
![0921_165659_s019_bev.png](0921_165659_s019_bev.png)
![0921_165659_s020_bev.png](0921_165659_s020_bev.png)
![0921_165659_s021_bev.png](0921_165659_s021_bev.png)
![0921_165659_s022_bev.png](0921_165659_s022_bev.png)
![0921_165659_s023_bev.png](0921_165659_s023_bev.png)
![0921_165659_s024_bev.png](0921_165659_s024_bev.png)
![0921_165659_s025_bev.png](0921_165659_s025_bev.png)
![0921_165659_s026_bev.png](0921_165659_s026_bev.png)
![0921_165659_s027_bev.png](0921_165659_s027_bev.png)
![0921_165659_s028_bev.png](0921_165659_s028_bev.png)
