# Pipeline evaluation

- scenes (25): 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- 7479 frames, 1476 keyframes scored (12.3 min)
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- parameters: defaults
- scored: vehicles that are not ignore boxes, within 40 m and with current-sweep points; a detection or track on any other labelled box is not counted. Label tracks stitched into one vehicle: 80.
- intervals: 90 % over 500 resamples of the scenes

## Main metrics

| metric | value | interval |
|---|---|---|
| detections: recall within near range | **0.845** | 0.774 – 0.898 |
| detections: precision within near range | **0.738** | 0.610 – 0.815 |
| detections: F1 within near range | **0.788** | 0.700 – 0.842 |
| detections: false per keyframe | **0.762** | 0.588 – 0.934 |
| tracks: MOTA | **0.526** | 0.304 – 0.651 |
| tracks: IDF1 | **0.732** | 0.654 – 0.789 |
| tracks: ID switches / min | **3.902** | 1.536 – 6.449 |
| tracks: false tracks / min | **44.061** | 35.257 – 52.273 |
| tracks: confirm delay p90 [s] | **0.501** | 0.350 – 2.000 |
| motion: relevant moving stretches missed | **1** | 0.000 – 3.000 |
| motion: parked reported moving / h | **34.143** | 9.676 – 68.327 |
| prediction: FDE 3 s, moving [m] | **3.050** | 2.517 – 3.828 |
| prediction: miss rate 3 s, moving | **0.649** | 0.553 – 0.776 |

## Detections at the reporting threshold

Vehicle threshold 0.35 (`tracker.high_score`), 2 m match, keyframes. Near: within 30 m.

| range | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0-30 m | 0.738 | 0.845 | 2036 | 721 | 373 |
| 30-40 m | 0.700 | 0.784 | 939 | 403 | 258 |

False detections per keyframe 0.762. AP over all scores: AP@2.0 0.825, AP@4.0 0.839, AP@2.0 0-20 m 0.872, 20-40 m 0.770.

## Tracking

Reported tracks (confirmed, coasting included) against the scored vehicles, keyframes, 2 m.

| MOTA | MOTP m | IDF1 | recall | precision | IDSW | IDSW / min | FRAG | MT | ML | vehicles | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.526** | 0.23 | **0.732** | 0.880 | 0.721 | 48 | 3.90 | 78 | 113 | 11 | 147 | 3173 | 1227 | 433 |

- false tracks (a track false on consecutive keyframes counts once): 44.06 per minute
- a vehicle coming into 30 m waits for its first matched track: median 0.0 s, p90 0.5 s (63 vehicles; 5 never tracked; 0.5 s resolution)

### Where the errors are

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 1 |
| beside a vehicle (2-4 m off) | 60 |
| coasting after the vehicle was lost | 121 |
| nothing labelled nearby | 1045 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 84 |
| not detected | 349 |
| 0-10 m | 80 |
| 10-20 m | 70 |
| 20-30 m | 138 |
| 30-40 m | 145 |
| 1-9 pts | 80 |
| 10-49 pts | 143 |
| >=50 pts | 210 |

Blind sector (142-218 deg behind the robot) and the 20 deg bands beside it:

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 3192 | 358 | 0.112 | 3936 | 1097 | 0.279 |
| edge | 324 | 52 | 0.160 | 387 | 113 | 0.292 |
| blind | 90 | 23 | 0.256 | 77 | 17 | 0.221 |

## Motion state

Online state against the labels' moving flags (tools/dog/motion_eval.py). Relevant: >= 1 m/s and within 30 m.

| moving recall | false moving | parked reported moving | false episodes (per h) | moving stretches missed | latency median / p90 s | relevant missed | relevant latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.843 | 0.0204 | 7 / 122 | 7 (34.1) | 10 / 30 | 0.50 / 0.55 | **1 / 9** | 0.50 / 0.65 |

## Prediction

Tracks matched to a vehicle with 3 s of labelled future (keyframes every 0.5 s). Moving: >= 1.5 m in 3 s; turning: heading changes > 15 deg. ADE1 / FDE1: the most likely mode; minADE / minFDE: the best; MR: no mode ends within 2 m. IMM is what runs online, CV keeps the track's velocity.

**all matched** (2290 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.34 | 0.39 | 0.34 | 0.39 | 0.024 |
| IMM | 2 s | 0.44 | 0.59 | 0.44 | 0.58 | 0.071 |
| IMM | 3 s | 0.54 | 0.81 | 0.54 | 0.80 | 0.119 |
| CV | 1 s | 0.34 | 0.39 | 0.34 | 0.39 | 0.025 |
| CV | 2 s | 0.44 | 0.59 | 0.44 | 0.59 | 0.074 |
| CV | 3 s | 0.54 | 0.81 | 0.54 | 0.81 | 0.121 |

**moving** (174 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.97 | 1.15 | 0.96 | 1.14 | 0.138 |
| IMM | 2 s | 1.38 | 2.02 | 1.36 | 1.99 | 0.385 |
| IMM | 3 s | 1.85 | 3.05 | 1.81 | 2.96 | 0.649 |
| CV | 1 s | 0.97 | 1.16 | 0.97 | 1.16 | 0.138 |
| CV | 2 s | 1.39 | 2.04 | 1.39 | 2.04 | 0.420 |
| CV | 3 s | 1.86 | 3.08 | 1.86 | 3.08 | 0.672 |

**moving, straight** (130 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 1.00 | 1.18 | 1.00 | 1.18 | 0.162 |
| IMM | 2 s | 1.39 | 1.99 | 1.38 | 1.98 | 0.354 |
| IMM | 3 s | 1.83 | 2.94 | 1.81 | 2.90 | 0.592 |
| CV | 1 s | 1.00 | 1.18 | 1.00 | 1.18 | 0.162 |
| CV | 2 s | 1.39 | 2.00 | 1.39 | 2.00 | 0.369 |
| CV | 3 s | 1.83 | 2.95 | 1.83 | 2.95 | 0.600 |

**moving, turning** (44 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.86 | 1.07 | 0.84 | 1.04 | 0.068 |
| IMM | 2 s | 1.35 | 2.12 | 1.30 | 2.01 | 0.477 |
| IMM | 3 s | 1.92 | 3.37 | 1.82 | 3.13 | 0.818 |
| CV | 1 s | 0.87 | 1.08 | 0.87 | 1.08 | 0.068 |
| CV | 2 s | 1.37 | 2.16 | 1.37 | 2.16 | 0.568 |
| CV | 3 s | 1.96 | 3.46 | 1.96 | 3.46 | 0.886 |

**moving, < 3 m/s** (124 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.12 | 0.93 | 1.12 | 0.121 |
| IMM | 2 s | 1.38 | 2.07 | 1.36 | 2.04 | 0.419 |
| IMM | 3 s | 1.88 | 3.17 | 1.85 | 3.10 | 0.710 |
| CV | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.121 |
| CV | 2 s | 1.39 | 2.09 | 1.39 | 2.09 | 0.444 |
| CV | 3 s | 1.90 | 3.21 | 1.90 | 3.21 | 0.734 |

**moving, 3-6 m/s** (50 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 1.04 | 1.22 | 1.03 | 1.20 | 0.180 |
| IMM | 2 s | 1.40 | 1.92 | 1.36 | 1.86 | 0.300 |
| IMM | 3 s | 1.78 | 2.76 | 1.71 | 2.61 | 0.500 |
| CV | 1 s | 1.04 | 1.22 | 1.04 | 1.22 | 0.180 |
| CV | 2 s | 1.39 | 1.92 | 1.39 | 1.92 | 0.360 |
| CV | 3 s | 1.77 | 2.75 | 1.77 | 2.75 | 0.520 |

## Hazard proxy

A moving vehicle that comes within 5 m of the robot within 3 s (labelled future, the robot's own path) should raise an alarm: a moving track whose most likely predicted path, or its position now, comes that close. Few such vehicles in 0921; the numbers are for the controlled scenes to come (DATA_COLLECTION_PLAN.md 3.2).

- events 2, reported 1, lead before the closest approach: median 4.5 s, min 4.5 s
- alarms without such a vehicle: 7 (34.1 per hour)

## Timing (this machine, ms per frame: median / p90)

| stage | median | p90 |
|---|---|---|
| filter | 1.24 | 3.39 |
| track | 2.25 | 8.30 |
| predict | 1.09 | 3.48 |

## Issues (`issues.csv`)

| kind | count |
|---|---|
| ID switch | 48 |
| bad prediction | 74 |
| false moving | 7 |
| false track | 240 |
| hazard not reported | 1 |
| lost vehicle | 39 |
| missed moving vehicle | 1 |

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
