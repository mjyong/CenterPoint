# Yield gate, offline

Gate: classes=('vehicle', 'cyclist'), dog_speed=1.0, dog_length=1.0, dog_width=0.4, lateral_margin=0.5, vehicle_margin=0.5, sigma_k=1.0, min_mode_prob=0.1, horizon=8.0, sigma_growth=0.3, sigma_max=1.0, dt=0.1, path_step=0.1, path_length=10.0, pet=2.0, clear_time=2.0, stopped_speed=0.5, stopped_wait=3.0, start_zone=10.0, stale_after=0.3, young_age=1.0, young_range=0.0, turn_range=0.0, turn_radius=10.0, turn_max_deg=90.0
Evaluation: arrive_every=1.0, warmup=3.0, max_wait=30.0, latency=0.15, tau_true=1.0, true_margin=0.3, min_speed=0.3, max_gap=1.1, grid_dt=0.1, classes=('vehicle', 'cyclist'), blind_sector=(142.0, 218.0), blind_margin=15.0, recent=1.5, far=35.0, IMM horizon 3.0 s

- 1104_cp01_s000: 306 frames, 31 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 23.2 / 25.0 km/h); from >= 25 m: 2, first tracked at 35.0 m (median)
- 1104_cp01_s001: 307 frames, 30 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 29.1 / 32.5 km/h); from >= 25 m: 1, first tracked at 45.2 m (median)
- 1104_cp01_s002: 306 frames, 31 s labelled, robot moved up to 0.01 m; 2 labelled vehicles moving within 30 m (top speed median / max 18.2 / 19.3 km/h); from >= 25 m: 2, first tracked at 44.0 m (median)
- 1104_cp01_s003: 306 frames, 31 s labelled, robot moved up to 0.01 m; 4 labelled vehicles moving within 30 m (top speed median / max 21.7 / 27.8 km/h); from >= 25 m: 2, first tracked at 36.0 m (median)
- 1104_cp02_s000: 285 frames, 28 s labelled, robot moved up to 0.01 m; 5 labelled vehicles moving within 30 m (top speed median / max 24.2 / 26.8 km/h); from >= 25 m: 4, first tracked at 43.8 m (median)
- 1104_cp02_s001: 285 frames, 28 s labelled, robot moved up to 0.02 m; 2 labelled vehicles moving within 30 m (top speed median / max 23.3 / 26.4 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp02_s002: 286 frames, 28 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 8.0 / 10.0 km/h); from >= 25 m: 2, first tracked at 46.6 m (median)
- 1104_cp02_s003: 285 frames, 28 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 11.4 / 21.5 km/h); from >= 25 m: 1, first tracked at 37.7 m (median)
- 1104_cp02_s004: 285 frames, 28 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 23.3 / 26.4 km/h); from >= 25 m: 2, first tracked at 45.5 m (median)
- 1104_cp02_s005: 285 frames, 28 s labelled, robot moved up to 0.01 m; 5 labelled vehicles moving within 30 m (top speed median / max 20.0 / 27.6 km/h); from >= 25 m: 4, first tracked at 40.9 m (median)
- 1104_cp03_s000: 305 frames, 30 s labelled, robot moved up to 0.24 m; 1 labelled vehicles moving within 30 m (top speed median / max 13.9 / 13.9 km/h); from >= 25 m: 1, first tracked at 49.7 m (median)
- 1104_cp03_s001: 304 frames, 30 s labelled, robot moved up to 0.01 m; 7 labelled vehicles moving within 30 m (top speed median / max 19.4 / 26.6 km/h); from >= 25 m: 6, first tracked at 43.1 m (median)
- 1104_cp03_s002: 305 frames, 30 s labelled, robot moved up to 0.02 m; 5 labelled vehicles moving within 30 m (top speed median / max 20.5 / 24.3 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp03_s004: 305 frames, 30 s labelled, robot moved up to 0.01 m; 5 labelled vehicles moving within 30 m (top speed median / max 22.1 / 24.6 km/h); from >= 25 m: 2, first tracked at 30.2 m (median)
- 1104_cp03_s005: 304 frames, 30 s labelled, robot moved up to 0.01 m; 2 labelled vehicles moving within 30 m (top speed median / max 24.9 / 25.8 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp03_s008: 304 frames, 30 s labelled, robot moved up to 0.00 m; 2 labelled vehicles moving within 30 m (top speed median / max 22.3 / 24.2 km/h); from >= 25 m: 2, first tracked at 48.2 m (median)
- 1104_cp03_s009: 305 frames, 30 s labelled, robot moved up to 0.01 m; 1 labelled vehicles moving within 30 m (top speed median / max 22.3 / 22.3 km/h); from >= 25 m: 1, first tracked at 49.3 m (median)
- 1104_cp03_s010: 304 frames, 30 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 23.3 / 26.6 km/h); from >= 25 m: 1, first tracked at 47.9 m (median)
- 1104_cp03_s011: 305 frames, 30 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 25.0 / 26.0 km/h); from >= 25 m: 1, first tracked at 39.8 m (median)
- 1104_cp03_s012: 304 frames, 30 s labelled, robot moved up to 0.00 m; 1 labelled vehicles moving within 30 m (top speed median / max 19.3 / 19.3 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp03_s013: 305 frames, 30 s labelled, robot moved up to 0.01 m; 1 labelled vehicles moving within 30 m (top speed median / max 26.0 / 26.0 km/h); from >= 25 m: 1, first tracked at 47.2 m (median)
- 1104_cp03_s014: 304 frames, 30 s labelled, robot moved up to 0.00 m; 1 labelled vehicles moving within 30 m (top speed median / max 16.8 / 16.8 km/h); from >= 25 m: 1, first tracked at 45.2 m (median)
- 1104_cp03_s015: 305 frames, 30 s labelled, robot moved up to 0.02 m; 3 labelled vehicles moving within 30 m (top speed median / max 24.4 / 27.4 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp03_s016: 304 frames, 30 s labelled, robot moved up to 0.01 m; 4 labelled vehicles moving within 30 m (top speed median / max 25.4 / 25.8 km/h); from >= 25 m: 1, first tracked at 37.0 m (median)
- 1104_cp04_s000: 306 frames, 30 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 24.3 / 30.1 km/h); from >= 25 m: 1, first tracked at 39.9 m (median)
- 1104_cp04_s001: 306 frames, 30 s labelled, robot moved up to 0.00 m; 0 labelled vehicles moving within 30 m (top speed median / max – / – km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp04_s002: 306 frames, 31 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 22.1 / 24.9 km/h); from >= 25 m: 3, first tracked at 46.5 m (median)
- 1104_cp04_s003: 306 frames, 30 s labelled, robot moved up to 0.01 m; 5 labelled vehicles moving within 30 m (top speed median / max 26.0 / 32.2 km/h); from >= 25 m: 3, first tracked at 47.6 m (median)
- 1104_cp04_s004: 306 frames, 30 s labelled, robot moved up to 0.01 m; 7 labelled vehicles moving within 30 m (top speed median / max 26.8 / 31.9 km/h); from >= 25 m: 4, first tracked at 48.0 m (median)
- 1104_cp04_s005: 305 frames, 30 s labelled, robot moved up to 0.01 m; 3 labelled vehicles moving within 30 m (top speed median / max 19.2 / 27.6 km/h); from >= 25 m: 0, first tracked at – m (median)
- 1104_cp04_s006: 306 frames, 31 s labelled, robot moved up to 0.00 m; 5 labelled vehicles moving within 30 m (top speed median / max 16.3 / 21.1 km/h); from >= 25 m: 3, first tracked at 39.1 m (median)
- 1104_cp04_s007: 306 frames, 30 s labelled, robot moved up to 0.01 m; 5 labelled vehicles moving within 30 m (top speed median / max 12.6 / 26.0 km/h); from >= 25 m: 0, first tracked at – m (median)

## Per path

crossing 8.5 s (8.0 m + robot 1.0 m at 1.0 m/s); unsafe: labelled PET < 1.0 s; risky: went where the oracle (labels, gate margins, pet 2.0 s) would wait; waits in s; extra = gate - oracle

| path | episodes | unknown | **unsafe** | risky | timeouts | wait median / p90 / max | oracle wait median / p90 | oracle waited | extra median / p90 / max | parked in path |
|---|---|---|---|---|---|---|---|---|---|---|
| fan000 | 504 | 380 | **35** | 49 | 0 | 0.0 / 0.0 / 11.8 | 0.0 / 0.0 | 48 | 0.0 / 0.0 / 11.8 | 1% |
| fan045 | 512 | 372 | **63** | 100 | 0 | 0.0 / 0.0 / 8.5 | 0.0 / 2.4 | 69 | 0.0 / 0.0 / 8.5 | 1% |
| fan090 | 512 | 372 | **81** | 108 | 0 | 0.0 / 0.0 / 7.7 | 0.0 / 2.1 | 64 | 0.0 / 0.0 / 5.9 | 1% |
| fan135 | 523 | 361 | **46** | 73 | 0 | 0.0 / 0.0 / 6.5 | 0.0 / 0.0 | 33 | 0.0 / 0.0 / 6.5 | 0% |
| fan180 | 528 | 356 | **3** | 8 | 0 | 0.0 / 0.0 / 5.1 | 0.0 / 0.0 | 9 | 0.0 / 0.0 / 5.1 | 0% |
| fan225 | 526 | 358 | **0** | 3 | 0 | 0.0 / 0.0 / 6.1 | 0.0 / 0.0 | 5 | 0.0 / 0.0 / 6.1 | 0% |
| fan270 | 521 | 363 | **0** | 3 | 0 | 0.0 / 0.0 / 8.2 | 0.0 / 0.0 | 5 | 0.0 / 0.0 / 8.2 | 0% |
| fan315 | 508 | 376 | **2** | 6 | 0 | 0.0 / 0.0 / 13.2 | 0.0 / 0.0 | 5 | 0.0 / 0.0 / 13.2 | 0% |
| all | 4134 | 2938 | **230** | 350 | 0 | 0.0 / 0.0 / 13.2 | 0.0 / 0.0 | 238 | 0.0 / 0.0 / 13.2 | 0% |

## Why unsafe

blind / far / near: the vehicle was first labelled less than 1.5 s before the robot went, or after: out of the rear blind sector (142-218 deg +- 15), from 35 m or more, or closer (round a corner); occluded: labelled before but not when it went; detection: labelled, not tracked; prediction: tracked, no conflict seen

| path | unsafe | blind | far | near | occluded | detection | prediction |
|---|---|---|---|---|---|---|---|
| fan000 | 35 | 30 | 0 | 0 | 0 | 2 | 3 |
| fan045 | 63 | 58 | 0 | 0 | 0 | 2 | 3 |
| fan090 | 81 | 78 | 2 | 0 | 0 | 1 | 0 |
| fan135 | 46 | 46 | 0 | 0 | 0 | 0 | 0 |
| fan180 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| fan225 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan315 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| all | 230 | 217 | 2 | 0 | 0 | 5 | 6 |

## Unsafe episodes (230)

- 1104_cp01_s000 fan000: arrived 11.1 s, went after 0.0 s; labelled track 7 came 0.85 s after the robot left; then 43 m away at 129 deg, 21 km/h, NOT tracked; first labelled -0.9 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan045: arrived 10.1 s, went after 0.0 s; labelled track 7 came 0.15 s after the robot left; not labelled then; first labelled +0.1 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan045: arrived 11.1 s, went after 0.0 s; labelled track 7 came -0.85 s after the robot left; then 43 m away at 129 deg, 21 km/h, NOT tracked; first labelled -0.9 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan090: arrived 8.0 s, went after 0.0 s; labelled track 7 came 0.35 s after the robot left; not labelled then; first labelled +2.1 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan090: arrived 9.1 s, went after 0.0 s; labelled track 7 came -0.65 s after the robot left; not labelled then; first labelled +1.1 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan090: arrived 10.1 s, went after 0.0 s; labelled track 7 came -0.95 s after the robot left; not labelled then; first labelled +0.1 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s000 fan090: arrived 11.1 s, went after 0.0 s; labelled track 7 came -0.95 s after the robot left; then 43 m away at 129 deg, 21 km/h, NOT tracked; first labelled -0.9 s from the go at 49 m, 130 deg [blind]
- 1104_cp01_s002 fan045: arrived 3.0 s, went after 0.0 s; labelled track 2 came -0.85 s after the robot left; then 38 m away at 25 deg, 14 km/h, tracked; first labelled -2.4 s from the go at 46 m, 29 deg [prediction]
- 1104_cp01_s002 fan045: arrived 4.0 s, went after 0.0 s; labelled track 2 came -0.95 s after the robot left; then 34 m away at 22 deg, 14 km/h, tracked; first labelled -3.4 s from the go at 46 m, 29 deg [prediction]
- 1104_cp02_s000 fan000: arrived 13.1 s, went after 0.0 s; labelled track 6 came 0.65 s after the robot left; not labelled then; first labelled +1.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan000: arrived 14.1 s, went after 0.0 s; labelled track 6 came -0.35 s after the robot left; not labelled then; first labelled +0.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan000: arrived 15.1 s, went after 0.0 s; labelled track 6 came -0.95 s after the robot left; then 46 m away at 134 deg, 22 km/h, tracked; first labelled -0.9 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan045: arrived 14.1 s, went after 0.0 s; labelled track 6 came 0.25 s after the robot left; not labelled then; first labelled +0.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan045: arrived 15.1 s, went after 0.0 s; labelled track 6 came -0.75 s after the robot left; then 46 m away at 134 deg, 22 km/h, tracked; first labelled -0.9 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan090: arrived 11.1 s, went after 0.0 s; labelled track 6 came 0.35 s after the robot left; not labelled then; first labelled +3.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan090: arrived 12.1 s, went after 0.0 s; labelled track 6 came -0.65 s after the robot left; not labelled then; first labelled +2.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan090: arrived 13.1 s, went after 0.0 s; labelled track 6 came -0.95 s after the robot left; not labelled then; first labelled +1.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan090: arrived 14.1 s, went after 0.0 s; labelled track 6 came -0.95 s after the robot left; not labelled then; first labelled +0.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan090: arrived 15.1 s, went after 0.0 s; labelled track 6 came -0.95 s after the robot left; then 46 m away at 134 deg, 22 km/h, tracked; first labelled -0.9 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan315: arrived 14.1 s, went after 0.0 s; labelled track 6 came 0.85 s after the robot left; not labelled then; first labelled +0.1 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s000 fan315: arrived 15.1 s, went after 0.0 s; labelled track 6 came -0.15 s after the robot left; then 46 m away at 134 deg, 22 km/h, tracked; first labelled -0.9 s from the go at 51 m, 134 deg [blind]
- 1104_cp02_s003 fan000: arrived 16.0 s, went after 0.0 s; labelled track 10 came 0.95 s after the robot left; not labelled then; first labelled +2.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan000: arrived 17.0 s, went after 0.0 s; labelled track 10 came -0.05 s after the robot left; not labelled then; first labelled +1.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan045: arrived 15.0 s, went after 0.0 s; labelled track 10 came 0.65 s after the robot left; not labelled then; first labelled +3.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan045: arrived 16.0 s, went after 0.0 s; labelled track 10 came -0.35 s after the robot left; not labelled then; first labelled +2.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan045: arrived 17.0 s, went after 0.0 s; labelled track 10 came -0.95 s after the robot left; not labelled then; first labelled +1.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan090: arrived 14.0 s, went after 0.0 s; labelled track 10 came 0.35 s after the robot left; not labelled then; first labelled +4.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan090: arrived 15.0 s, went after 0.0 s; labelled track 10 came -0.65 s after the robot left; not labelled then; first labelled +3.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan090: arrived 16.0 s, went after 0.0 s; labelled track 10 came -0.95 s after the robot left; not labelled then; first labelled +2.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s003 fan090: arrived 17.0 s, went after 0.0 s; labelled track 10 came -0.95 s after the robot left; not labelled then; first labelled +1.6 s from the go at 38 m, 129 deg [blind]
- 1104_cp02_s004 fan045: arrived 13.1 s, went after 0.0 s; labelled track 5 came 0.55 s after the robot left; not labelled then; first labelled +1.6 s from the go at 46 m, 130 deg [blind]
- 1104_cp02_s004 fan090: arrived 12.1 s, went after 0.0 s; labelled track 5 came 0.35 s after the robot left; not labelled then; first labelled +2.6 s from the go at 46 m, 130 deg [blind]
- 1104_cp02_s004 fan090: arrived 13.1 s, went after 0.0 s; labelled track 5 came -0.65 s after the robot left; not labelled then; first labelled +1.6 s from the go at 46 m, 130 deg [blind]
- 1104_cp03_s001 fan090: arrived 17.1 s, went after 0.0 s; labelled track 24 came 0.85 s after the robot left; not labelled then; first labelled +6.1 s from the go at 13 m, 137 deg [blind]
- 1104_cp03_s001 fan135: arrived 14.1 s, went after 0.0 s; labelled track 24 came 0.55 s after the robot left; not labelled then; first labelled +9.1 s from the go at 13 m, 137 deg [blind]
- 1104_cp03_s001 fan135: arrived 15.1 s, went after 0.0 s; labelled track 24 came -0.45 s after the robot left; not labelled then; first labelled +8.1 s from the go at 13 m, 137 deg [blind]
- 1104_cp03_s001 fan135: arrived 16.1 s, went after 0.0 s; labelled track 24 came -0.95 s after the robot left; not labelled then; first labelled +7.1 s from the go at 13 m, 137 deg [blind]
- 1104_cp03_s001 fan135: arrived 17.1 s, went after 0.0 s; labelled track 24 came -0.95 s after the robot left; not labelled then; first labelled +6.1 s from the go at 13 m, 137 deg [blind]
- 1104_cp03_s004 fan000: arrived 3.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +3.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan000: arrived 4.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +2.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan000: arrived 5.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +1.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan000: arrived 6.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +0.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan000: arrived 8.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; then 11 m away at 128 deg, 18 km/h, tracked; first labelled -1.4 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan045: arrived 3.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +3.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan045: arrived 4.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +2.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan045: arrived 5.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +1.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan045: arrived 6.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +0.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan090: arrived 3.0 s, went after 2.5 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +1.1 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan090: arrived 4.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +2.6 s from the go at 17 m, 140 deg [blind]
- 1104_cp03_s004 fan090: arrived 5.0 s, went after 0.0 s; labelled track 8 came -0.95 s after the robot left; not labelled then; first labelled +1.6 s from the go at 17 m, 140 deg [blind]

## Traffic

Labelled vehicles moving within 30 m: 103; top speed median 22.1, p90 27.3, max 32.5 km/h; above 15 km/h: 83, above 20 km/h: 69

Vehicles coming from >= 25 m: 51; never tracked 0; first tracked at median 44.7 m, p10 27.3 m, min 21.6 m

## Sight range needed for a vehicle nobody has seen yet

crossing 8.0 m at 1.0 m/s, gap 2.0 s, latency 1.1 s (tracker p90 1.0 s + node + planning)

| vehicle speed | needed sight range |
|---|---|
| 10 km/h | 40 m |
| 15 km/h | 56 m |
| 20 km/h | 72 m |
| 30 km/h | 105 m |

7072 episodes -> data/reports/T45/episodes_t44.csv
