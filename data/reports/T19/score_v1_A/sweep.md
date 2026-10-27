# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `results` from `data/runs_veh_v1/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.798 | 0.950 | 0.867 | 0.162 | 0.720 | 0.826 | 1.034 | 11.379 | 3.500 | 1 | 74.478 | 2.482 | 0.420 |
| 2 | tracker.high_score=0.2 | 0.856 | 0.858 | 0.857 | 0.560 | 0.711 | 0.822 | 1.034 | 13.034 | 3.500 | 1 | 74.478 | 2.063 | 0.402 |
| 3 | tracker.high_score=0.25 | 0.845 | 0.908 | 0.875 | 0.348 | 0.710 | 0.822 | 1.034 | 14.482 | 3.500 | 1 | 74.478 | 2.280 | 0.414 |
| 4 | tracker.high_score=0.3 | 0.826 | 0.938 | 0.878 | 0.231 | 0.714 | 0.824 | 1.034 | 13.034 | 3.000 | 1 | 74.478 | 2.280 | 0.414 |
| 5 | tracker.high_score=0.4 | 0.759 | 0.958 | 0.847 | 0.121 | 0.721 | 0.824 | 1.241 | 9.930 | 3.500 | 2 | 74.478 | 2.509 | 0.429 |
| 6 | tracker.high_score=0.45 | 0.715 | 0.970 | 0.823 | 0.090 | 0.725 | 0.825 | 1.241 | 8.068 | 3.751 | 2 | 74.478 | 2.508 | 0.429 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.058 (100%) | -0.092 (0%) | -0.010 (18%) | 0.398 (0%) | -0.009 (0%) | -0.005 (0%) | 0.000 (0%) | 1.655 (1%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.419 (100%) | -0.018 (67%) |
| 3 | 0.047 (100%) | -0.041 (0%) | 0.008 (72%) | 0.186 (0%) | -0.010 (0%) | -0.005 (0%) | 0.000 (0%) | 3.103 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.201 (100%) | -0.006 (67%) |
| 4 | 0.028 (99%) | -0.012 (0%) | 0.011 (89%) | 0.069 (0%) | -0.006 (0%) | -0.002 (0%) | 0.000 (0%) | 1.655 (0%) | -0.500 (54%) | 0 (0%) | 0.000 (0%) | -0.201 (100%) | -0.006 (67%) |
| 5 | -0.039 (0%) | 0.009 (100%) | -0.020 (7%) | -0.041 (100%) | 0.001 (66%) | -0.002 (26%) | 0.207 (0%) | -1.448 (99%) | 0.000 (0%) | 1 (0%) | 0.000 (0%) | 0.027 (8%) | 0.009 (0%) |
| 6 | -0.083 (0%) | 0.020 (100%) | -0.044 (7%) | -0.072 (100%) | 0.005 (92%) | -0.001 (41%) | 0.207 (0%) | -3.310 (100%) | 0.250 (0%) | 1 (0%) | 0.000 (0%) | 0.026 (8%) | 0.009 (0%) |
