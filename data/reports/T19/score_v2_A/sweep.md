# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `results` from `data/runs_veh_v2/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.844 | 0.889 | 0.866 | 0.310 | 0.690 | 0.814 | 1.448 | 13.654 | 7.250 | 1 | 62.065 | 2.846 | 0.500 |
| 2 | tracker.high_score=0.2 | 0.896 | 0.786 | 0.838 | 0.814 | 0.684 | 0.805 | 1.655 | 15.723 | 6.750 | 0 | 62.065 | 2.664 | 0.515 |
| 3 | tracker.high_score=0.25 | 0.884 | 0.826 | 0.854 | 0.571 | 0.682 | 0.812 | 1.448 | 16.551 | 6.750 | 1 | 62.065 | 2.655 | 0.500 |
| 4 | tracker.high_score=0.3 | 0.864 | 0.862 | 0.863 | 0.419 | 0.684 | 0.811 | 1.655 | 15.723 | 3.750 | 1 | 62.065 | 2.652 | 0.485 |
| 5 | tracker.high_score=0.4 | 0.821 | 0.921 | 0.868 | 0.210 | 0.701 | 0.818 | 1.448 | 11.379 | 7.750 | 1 | 62.065 | 2.857 | 0.495 |
| 6 | tracker.high_score=0.45 | 0.795 | 0.953 | 0.867 | 0.129 | 0.714 | 0.826 | 1.448 | 8.068 | 1.300 | 0 | 62.065 | 2.883 | 0.500 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.052 (100%) | -0.103 (0%) | -0.029 (4%) | 0.503 (0%) | -0.006 (1%) | -0.009 (5%) | 0.207 (0%) | 2.069 (0%) | -0.499 (52%) | -1 (67%) | 0.000 (0%) | -0.182 (62%) | 0.015 (10%) |
| 3 | 0.040 (100%) | -0.064 (0%) | -0.012 (14%) | 0.260 (0%) | -0.008 (0%) | -0.002 (2%) | 0.000 (0%) | 2.896 (0%) | -0.499 (66%) | 0 (0%) | 0.000 (0%) | -0.191 (77%) | 0.000 (28%) |
| 4 | 0.020 (100%) | -0.027 (0%) | -0.003 (28%) | 0.109 (0%) | -0.005 (0%) | -0.003 (2%) | 0.207 (0%) | 2.069 (0%) | -3.500 (66%) | 0 (0%) | 0.000 (0%) | -0.194 (97%) | -0.015 (92%) |
| 5 | -0.023 (0%) | 0.031 (100%) | 0.002 (59%) | -0.100 (100%) | 0.011 (100%) | 0.004 (97%) | 0.000 (0%) | -2.276 (100%) | 0.500 (0%) | 0 (0%) | 0.000 (0%) | 0.011 (28%) | -0.005 (65%) |
| 6 | -0.049 (0%) | 0.063 (100%) | 0.001 (51%) | -0.181 (100%) | 0.025 (100%) | 0.012 (100%) | 0.000 (0%) | -5.586 (100%) | -5.950 (60%) | -1 (67%) | 0.000 (0%) | 0.037 (22%) | 0.000 (36%) |
