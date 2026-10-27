# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=0.8 | 0.838 | 0.826 | 0.832 | 0.612 | 0.578 | 0.773 | 1.862 | 36.411 | 1.200 | 2 | 86.891 | 2.763 | 0.530 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.6 | 0.839 | 0.825 | 0.832 | 0.655 | 0.573 | 0.773 | 1.862 | 37.860 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.2 | 0.801 | 0.826 | 0.813 | 0.622 | 0.541 | 0.750 | 3.103 | 37.446 | 1.200 | 2 | 86.891 | 2.845 | 0.539 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=0.8 | 0.800 | 0.827 | 0.813 | 0.591 | 0.546 | 0.750 | 2.896 | 35.998 | 1.200 | 2 | 86.891 | 2.809 | 0.539 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.6 | 0.801 | 0.826 | 0.813 | 0.634 | 0.541 | 0.751 | 2.896 | 37.239 | 1.200 | 2 | 86.891 | 2.845 | 0.539 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.2 | 0.849 | 0.808 | 0.828 | 0.707 | 0.542 | 0.758 | 2.483 | 40.756 | 1.200 | 2 | 74.478 | 2.802 | 0.533 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=0.8 | 0.849 | 0.809 | 0.829 | 0.676 | 0.547 | 0.759 | 2.276 | 39.721 | 1.200 | 2 | 74.478 | 2.770 | 0.542 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.6 | 0.849 | 0.808 | 0.828 | 0.719 | 0.542 | 0.760 | 2.276 | 40.549 | 1.200 | 2 | 74.478 | 2.802 | 0.533 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.001 (0%) | 0.002 (90%) | 0.000 (68%) | -0.031 (100%) | 0.006 (88%) | 0.001 (68%) | -0.207 (67%) | -1.655 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.032 (86%) | 0.009 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.012 (0%) | 0.001 (57%) | 0.001 (80%) | -0.207 (67%) | -0.207 (64%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.000 (65%) | 0.000 (0%) |
| 4 | -0.038 (0%) | 0.001 (59%) | -0.019 (0%) | -0.021 (100%) | -0.031 (6%) | -0.022 (2%) | 1.034 (0%) | -0.621 (76%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.050 (3%) | 0.018 (0%) |
| 5 | -0.039 (0%) | 0.002 (78%) | -0.018 (2%) | -0.052 (100%) | -0.027 (13%) | -0.022 (5%) | 0.828 (13%) | -2.069 (96%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.014 (38%) | 0.018 (0%) |
| 6 | -0.038 (0%) | 0.001 (59%) | -0.019 (0%) | -0.009 (94%) | -0.031 (4%) | -0.021 (2%) | 0.828 (13%) | -0.828 (79%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.050 (3%) | 0.018 (0%) |
| 7 | 0.011 (98%) | -0.017 (0%) | -0.004 (35%) | 0.064 (0%) | -0.031 (9%) | -0.014 (3%) | 0.414 (0%) | 2.689 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | 0.007 (53%) | 0.012 (0%) |
| 8 | 0.010 (98%) | -0.015 (0%) | -0.003 (37%) | 0.033 (21%) | -0.025 (15%) | -0.013 (1%) | 0.207 (0%) | 1.655 (7%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | -0.025 (90%) | 0.020 (0%) |
| 9 | 0.011 (98%) | -0.017 (0%) | -0.004 (35%) | 0.076 (0%) | -0.030 (10%) | -0.012 (6%) | 0.207 (0%) | 2.483 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | 0.007 (54%) | 0.012 (0%) |
