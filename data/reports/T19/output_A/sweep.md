# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.510 | 0.743 | 2.276 | 43.859 | 1.200 | 2 | 86.891 | 2.851 | 0.537 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.510 | 0.751 | 2.276 | 44.480 | 1.200 | 2 | 74.478 | 2.835 | 0.533 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.509 | 0.755 | 2.069 | 44.273 | 1.200 | 2 | 86.891 | 2.819 | 0.528 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.310 | 0.683 | 2.896 | 50.479 | 1.200 | 1 | 99.304 | 2.753 | 0.522 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.334 | 0.701 | 2.896 | 44.066 | 1.150 | 1 | 86.891 | 2.742 | 0.522 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.346 | 0.707 | 3.103 | 42.411 | 1.150 | 1 | 86.891 | 2.735 | 0.504 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.422 | 0.716 | 2.896 | 48.617 | 1.200 | 1 | 99.304 | 2.780 | 0.527 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.424 | 0.726 | 2.896 | 46.342 | 1.150 | 1 | 86.891 | 2.757 | 0.523 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.420 | 0.729 | 2.689 | 45.307 | 1.150 | 1 | 86.891 | 2.760 | 0.515 |
| 10 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.576 | 0.767 | 2.276 | 36.204 | 1.200 | 2 | 74.478 | 2.797 | 0.530 |
| 11 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.768 | 2.276 | 37.653 | 1.200 | 2 | 74.478 | 2.790 | 0.521 |
| 12 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (49%) | 0.008 (83%) | 0.000 (38%) | 0.621 (18%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.017 (92%) | -0.004 (63%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.001 (45%) | 0.012 (70%) | -0.207 (58%) | 0.414 (31%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.032 (94%) | -0.009 (92%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.199 (0%) | -0.060 (0%) | 0.621 (0%) | 6.620 (14%) | 0.000 (0%) | -1 (67%) | 12.413 (0%) | -0.099 (96%) | -0.015 (97%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.176 (0%) | -0.042 (0%) | 0.621 (13%) | 0.207 (47%) | -0.050 (43%) | -1 (67%) | 0.000 (32%) | -0.109 (96%) | -0.015 (86%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.164 (0%) | -0.036 (1%) | 0.828 (22%) | -1.448 (64%) | -0.050 (43%) | -1 (67%) | 0.000 (32%) | -0.116 (99%) | -0.033 (89%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.088 (0%) | -0.028 (0%) | 0.621 (0%) | 4.758 (3%) | 0.000 (0%) | -1 (67%) | 12.413 (0%) | -0.072 (89%) | -0.010 (88%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.085 (0%) | -0.017 (5%) | 0.621 (13%) | 2.483 (18%) | -0.050 (42%) | -1 (67%) | 0.000 (32%) | -0.094 (96%) | -0.014 (79%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.089 (0%) | -0.015 (17%) | 0.414 (27%) | 1.448 (30%) | -0.050 (42%) | -1 (67%) | 0.000 (32%) | -0.091 (99%) | -0.022 (89%) |
| 10 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.067 (100%) | 0.023 (100%) | 0.000 (0%) | -7.655 (100%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.054 (67%) | -0.007 (67%) |
| 11 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.063 (100%) | 0.024 (100%) | 0.000 (38%) | -6.206 (99%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.061 (92%) | -0.016 (92%) |
| 12 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.063 (100%) | 0.029 (100%) | -0.207 (58%) | -5.793 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.056 (94%) | -0.016 (92%) |
