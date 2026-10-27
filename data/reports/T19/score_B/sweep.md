# Parameter sweep

- scenes: 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.845 | 0.738 | 0.788 | 0.762 | 0.526 | 0.732 | 3.902 | 44.061 | 0.501 | 1 | 34.143 | 3.050 | 0.649 |
| 2 | tracker.high_score=0.3 | 0.866 | 0.662 | 0.751 | 1.124 | 0.491 | 0.721 | 4.227 | 52.353 | 0.500 | 1 | 48.776 | 3.024 | 0.636 |
| 3 | tracker.high_score=0.4 | 0.808 | 0.808 | 0.808 | 0.499 | 0.569 | 0.743 | 3.983 | 35.850 | 0.502 | 0 | 39.021 | 3.161 | 0.653 |
| 4 | tracker.high_score=0.45 | 0.761 | 0.868 | 0.811 | 0.311 | 0.625 | 0.765 | 3.739 | 25.282 | 1.000 | 0 | 34.143 | 3.177 | 0.655 |
| 5 | tracker.high_score=0.5 | 0.702 | 0.913 | 0.794 | 0.185 | 0.672 | 0.781 | 3.333 | 17.722 | 1.500 | 0 | 34.143 | 3.125 | 0.641 |
| 6 | tracker.high_score=0.55 | 0.632 | 0.939 | 0.755 | 0.114 | 0.703 | 0.794 | 3.252 | 11.056 | 1.700 | 0 | 29.265 | 3.060 | 0.640 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.021 (100%) | -0.076 (0%) | -0.038 (0%) | 0.362 (0%) | -0.035 (0%) | -0.011 (0%) | 0.325 (12%) | 8.292 (0%) | -0.001 (89%) | 0 (0%) | 14.633 (0%) | -0.026 (83%) | -0.014 (96%) |
| 3 | -0.037 (0%) | 0.070 (100%) | 0.020 (99%) | -0.263 (100%) | 0.043 (100%) | 0.011 (100%) | 0.081 (19%) | -8.211 (100%) | 0.001 (30%) | -1 (66%) | 4.878 (0%) | 0.111 (10%) | 0.004 (34%) |
| 4 | -0.084 (0%) | 0.130 (100%) | 0.023 (93%) | -0.451 (100%) | 0.099 (100%) | 0.033 (100%) | -0.163 (66%) | -18.779 (100%) | 0.499 (12%) | -1 (66%) | 0.000 (0%) | 0.127 (5%) | 0.006 (13%) |
| 5 | -0.143 (0%) | 0.175 (100%) | 0.005 (58%) | -0.577 (100%) | 0.145 (100%) | 0.049 (100%) | -0.569 (99%) | -26.339 (100%) | 0.999 (5%) | -1 (66%) | 0.000 (36%) | 0.075 (0%) | -0.009 (91%) |
| 6 | -0.214 (0%) | 0.200 (100%) | -0.033 (14%) | -0.648 (100%) | 0.177 (100%) | 0.062 (100%) | -0.650 (98%) | -33.005 (100%) | 1.199 (7%) | -1 (66%) | -4.878 (61%) | 0.010 (41%) | -0.010 (90%) |
