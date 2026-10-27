# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.0 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=0.5 | 0.839 | 0.825 | 0.832 | 0.643 | 0.555 | 0.726 | 4.345 | 39.721 | 1.200 | 3 | 49.652 | 2.777 | 0.517 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.575 | 0.773 | 1.862 | 36.825 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=2.0 | 0.839 | 0.827 | 0.833 | 0.634 | 0.578 | 0.774 | 1.448 | 36.204 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=3.0 | 0.839 | 0.831 | 0.835 | 0.622 | 0.581 | 0.778 | 1.655 | 35.998 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.018 (0%) | -0.046 (0%) | 2.276 (0%) | 1.655 (1%) | 0.000 (0%) | 1 (0%) | -37.239 (67%) | -0.018 (67%) | -0.004 (67%) |
| 3 | 0.000 (0%) | 0.001 (65%) | 0.000 (65%) | -0.002 (65%) | 0.003 (100%) | 0.001 (100%) | -0.207 (67%) | -1.241 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |
| 4 | 0.000 (0%) | 0.002 (98%) | 0.001 (98%) | -0.009 (98%) | 0.006 (100%) | 0.002 (64%) | -0.621 (76%) | -1.862 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |
| 5 | 0.000 (0%) | 0.006 (100%) | 0.003 (100%) | -0.021 (100%) | 0.009 (95%) | 0.006 (71%) | -0.414 (62%) | -2.069 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |
