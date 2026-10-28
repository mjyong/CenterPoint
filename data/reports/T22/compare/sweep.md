# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/final/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35, tracker.output_min_score=0.3, tracker.low_score=0.1, tracker.vehicle.max_age=1.0, tracker.vehicle.min_hits=2 | 0.872 | 0.758 | 0.811 | 1.040 | 0.459 | 0.726 | 2.483 | 58.134 | 1.200 | 2 | 86.891 | 2.897 | 0.541 |
| 2 | defaults | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.033 (0%) | 0.068 (100%) | 0.021 (99%) | -0.398 (100%) | 0.150 (100%) | 0.060 (100%) | -0.621 (82%) | -31.032 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.102 (95%) | -0.020 (67%) |
