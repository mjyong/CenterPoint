# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `labels` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=1.0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.389 | 0.286 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.000 (55%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.001 (0%) | 0.000 (0%) |
