# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=greedy | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=hungarian | 0.839 | 0.825 | 0.832 | 0.641 | 0.610 | 0.787 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=greedy | 0.839 | 0.825 | 0.832 | 0.641 | 0.623 | 0.788 | 1.862 | 26.274 | 1.250 | 1 | 484.105 | 1.773 | 0.331 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=hungarian | 0.839 | 0.825 | 0.832 | 0.641 | 0.624 | 0.788 | 1.862 | 26.274 | 1.250 | 1 | 484.105 | 1.773 | 0.331 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.001 (64%) | 0.000 (64%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.014 (99%) | 0.002 (58%) | 0.000 (33%) | -0.828 (84%) | 0.050 (26%) | -1 (67%) | 397.215 (0%) | -1.022 (100%) | -0.191 (100%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.015 (100%) | 0.002 (60%) | 0.000 (33%) | -0.828 (84%) | 0.050 (26%) | -1 (67%) | 397.215 (0%) | -1.022 (100%) | -0.191 (100%) |
