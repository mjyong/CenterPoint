# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.0+ct:1.0:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.612 | 0.787 | 1.862 | 26.895 | 1.200 | 2 | 86.891 | 2.499 | 0.483 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:2.5+ct:2.5:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.606 | 0.781 | 2.069 | 27.309 | 1.200 | 2 | 74.478 | 3.213 | 0.573 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.792 | 0.521 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.794 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.003 (90%) | 0.001 (90%) | 0.000 (0%) | -0.207 (64%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.296 (100%) | -0.038 (98%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (0%) | -0.006 (0%) | 0.207 (0%) | 0.207 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (67%) | 0.418 (0%) | 0.051 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.003 (89%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.002 (64%) | 0.000 (0%) |
