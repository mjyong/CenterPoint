# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 124.130 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 24.826 | 2.795 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 124.130 | 2.795 | 0.521 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 1 | 198.607 | 2.795 | 0.521 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 3 | 49.652 | 2.795 | 0.521 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 2.795 | 0.521 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 3 | 99.304 | 2.795 | 0.521 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 24.826 | 2.795 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 37.239 (0%) | 0.000 (7%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -62.065 (96%) | 0.000 (7%) | 0.000 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 37.239 (0%) | 0.000 (7%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -1 (69%) | 111.717 (0%) | 0.000 (7%) | 0.000 (0%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | -37.239 (89%) | 0.000 (7%) | 0.000 (0%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (90%) | 0.000 (7%) | 0.000 (0%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | 12.413 (0%) | 0.000 (7%) | 0.000 (0%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -62.065 (96%) | -0.000 (63%) | 0.000 (0%) |
