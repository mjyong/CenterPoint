# Parameter sweep

- scenes: 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=greedy | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=hungarian | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.274 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=greedy | 0.808 | 0.809 | 0.808 | 0.497 | 0.649 | 0.768 | 3.821 | 22.843 | 1.500 | 0 | 614.574 | 2.737 | 0.539 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=hungarian | 0.808 | 0.809 | 0.808 | 0.497 | 0.649 | 0.767 | 3.821 | 22.925 | 1.500 | 0 | 629.207 | 2.737 | 0.539 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.001 (0%) | -0.000 (0%) | 0.000 (0%) | 0.081 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.009 (2%) | -0.017 (1%) | 1.382 (0%) | 0.650 (9%) | 0.448 (0%) | 0 (0%) | 575.553 (0%) | -0.261 (74%) | -0.097 (85%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.010 (1%) | -0.017 (0%) | 1.382 (0%) | 0.732 (6%) | 0.448 (0%) | 0 (0%) | 590.186 (0%) | -0.261 (74%) | -0.097 (85%) |
