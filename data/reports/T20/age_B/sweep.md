# Parameter sweep

- scenes: 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.0 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=0.5 | 0.808 | 0.807 | 0.808 | 0.501 | 0.604 | 0.706 | 7.560 | 32.761 | 0.501 | 0 | 34.143 | 3.168 | 0.647 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.624 | 0.771 | 2.764 | 31.216 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=2.0 | 0.808 | 0.809 | 0.808 | 0.496 | 0.626 | 0.770 | 2.683 | 31.135 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=3.0 | 0.808 | 0.809 | 0.808 | 0.493 | 0.630 | 0.779 | 2.195 | 30.647 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | -0.001 (0%) | -0.001 (0%) | 0.003 (0%) | -0.014 (0%) | -0.051 (0%) | 3.577 (0%) | 0.732 (4%) | -0.000 (40%) | 0 (0%) | 0.000 (0%) | 0.029 (0%) | 0.002 (0%) |
| 3 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.002 (98%) | 0.006 (100%) | 0.014 (100%) | -1.219 (100%) | -0.813 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |
| 4 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.003 (99%) | 0.008 (100%) | 0.013 (96%) | -1.301 (100%) | -0.894 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |
| 5 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.005 (100%) | 0.012 (100%) | 0.022 (99%) | -1.788 (100%) | -1.382 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |
