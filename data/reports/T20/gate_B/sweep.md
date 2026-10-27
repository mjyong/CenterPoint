# Parameter sweep

- scenes: 0921_165659_s000, 0921_165659_s001, 0921_165659_s002, 0921_165659_s007, 0921_165659_s008, 0921_165659_s009, 0921_165659_s010, 0921_165659_s011, 0921_165659_s012, 0921_165659_s013, 0921_165659_s014, 0921_165659_s015, 0921_165659_s016, 0921_165659_s017, 0921_165659_s018, 0921_165659_s019, 0921_165659_s020, 0921_165659_s021, 0921_165659_s022, 0921_165659_s023, 0921_165659_s024, 0921_165659_s025, 0921_165659_s026, 0921_165659_s027, 0921_165659_s028
- detections: `raw` from `data/tune/nusc_pp_train/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=3 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.605 | 0.744 | 5.040 | 32.436 | 0.502 | 0 | 24.388 | 3.138 | 0.642 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2.5 | 0.808 | 0.808 | 0.808 | 0.499 | 0.613 | 0.753 | 4.227 | 32.355 | 0.502 | 0 | 29.265 | 3.139 | 0.647 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=4 | 0.808 | 0.808 | 0.808 | 0.499 | 0.615 | 0.756 | 3.983 | 31.948 | 0.502 | 0 | 39.021 | 3.107 | 0.649 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.012 (4%) | -0.013 (1%) | 1.057 (0%) | 0.406 (21%) | 0.001 (0%) | 0 (0%) | -9.755 (89%) | -0.001 (51%) | -0.003 (56%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.004 (21%) | -0.004 (9%) | 0.244 (0%) | 0.325 (10%) | 0.001 (0%) | 0 (0%) | -4.878 (69%) | -0.000 (55%) | 0.002 (38%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (13%) | -0.001 (31%) | 0.000 (37%) | -0.081 (41%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.032 (93%) | 0.004 (32%) |
